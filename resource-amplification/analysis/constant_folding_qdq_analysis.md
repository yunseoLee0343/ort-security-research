# ORT Constant Folding / Materialization / QDQ 감사 세션 정리

## 0. 세션 전체 목표

이번 세션의 핵심 목표는 ONNX Runtime(ORT)의 정적 그래프 최적화 및 런타임 실행 경계에서 발생할 수 있는 **Verification-Execution Gap**을 소스 코드 수준에서 추적하는 것이었다. 특히 ORT의 `ConstantFolding`, CPU EP의 `Expand`/`ConstantOfShape` materialization kernel, `Tile`의 byte-budget guard, `CPUAllocator`, 그리고 QDQ 최적화 패스(`qdq_propagation.cc`, `qdq_util.cc`)를 실제 소스 기준으로 분석했다.

분석의 중심 질문은 다음이었다.

- ORT는 어떤 조건에서 노드를 compile-time constant로 접는가?
- `AllNodeInputsAreConstant` 조건을 만족하지 않는 runtime-input-dependent graph는 어떻게 실행 레이어까지 살아남는가?
- `Expand`와 `ConstantOfShape`는 output tensor를 언제 물리적으로 materialize하는가?
- `Tile`에는 출력 바이트 상한이 있는데 왜 `Expand`/`ConstantOfShape`에는 같은 runtime guard가 없는가?
- QDQ propagation은 `Reshape`, `Transpose`, `Unsqueeze`, `Slice` 같은 layout adapter를 통과하면서 materialization boundary 근처에서 어떤 구조적 위험을 만든다?
- Hugging Face의 well-known 모델 코드가 ONNX lowering을 거칠 때 어떤 실제 PyTorch 연산이 ORT `Expand`/`Reshape`/`ConstantOfShape`/QDQ 경계와 연결되는가?
- 과거 ORT GitHub PR/Issue에서 이 문제가 어떤 방식으로 부분 수정되었고, 어떤 부분이 여전히 남아 있는가?

---

## 1. `GenericConstantFoldable(n)` 명제식과 ORT 실제 구현 매핑

세션 중 추상 수식으로 다음을 정의했다.

```text
GenericConstantFoldable(n) =
  node exists
  ∧ AllowConstantFolding(n)
  ∧ IsSupportedProvider(n)
  ∧ IsOperationDeterministic(domain, op)
  ∧ !ContainsSubgraph(n)
  ∧ AllNodeInputsAreConstant(graph, n, ...)
  ∧ EstimatedOutputSizeKnown(n)
  ∧ EstimatedOutputSize(n) <= configured_limit
  ∧ CPU kernel exists
  ∧ kernel->Compute succeeds
  ∧ actual output size <= configured_limit
```

실제 ORT `onnxruntime/core/optimizer/constant_folding.cc`의 `ConstantFolding::ApplyImpl()`를 확인한 결과, 이 식은 ORT 소스의 단일 boolean expression은 아니지만, 여러 `continue` 기반 short-circuit 조건을 추상화한 모델로는 정확하다.

실제 구현상 주요 단계는 다음과 같다.

### 1.1 Pre-check Stage

`graph.GetNode(i)`로 node pointer를 얻고, `!node || !AllowConstantFolding(*node)`이면 후보에서 제외한다. 이후 일반 op folding 경로에서는 `can_constant_fold_node` 람다를 통해 다음 조건을 검사한다.

```cpp
graph_utils::IsSupportedProvider(n, GetCompatibleExecutionProviders()) &&
optimizer_utils::IsOperationDeterministic(n.Domain(), n.OpType()) &&
!n.ContainsSubgraph() &&
(skip_inputs_constant_check ||
 graph_utils::AllNodeInputsAreConstant(graph, n, constant_inputs, excluded_initializers_))
```

이 조건들은 각각 provider 호환성, 결정론성, subgraph-free 여부, 입력 initializer/constant 여부를 검사한다.

### 1.2 Output-size Pre-check

`GetConstantFoldingMaxOutputSize(config_options_)`로 configured limit을 얻고, `EstimateNodeOutputSizeInBytes(*node)`로 출력 크기를 예측한다. 결과가 음수이면 추정 불가로 보고 folding을 skip한다. 예측값이 limit보다 크면 folding하지 않는다. `SafeInt` overflow가 발생해도 skip한다.

### 1.3 Kernel Creation Stage

`OptimizerExecutionFrame::Info`를 만들고, output fetch index를 수집한다. 그 다음 CPU kernel을 생성한다. 노드가 CPU EP에 있지 않으면 일시적으로 CPU EP로 바꿔 `info.CreateKernel(node, config_options_)`를 호출한 뒤 원래 EP로 되돌린다. 생성된 `kernel == nullptr`이면 folding하지 않는다.

### 1.4 Compute Stage

`OptimizerExecutionFrame`과 `OpKernelContext`를 만들고 `kernel->Compute(&op_kernel_context)`를 실행한다. 이 호출은 `try/catch`로 감싸져 있다. exception 또는 non-OK status가 발생하면 해당 node만 skip하고 전체 optimization pass는 계속 진행한다.

### 1.5 Post-check Stage

`frame.GetOutputs(fetches)`로 실제 결과를 가져오고, allocated tensor들의 `SizeInBytes()`를 누적해 `actual_total_size > max_output_size`인지 확인한다. 이 사후 검증은 shape inference가 놓친 실제 output size를 다시 검증하는 단계다.

---

## 2. ORT CPU Materialization Kernel 분석

### 2.1 `Expand<T>::Compute`

확인한 파일:

```text
onnxruntime/core/providers/cpu/tensor/expand.cc
```

핵심 구조는 다음과 같다.

```cpp
const auto* shape_tensor = context->Input<Tensor>(1);
const auto* shape_dims = shape_tensor->Data<int64_t>();
std::vector<int64_t> output_shape{shape_dims, shape_dims + shape_tensor->Shape().Size()};

// broadcast legality check ...

TensorShape output_tensor_shape(output_shape);
auto* output_tensor = context->Output(0, output_tensor_shape);
auto* output_data = output_tensor->MutableData<T>();
```

그 후 `memcpy` 기반으로 output buffer를 실제로 채운다.

```cpp
memcpy(output_data + output_offset, input_data + input_offset, copy_byte);
```

그리고 doubling-style copy loop도 존재한다.

핵심 결론은 다음이다.

- `SafeInt`는 dimension accumulation, offset, copy length overflow를 막는다.
- 하지만 `context->Output()` 전에 `Π(output_shape) * sizeof(T)`가 정책상 허용 가능한지 검사하는 byte-budget guard는 없다.
- 따라서 overflow는 아니지만 “합법적으로 매우 큰 양수”인 output shape는 runtime allocation으로 진행될 수 있다.

### 2.2 `ConstantOfShape::Compute`

확인한 파일:

```text
onnxruntime/core/providers/cpu/generator/constant_of_shape.cc
```

핵심 구조는 다음이다.

```cpp
Tensor* output_tensor = nullptr;
ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor));

auto output_data = output_tensor->MutableDataRaw();
const auto size = output_tensor->Shape().Size();
const auto element_size = output_tensor->DataType()->Size();

FilloutOutput(..., output_data, onnxruntime::narrow<size_t>(size));
```

`FilloutOutput`는 다음과 같다.

```cpp
template <class T>
inline void FilloutOutput(T value, void* output_data, size_t size) {
  std::fill_n(reinterpret_cast<T*>(output_data), size, value);
}
```

즉 `PrepareCompute()` 이후 output tensor가 준비되고, `std::fill_n`이 output 전체를 순차적으로 write한다. 이 지점이 page touch가 발생하는 핵심 구간이다.

### 2.3 `Tile::Compute`와의 대조

확인한 파일:

```text
onnxruntime/core/providers/cpu/tensor/tile.cc
```

`Tile`에는 다음과 같은 hardcoded guard가 있다.

```cpp
constexpr int64_t kMaxTileOutputBytes = int64_t{4} * 1024 * 1024 * 1024;
```

그리고 per-axis repeat 및 total element product를 `max_elements` 기준으로 검사한다. 즉 `Tile`은 “overflow는 아니지만 정책상 지나치게 큰 output”을 사전에 reject한다. 이 guard와 비교했을 때 `Expand`와 `ConstantOfShape`에는 같은 runtime byte-budget guard가 없다.

---

## 3. CPUAllocator 및 OS Memory Policy 분석

확인한 파일:

```text
include/onnxruntime/core/framework/allocator.h
onnxruntime/core/framework/allocator.cc
```

`IAllocator`는 다음 추상 인터페이스를 제공한다.

```cpp
virtual void* Alloc(size_t size) = 0;
virtual void Free(void* p) = 0;
virtual void* Reserve(size_t size) { return Alloc(size); }
```

`CPUAllocator::Alloc()`은 기본적으로 aligned allocation을 수행한다.

```cpp
void* CPUAllocator::Alloc(size_t size) {
  const auto alignment = std::max(Info().device.GetAlignment(), MlasGetPreferredBufferAlignment());
  return AllocatorDefaultAllocAligned(size, alignment);
}
```

기본 빌드에서는 다음 경로를 탄다.

```cpp
return ::operator new(size, std::align_val_t{alignment});
```

`USE_MIMALLOC` 빌드에서는 `mi_posix_memalign()` 경로를 사용한다.

중요한 확인점은 다음이다.

- ORT CPUAllocator는 직접 `mmap()`을 호출하지 않는다.
- 실제 low-level mapping은 C++ allocator, glibc, mimalloc 내부 정책에 위임된다.
- ORT 코드베이스 검색에서 `madvise`, `MADV_HUGEPAGE`, `MADV_NOHUGEPAGE` 사용은 확인되지 않았다.
- CPUAllocator 자체에는 per-allocation byte budget이 없다.
- SafeInt는 size 계산 overflow를 막지만, “64GiB처럼 overflow는 아니지만 운영 환경상 수용 불가능한 크기”를 정책적으로 거부하지 않는다.

OS 관점에서 이 구조는 Linux `vm.overcommit_memory`와 THP(Transparent Huge Pages) 정책에 영향을 받는다. 큰 output allocation은 virtual address range만 먼저 확보하고 성공할 수 있으며, 실제 RSS 증가는 `Expand`의 `memcpy` 또는 `ConstantOfShape`의 `std::fill_n`이 page를 write할 때 발생한다. THP가 활성화된 환경에서는 sequential write가 2MiB huge page allocation 또는 direct compaction 압력으로 이어질 수 있다.

---

## 4. Hugging Face `transformers` 실제 모델 소스 좌표

분석 대상은 `huggingface/transformers` main 브랜치였다.

### 4.1 Llama

파일:

```text
src/transformers/models/llama/modeling_llama.py
```

주요 좌표:

```python
inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
position_ids_expanded = position_ids[:, None, :].float()
```

그리고 `repeat_kv`:

```python
hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

이 구조는 PyTorch eager에서는 view/broadcast로 동작할 수 있지만, ONNX lowering에서는 `Unsqueeze -> Expand -> Reshape` chain으로 내려갈 수 있다.

### 4.2 Qwen2 / Qwen2.5 계열

파일:

```text
src/transformers/models/qwen2/modeling_qwen2.py
```

Llama와 거의 같은 RoPE 및 `repeat_kv` 구조를 가진다. `Qwen2Attention.forward`는 projection 결과를 `view(...).transpose(...)`로 multi-head layout으로 변환한 뒤 RoPE와 attention backend로 넘긴다.

### 4.3 Phi-3

파일:

```text
src/transformers/models/phi3/modeling_phi3.py
```

주요 좌표는 Llama/Qwen과 동일한 RoPE 및 `repeat_kv`다. Phi-3는 partial rotary dimension 처리 때문에 `Slice`/`Concat` 성격의 adapter zone이 더 뚜렷하다.

### 4.4 DeepSeek V2

파일:

```text
src/transformers/models/deepseek_v2/modeling_deepseek_v2.py
```

MLA latent KV decompression 핵심 좌표:

```python
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
k_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings.to(q_pe.device))
k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
```

MoE router dense mask 좌표:

```python
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)
score_mask = (
    group_mask.unsqueeze(-1)
    .expand(batch_size * seq_len, self.num_group, self.num_experts // self.num_group)
    .reshape(batch_size * seq_len, -1)
)
tmp_scores = router_logits.masked_fill(~score_mask.bool(), 0.0)
```

이 경로는 `B * S * num_experts`에 비례하는 dense mask materialization 가능성을 만든다.

### 4.5 AttentionMaskConverter

파일:

```text
src/transformers/modeling_attn_mask_utils.py
```

핵심 좌표:

```python
return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
```

그리고:

```python
expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
```

이 경로는 prefill 상황에서 `O(B * S^2)` dense 4D mask를 생성할 수 있다.

---

## 5. QDQ Propagation 및 Quantization 관련 분석

확인한 파일:

```text
onnxruntime/core/optimizer/qdq_transformer/qdq_propagation.cc
onnxruntime/core/optimizer/qdq_transformer/qdq_util.cc
```

`qdq_propagation.cc`의 `CanNodePropagate()`는 다음 op들을 값 보존 layout op로 보고 QDQ propagation을 허용한다.

```text
MaxPool
Reshape
Transpose
Squeeze
Unsqueeze
Slice
```

핵심은 QDQ가 직접 `Expand`를 통과한다는 것이 아니라, `Reshape`, `Unsqueeze`, `Slice`, `Transpose` 같은 layout adapter를 통과해 materialization boundary 바로 앞뒤로 이동할 수 있다는 점이다.

`qdq_util.cc`에서는 `QOrDQNodeHasConstantScalarScaleAndZeroPoint()`가 scale/zero-point가 constant scalar인지 확인한다. 하지만 해당 QDQ edge가 graph-input-dependent anti-folding chain인지, downstream에 `Expand`, `Where`, `ReduceSum` 같은 materialization/predicate anchor가 있는지는 확인하지 않는다.

또한 `IsClipMadeRedundantByQ()`에서는 다음 구조가 확인됐다.

```cpp
int32_t q_clip_min = static_cast<int32_t>(::rint(clip_min / scale)) + zp;
int32_t q_clip_max = static_cast<int32_t>(::rint(clip_max / scale)) + zp;
```

이 로직은 Clip redundancy 판단에 사용된다. 세션에서는 여기에 `std::isfinite(scale)`, min-scale lower bound, int32 range check, predicate stability check가 필요할 수 있다고 정리했다.

---

## 6. GitHub Issue / PR 히스토리 발굴

### 6.1 ORT PR #28055

제목:

```text
fix(security): add SafeInt overflow protection in Expand and constant folding output size limit
```

핵심 내용:

- `Expand::Compute()`의 raw int64 dimension accumulation 및 offset/copy length 계산에 SafeInt 적용
- `constant_folding.cc`에 output-size limit 추가
- session option `optimization.constant_folding_max_output_size_in_bytes` 추가
- constant folding 중 kernel compute exception isolation 추가

의미:

- constant-input folding-time Expand/ConstantOfShape bomb는 hardening됨
- 그러나 graph-input-dependent dynamic shape는 `AllNodeInputsAreConstant == false`로 folding되지 않고 runtime으로 넘어가므로 runtime materialization guard는 별도 축으로 남음

### 6.2 ORT PR #22888

제목:

```text
Add Optional Redundant Clip Node to NodeUnit
```

핵심 내용:

- QDQ NodeUnit에서 Q node에 의해 redundant하다고 판단되는 Clip/Relu를 metadata로 포함
- EP가 이를 무시하거나 fused unit으로 처리할 수 있게 함
- `IsClipMadeRedundantByQ()` 관련 설계 축 확인

의미:

- QDQ/Clip redundancy 최적화의 실제 출처
- overflow-safe quantized interval check, predicate stability proof는 별도 hardening 후보

### 6.3 ORT PR #28178

제목:

```text
fix: support N-D weights with unit leading dims in MatMulNBitsQuantizer
```

핵심 내용:

- N-D weight with unit leading dims를 2D로 squeeze해 MatMulNBits quantization 수행
- output shape 복원을 위해 `Shape/Gather/Max/Sub/ConstantOfShape/Slice/Concat -> Reshape` helper chain 추가

의미:

- quantizer가 shape/layout helper graph를 실제로 생성한다는 근거
- QDQ/INT4 quantization이 단순 weight rewrite가 아니라 graph topology를 바꾼다는 증거

### 6.4 ORT PR #28297

제목:

```text
[OVEP] OpenVINO EP 1.26.0 Development Release Updates
```

핵심 내용:

- OpenVINO 2026.1 이상에서 OVEP-level QDQ stripping disable
- NPU에서 특정 조건 외에는 `disable_dynamic_shapes=true`
- `ReduceSum`을 no-dimension-supported ops에 추가
- OpenVINO EP-specific workaround/policy switch 확인

의미:

- EP-local workaround는 존재하지만, ORT core invariant는 아님
- CPU EP / QDQ core / runtime materialization kernel 문제는 별도임

### 6.5 ORT Issue #3041

제목:

```text
torchvision MaskRCNN export to ONNX throws error; ConstantOfShape ... Invalid shape value: 0
```

핵심 내용:

- PyTorch torchvision MaskRCNN export 후 ORT load 시 `ConstantOfShape` shape inference error
- stale bot에 의해 닫혔고, 이후에도 “same problem” 댓글이 이어짐

의미:

- exporter-generated dynamic shape graph와 ORT verification boundary의 역사적 충돌 사례
- RSS 폭발 직접 사례는 아니지만 `ConstantOfShape` 계열 shape graph가 반복 문제를 만든다는 근거

---

## 7. 도출한 PR / 패치 방향

세션에서 도출한 핵심 PR 방향은 다음이다.

### 7.1 Runtime materialization byte-budget guard

대상:

```text
Expand<T>::Compute
ConstantOfShape::Compute / ConstantOfShapeBase::PrepareCompute
Resize / NonZero / 기타 materialization op 후보
```

핵심 invariant:

```text
No materialized tensor may be allocated unless
Π(shape_dims) * element_size <= configured_budget
```

`Tile`의 4GiB guard를 일반화하는 형태가 적절하다.

### 7.2 QDQ materialization-risk fence

대상:

```text
qdq_propagation.cc
PropagateDQForward
PropagateQBackward
InsertQDQPairs
```

핵심 아이디어:

```text
QDQ propagation 자체를 끄지 않는다.
다만 QDQ가 layout adapter를 거쳐 Expand / ConstantOfShape / Where / NonZero / Reduce* 같은 materialization-risk consumer로 들어가는 경우만 국소적으로 fence를 친다.
```

추상 함수:

```text
EdgeFeedsMaterializationRisk(edge)
```

### 7.3 Clip redundancy hardening

대상:

```text
qdq_util.cc
IsClipMadeRedundantByQ
```

필요 조건:

- `scale` finite check
- `scale > 0`
- min safe scale lower bound
- `clip / scale + zp` 계산의 double 기반 int32 range check
- predicate-sensitive downstream boundary에서는 Clip 제거 보수화

### 7.4 OS memory interaction mitigation

대상:

```text
CPU materialization kernels
CPUAllocator optional policy layer
```

아이디어:

- allocator 전역보다 materialization kernel-level byte guard 우선
- large transient tensor에는 optional `MADV_NOHUGEPAGE` best-effort hint 가능
- 단, `MADV_NOHUGEPAGE`는 byte-budget guard의 대체가 아니라 보조 수단

---

## 8. 최종 결론

이번 세션의 핵심 결론은 다음이다.

ORT의 constant folding은 `node exists`, `AllowConstantFolding`, provider compatibility, determinism, subgraph exclusion, input constness, output-size precheck, CPU kernel creation, kernel compute, actual output-size postcheck라는 단계적 short-circuit 제어 흐름을 가진다. 이 구조는 compile-time constant evaluation의 정합성을 잘 보장하지만, runtime graph input에 의존하는 dynamic shape chain은 `AllNodeInputsAreConstant` 조건을 깨기 때문에 folding되지 않고 runtime kernel로 넘어간다.

런타임에서는 `Tile`처럼 바이트 상한을 가진 op가 있는 반면, `Expand`와 `ConstantOfShape`는 같은 수준의 pre-allocation byte-budget guard가 확인되지 않았다. CPUAllocator는 aligned `operator new` 또는 mimalloc에 의존하며, ORT 코드 레벨에서 `madvise(MADV_NOHUGEPAGE)` 같은 THP 제어 힌트도 확인되지 않았다. 따라서 큰 dynamic output tensor는 OS overcommit과 page fault 정책에 따라 allocation 성공 후 write 시점에 RSS가 폭증할 수 있다.

Hugging Face의 Llama/Qwen/Phi `repeat_kv`, RoPE, DeepSeek MLA/MoE, 그리고 공통 4D attention mask converter는 실제로 `expand`, `reshape`, `slice`, `masked_fill`, `concat` 계열 연산을 많이 사용한다. 이들은 ONNX lowering 이후 ORT `Expand`/`ConstantOfShape`/QDQ layout adapter와 연결될 수 있다.

기존 ORT GitHub 히스토리에서는 `#28055`가 constant-folding-time Expand/ConstantOfShape risk를 일부 해결했고, `#22888`이 QDQ Clip redundancy를 도입했으며, `#28178`은 quantizer가 dynamic reshape helper를 생성하는 실제 사례를 보여주고, `#28297`은 EP-specific QDQ/dynamic-shape workaround를 보여준다. 그러나 runtime dynamic materialization byte-budget guard, QDQ materialization-risk fence, Clip redundancy의 overflow/predicate-stability hardening은 별도 core PR 축으로 남는다.
