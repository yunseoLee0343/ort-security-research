# Hugging Face ⇄ ONNX Runtime Verification-Execution Gap & Resource Amplification Audit Report

**Date:** 2026-05-19  
**Scope:** Hugging Face `transformers` model code paths and ONNX Runtime CPU/CUDA Execution Provider materialization paths  
**Primary Risk Theme:** Runtime-derived symbolic/dynamic shapes survive framework/export optimization and reach ORT kernels that allocate contiguous physical output buffers.

---

## 0. Executive Summary

This report consolidates the investigation of two connected layers:

1. **ONNX Runtime kernel layer**
   - `Expand`
   - `ConstantOfShape`
   - `RandomUniformLike` / `RandomNormalLike`
   - `OpKernel::Compute()` virtual dispatch
   - graph optimizer / constant folding registration

2. **Hugging Face model layer**
   - dynamic RoPE and dynamic NTK/Yarn/LongRoPE update paths
   - GQA/MQA `repeat_kv()` expansion
   - DeepSeek-V2 MoE routing masks
   - dense attention mask creation in `masking_utils.py`
   - generation-loop RNG candidates

The central finding is:

```text
HF runtime input / runtime shape
→ ONNX dynamic shape graph
→ ORT kernel sees legal shape
→ ORT allocates contiguous output via context->Output()
→ full-buffer write / memcpy / fill / RNG loop
→ physical RSS spike and page-fault amplification
```

This is not just a generic "large tensor" problem. The more precise failure mode is a **Verification-Execution Gap**:

```text
High-level verifier checks semantic legality:
  broadcast-compatible shape
  valid attention mask
  valid MoE routing index
  valid RoPE position ids

Execution backend performs physical materialization:
  contiguous output allocation
  full sequential write
  allocator arena pressure
  OS page commit / RSS growth
```

The most actionable patch direction is:

```text
Add runtime output-size guards for shape-materializing CPU generator and broadcast kernels,
and add bounded ONNX export guards for dynamic RoPE, dense attention masks, and MoE routing masks.
```

---

## 1. Threat Model and Audit Vectors

### 1.1 Targeted resource amplification class

A resource amplification candidate exists when:

```text
small model file / small runtime control tensor
→ dynamic shape computation
→ shape-dependent materializing op
→ output bytes much larger than input bytes
```

Canonical formula:

```text
Amplification = materialized_output_bytes / control_input_bytes
```

For `ConstantOfShape([65536, 65536])` with float output:

```text
control shape bytes = 2 × 8 = 16 bytes
output bytes        = 65536 × 65536 × 4 = 17,179,869,184 bytes ≈ 16 GiB
amplification       ≈ 1,073,741,824x
```

### 1.2 ORT kernel audit vectors

| Vector | Description |
|---|---|
| ORT-RSS-001 | `Expand` or broadcast-style op materializes a huge logical shape into a contiguous physical tensor |
| ORT-RSS-002 | `RandomUniformLike` / `RandomNormalLike` inherits a huge input shape and writes a full output tensor |
| ORT-RSS-003 | `ConstantOfShape` / fill-like op converts small shape tensor into huge full-buffer allocation |
| ORT-PERF-001 | `OpKernel::Compute()` virtual dispatch and runtime dtype/branch dispatch add hot-path overhead |
| ORT-OPT-001 | runtime-dependent anti-folding chains survive graph optimization and reach execution |

### 1.3 Hugging Face model trigger vectors

| Vector | Description |
|---|---|
| HF-ORT-001 | Dynamic RoPE / dynamic NTK creates runtime shape-dependent frequency tensors |
| HF-ORT-002 | `repeat_kv()` uses `expand()` to virtually repeat KV heads before attention |
| HF-ORT-003 | MoE routing uses `one_hot`, `zeros_like`, `scatter_`, `expand`, `masked_fill` |
| HF-ORT-004 | dense attention mask creation builds `[B, 1, Q, KV]` masks |
| HF-ORT-005 | RNG noise inside model `forward()` may lower to `Random*Like`, but evidence is weak in core HF decoder paths |

---

# Part I — ONNX Runtime Kernel and Optimizer Findings

---

## Finding ORT-RSS-001: CPU `Expand` as eager materialization primitive

### Risk

**High**

### Source location

```text
onnxruntime/core/providers/cpu/tensor/expand.cc
template <typename T>
Status Expand<T>::Compute(OpKernelContext* context) const
```

### Relevant source URL

```text
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/tensor/expand.cc
```

### Evidence

The CPU `Expand` kernel:

```cpp
TensorShape output_tensor_shape(output_shape);
auto* output_tensor = context->Output(0, output_tensor_shape);
auto* output_data = output_tensor->MutableData<T>();
```

Then it builds dimension groups and uses `memcpy` to distribute and expand the data into the output buffer.

### Mechanism chain

```text
ONNX Expand shape tensor
→ TensorShape output_tensor_shape(output_shape)
→ context->Output(0, output_tensor_shape)
→ allocator creates contiguous output tensor
→ MutableData<T>() exposes writable pointer
→ memcpy/distributed copy writes output range
→ page faults commit physical RSS
```

### Mathematical proof

```text
Input elements  = Π input_shape
Output elements = Π output_shape
Amplification   = (Π output_shape / Π input_shape) × sizeof(T)
```

Example:

```text
input_shape  = [1]
output_shape = [1, 65536, 65536]
dtype        = float32

physical output bytes
= 1 × 65536 × 65536 × 4
= 17,179,869,184 bytes
≈ 16 GiB
```

### Interpretation

This implementation is not a lazy view. It is an eager materializer. The high-level ONNX semantics of broadcastability are valid, but ORT CPU execution pays the full physical output cost.

### Defensive payload condition

```text
small initializer or scalar input
→ Expand to [B, S, H] where B×S×H×sizeof(T) exceeds memory budget
```

### Patch direction

Add a pre-allocation byte budget guard:

```cpp
auto output_elems = output_tensor_shape.Size();
auto output_bytes = SafeInt<size_t>(output_elems) * sizeof(T);
ORT_RETURN_IF_NOT(output_bytes <= session_memory_budget,
                  "Expand output exceeds configured memory budget");
```

Add an optimizer/lint pass for:

```text
Expand(very_large_shape) → Random*Like / ConstantOfShape / Fill / Scatter / Gather / Pad
```

---

## Finding ORT-RSS-002: `RandomUniformLike` / `RandomNormalLike` full-shape output write

### Risk

**High**

### Source location

```text
onnxruntime/core/providers/cpu/generator/random.h
onnxruntime/core/providers/cpu/generator/random.cc
RandomUniformLike::Compute()
RandomNormalLike::Compute()
CreateOutputTensorFromTensorShape()
GenerateData()
```

### Relevant source URLs

```text
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/generator/random.h
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/generator/random.cc
```

### Evidence

The `Random*Like` kernels use the input tensor shape directly:

```cpp
const Tensor& X = *tensor_pointer;
Tensor* Y = nullptr;

auto status = CreateOutputTensorFromTensorShape(ctx, X, &Y);
```

The output is allocated with the same shape:

```cpp
static Status CreateOutputTensorFromTensorShape(OpKernelContext* ctx, const Tensor& X, Tensor** Y) {
  *Y = ctx->Output(0, X.Shape());
  return Status::OK();
}
```

Then `GenerateData()` writes every element:

```cpp
for (int64_t i = 0, end = tensor.Shape().Size(); i < end; ++i) {
  *out = distribution(generator);
  ++out;
}
```

### Mechanism chain

```text
Large input X
→ CreateOutputTensorFromTensorShape(ctx, X, &Y)
→ ctx->Output(0, X.Shape())
→ allocate Y with same full logical shape
→ GenerateData loops over Y.Shape().Size()
→ one RNG call + one store per element
```

### Amplification with preceding `Expand`

```text
Expand output allocation:       huge_shape × sizeof(T)
RandomLike output allocation:   huge_shape × sizeof(T2)
RandomLike sequential write:    huge_shape RNG stores
Peak memory:                    roughly 2 × huge_shape materialization
```

### Defensive payload condition

```text
x:              scalar or [1]
expand_shape:   [1, 32768, 32768]
expanded dtype: float32
consumer:       RandomUniformLike(dtype=float32)

Expand output ≈ 4 GiB
Random output ≈ 4 GiB
Peak minimum ≈ 8 GiB + allocator overhead
```

### Patch direction

Add generator-specific hard caps:

```cpp
const auto output_elems = X.Shape().Size();
const auto output_bytes = SafeInt<size_t>(output_elems) * SizeOf(dtype);
ORT_RETURN_IF_NOT(output_bytes <= max_generator_output_bytes,
                  "Random*Like output exceeds configured memory budget");
```

`Random*Like` should be classified as a shape-materializing generator op, not as a harmless elementwise operator.

---

## Finding ORT-RSS-003: `ConstantOfShape` runtime shape to full-buffer fill

### Risk

**High**

### Source location

```text
onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h
onnxruntime/core/providers/cpu/generator/constant_of_shape.cc
ConstantOfShapeCore::PrepareCompute()
ConstantOfShape::Compute()
```

### Relevant source URLs

```text
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/generator/constant_of_shape.cc
```

### Evidence

`ConstantOfShape` uses the input tensor values as output shape:

```cpp
const auto span = shape_tensor->template DataAsSpan<int64_t>();

TensorShape output_shape(span);
(*output_tensor) = ctx->Output(0, output_shape);
```

Then it fills the entire output:

```cpp
const auto size = output_tensor->Shape().Size();
FilloutOutput(..., output_data, onnxruntime::narrow<size_t>(size));
```

The fill primitive:

```cpp
template <class T>
inline void FilloutOutput(T value, void* output_data, size_t size) {
  std::fill_n(reinterpret_cast<T*>(output_data), size, value);
}
```

### Mechanism chain

```text
shape_tensor values
→ TensorShape output_shape(span)
→ ctx->Output(0, output_shape)
→ output_data = MutableDataRaw()
→ size = output_tensor->Shape().Size()
→ std::fill_n(output_data, size, value)
```

### Amplification proof

```text
shape tensor bytes ≈ rank × 8
output bytes       = Π shape[i] × element_size
amplification      = output_bytes / (rank × 8)
```

Example:

```text
shape tensor = [65536, 65536]
shape bytes  = 16 bytes
output float = 16 GiB
amplification ≈ 1,073,741,824x
```

### Anti-folding stress chain

```text
runtime_input_shape
→ Identity
→ Add(runtime_zero)
→ Cast<int64>
→ ConstantOfShape
```

### Patch direction

Add `PrepareCompute()` output product and byte cap:

```text
dim_i <= max_dim
Π dim_i × element_size <= max_tensor_bytes
```

`ConstantOfShape` requires early rejection before allocator invocation.

---

## Finding ORT-PERF-001: `OpKernel::Compute()` virtual dispatch is real but secondary

### Risk

**Medium**

### Source location

```text
include/onnxruntime/core/framework/op_kernel.h
class OpKernel
```

### Relevant source URL

```text
https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/framework/op_kernel.h
```

### Evidence

```cpp
[[nodiscard]] virtual Status Compute(_Inout_ OpKernelContext* context) const = 0;
```

`IsAsync()` and `ComputeAsync()` are also virtual.

### Mechanism chain

```text
Execution scheduler
→ OpKernel* selected at runtime
→ virtual Compute(ctx)
→ indirect branch
→ BTB / branch predictor resolves target
→ tiny-op-heavy graph may suffer dispatch overhead
```

### Interpretation

This matters for:

```text
many small ops
+ heterogeneous op types
+ dynamic EP assignment
+ tiny tensors
+ millions of node invocations
```

It does not dominate for:

```text
single Expand 16 GiB
single ConstantOfShape 16 GiB
single RandomUniformLike 16 GiB
```

In large materializing kernels, memory traffic and full-buffer writes dominate over virtual dispatch overhead.

### Patch direction

Prioritize:

```text
1. graph-level fusion
2. tiny-op static dispatch/fused kernels
3. dtype-specialized registrations for high-frequency tiny ops
```

Do not prioritize `Compute()` devirtualization before eliminating materialization.

---

## Finding ORT-OPT-001: Anti-folding chain survives because graph optimization is conservative around runtime inputs

### Risk

**Medium → High**

### Source location

```text
onnxruntime/core/optimizer/graph_transformer_utils.cc
GenerateRewriteRules()
GenerateTransformers()
```

### Relevant source URL

```text
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/graph_transformer_utils.cc
```

### Evidence

Level1 optimizers include:

```cpp
rules.push_back(std::make_unique<EliminateIdentity>());
rules.push_back(std::make_unique<ExpandElimination>());
...
transformers.emplace_back(std::make_unique<ConstantSharing>(...));
transformers.emplace_back(std::make_unique<CommonSubexpressionElimination>());
transformers.emplace_back(std::make_unique<ConstantFolding>(...));
```

### Mechanism chain

```text
runtime scalar/shape input
→ Identity / Add(0) / Mul(1) / Cast
→ shape tensor not classified as compile-time initializer
→ ConstantFolding cannot legally execute it at optimization time
→ ConstantOfShape / Expand receives runtime shape
→ ctx->Output allocates full physical tensor
```

### Interpretation

This is not necessarily a bug in constant folding. It is a correct conservative limitation. The security/performance failure occurs because the execution kernels do not have an equivalent runtime shape budget verifier.

### Patch direction

Add a runtime-shape budget pass:

```text
RuntimeShapeBudgetVerifier:
  runtime-derived shape → Expand / ConstantOfShape / Tile / Range / Scatter / Pad
```

This should be treated as safety verification, not optimization.

---

# Part II — Hugging Face Model-Level Trigger Findings

---

## Finding HF-ORT-001: Dynamic RoPE expand chain

### ORT risk and linked component

**High → ORT-RSS-001**

### Source locations

```text
src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding.forward()

src/transformers/models/deepseek_v2/modeling_deepseek_v2.py
class DeepseekV2RotaryEmbedding.forward()

src/transformers/modeling_rope_utils.py
dynamic_rope_update()
_compute_dynamic_ntk_parameters()
_compute_yarn_parameters()
_compute_longrope_parameters()
```

### Relevant source URLs

```text
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py
https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
```

### Evidence: Llama RoPE

```python
inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
position_ids_expanded = position_ids[:, None, :].float()

freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos() * self.attention_scaling
sin = emb.sin() * self.attention_scaling
```

### Evidence: DeepSeek-V2 RoPE

```python
inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
position_ids_expanded = position_ids[:, None, :].float()

freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

### Evidence: dynamic RoPE update

```python
seq_len = torch.max(position_ids) + 1
...
if seq_len > max_seq_len_cached:
    inv_freq, self.attention_scaling = rope_init_fn(
        self.config,
        device,
        seq_len=seq_len,
        layer_type=layer_type,
    )
    self.register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)
```

### Lowering and optimization-loss proof

```text
self.inv_freq: initializer-like model buffer
position_ids: runtime input
batch = position_ids.shape[0]
seq_len = position_ids.shape[-1]

expand_shape = [batch, rotary_dim/2, 1]
matmul output = [batch, rotary_dim/2, seq_len]
transpose/cat output = [batch, seq_len, rotary_dim]
```

If `position_ids` is exported with dynamic axes, the expand target shape is not compile-time constant. ORT must execute the materializing path.

Dynamic NTK is stronger:

```text
seq_len = max(position_ids) + 1
→ runtime value participates in frequency base calculation
→ inv_freq becomes runtime-dependent
→ cos/sin path cannot be statically initialized
```

### Payload threshold formula

```text
RoPE materialized bytes ≈ B × S × rotary_dim × bytes(dtype) × 2
```

For `rotary_dim=128`, `fp32`:

```text
B × S × 128 × 4 × 2 > 1 GiB
B × S > 1,048,576
```

Example stress profile:

```text
B=16, S=65536
```

This exceeds 1 GiB in RoPE intermediate buffers alone.

### Model-level mitigation

Add export-time bound checks:

```python
def _guard_rope_position_ids(position_ids: torch.Tensor, max_seq_len: int, max_batch: int):
    if torch.onnx.is_in_onnx_export():
        assert position_ids.shape[0] <= max_batch
        assert position_ids.shape[-1] <= max_seq_len
    return position_ids
```

Separate dynamic RoPE from ONNX export:

```python
if torch.onnx.is_in_onnx_export():
    # Export profile must use bounded/precomputed RoPE.
    inv_freq = self.original_inv_freq
else:
    # Runtime dynamic update path.
    ...
```

---

## Finding HF-ORT-002: `repeat_kv()` broadcast expansion before attention

### ORT risk and linked component

**Medium → High → ORT-RSS-001**

### Source locations

```text
src/transformers/models/llama/modeling_llama.py
repeat_kv()
eager_attention_forward()

src/transformers/models/deepseek_v2/modeling_deepseek_v2.py
repeat_kv()
eager_attention_forward()
```

### Evidence

```python
batch, num_key_value_heads, slen, head_dim = hidden_states.shape
if n_rep == 1:
    return hidden_states
hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

### Lowering and optimization-loss proof

PyTorch eager may represent `expand()` as stride-0 view. ONNX lowering typically emits:

```text
Unsqueeze → Expand → Reshape
```

ORT CPU `Expand` materializes this output.

### Amplification formula

```text
Input KV bytes  = B × H_kv × S × D × bytes
Output KV bytes = B × H_q  × S × D × bytes
Amplification   = H_q / H_kv = num_key_value_groups
```

For key and value:

```text
KV expansion bytes ≈ 2 × B × S × H_q × D × bytes
```

Example:

```text
B=8, S=32768, H_q=32, D=128, fp16

2 × 8 × 32768 × 32 × 128 × 2
≈ 4 GiB
```

### Model-level mitigation

Avoid exporting `repeat_kv()` as materializing `Expand` where possible. Prefer a GQA-aware fused attention symbolic.

Defensive guard:

```python
def repeat_kv_safe(hidden_states, n_rep, max_materialized_bytes=None):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    out_elems = batch * num_kv_heads * n_rep * slen * head_dim
    out_bytes = out_elems * hidden_states.element_size()
    if max_materialized_bytes is not None and out_bytes > max_materialized_bytes:
        raise RuntimeError("repeat_kv would materialize too much memory")
    ...
```

---

## Finding HF-ORT-003: DeepSeek-V2 MoE routing mask materialization

### ORT risk and linked component

**High → ORT-RSS-001 / ORT-RSS-003**

### Source location

```text
src/transformers/models/deepseek_v2/modeling_deepseek_v2.py
class DeepseekV2Experts.forward()
class DeepseekV2Moe.route_tokens_to_experts()
```

### Evidence

Expert one-hot mask:

```python
expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
expert_mask = expert_mask.permute(2, 1, 0)
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
```

Group routing mask:

```python
group_scores = router_logits.view(batch_size * seq_len, self.num_group, -1).max(dim=-1).values
group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)
score_mask = (
    group_mask.unsqueeze(-1)
    .expand(batch_size * seq_len, self.num_group, self.num_experts // self.num_group)
    .reshape(batch_size * seq_len, -1)
)
tmp_scores = router_logits.masked_fill(~score_mask.bool(), 0.0)
```

### Lowering and optimization-loss proof

```text
hidden_states: [B, S, H]
router_logits: [B, S, E]
view:          [B*S, E]
topk_idx:      [B*S, K]
expert_mask:   one_hot([B*S, K], E) → [B*S, K, E]
permute:       [E, K, B*S]
```

`topk_idx` depends on runtime logits, so it cannot be folded.

Likely lowering risks:

```text
torch.zeros_like(group_scores)
→ ONNX ConstantOfShape / Fill
→ ORT-RSS-003

group_mask.unsqueeze(-1).expand(...)
→ ONNX Expand
→ ORT-RSS-001

masked_fill(...)
→ full-buffer mask write
```

### Payload threshold formula

```text
one_hot bytes   = B × S × K × E × bytes(one_hot_dtype)
score_mask bytes = B × S × E × bytes(bool/int/float)
```

Audit threshold:

```text
B × S × K × E > route_mask_element_budget
```

For `K=8`, `E=256`:

```text
B × S × 2048 elements
```

### Model-level mitigation

Avoid dense one-hot expert dispatch during ONNX export:

```python
if torch.onnx.is_in_onnx_export():
    # one_hot([tokens, topk], num_experts) is not export-safe for large dynamic tokens.
    return sparse_expert_dispatch(hidden_states, top_k_index, top_k_weights)
```

Add routing element budget:

```python
tokens = batch_size * seq_len
if tokens * self.top_k * self.num_experts > self.config.max_moe_routing_elements:
    raise RuntimeError("MoE routing mask exceeds export/runtime budget")
```

---

## Finding HF-ORT-004: Dense attention mask materialization in `masking_utils`

### ORT risk and linked component

**High → ORT-RSS-001 / ORT-RSS-003**

### Source location

```text
src/transformers/masking_utils.py
sdpa_mask()
eager_mask()
```

### Relevant source URL

```text
https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py
```

### Evidence

The mask builder creates index ranges:

```python
batch_arange = torch.arange(batch_size, device=device)
head_arange = torch.arange(1, device=device)
q_arange = torch.arange(q_length, device=device) + q_offset
kv_arange = torch.arange(kv_length, device=device) + kv_offset
```

It broadcasts them into 4D:

```python
attention_mask = mask_function(*_non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange))
attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
```

`eager_mask()` converts boolean mask to float additive mask:

```python
mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
```

### Why export/tracing makes this worse

The code explicitly avoids some mask-skipping logic under tracing:

```text
if is_tracing(padding_mask):
    return False
```

Thus export paths are more likely to materialize dense masks instead of relying on runtime kernel `is_causal` flags.

### Lowering and optimization-loss proof

```text
q_length, kv_length, batch_size, kv_offset
→ runtime shape/value
→ torch.arange(q_length), torch.arange(kv_length)
→ broadcasted index tensors
→ compare/mask function
→ [B, 1, Q, KV] boolean mask
→ torch.where to float additive mask
```

If `Q` and `KV` are runtime dynamic, this cannot be folded.

### Payload threshold formula

```text
bool mask bytes  ≈ B × Q × KV × 1
float mask bytes ≈ B × Q × KV × dtype_bytes
```

For prefill:

```text
Q = S
KV = S
float mask bytes ≈ B × S² × dtype_bytes
```

For fp32 1 GiB threshold:

```text
B × S² × 4 > 2^30
B × S² > 268,435,456
```

Examples:

```text
B=1, S≈16384 → fp32 mask ≈ 1 GiB
B=4, S≈8192  → fp32 mask ≈ 1 GiB
```

### Model-level mitigation

1. For ONNX export, prefer fused attention symbolic with causal metadata.
2. Avoid generating dense `[B, 1, Q, KV]` masks for large dynamic sequence lengths.
3. Add export profile caps:

```python
max_mask_elems = config.onnx_max_batch * config.onnx_max_q_len * config.onnx_max_kv_len
if max_mask_elems > config.onnx_max_mask_elements:
    raise RuntimeError("Dense attention mask exceeds ONNX export budget")
```

4. If attention mask is known all-ones, use explicit export annotation to skip mask creation.

---

## Finding HF-ORT-005: RNG noise injector inside model `forward()` is weakly supported

### ORT risk and linked component

**Low → weak ORT-RSS-002 candidate**

### Evidence summary

The scan did not identify a strong pattern in common HF decoder model `forward()` paths of:

```python
x = y.expand(...)
noise = torch.randn_like(x)
# or
noise = torch.rand_like(x)
```

Generation sampling is generally outside the model forward path and uses generation utilities/logits sampling rather than a hard-coded model internal `rand_like` over expanded tensors.

### Risk condition if present in custom code

This would be dangerous:

```python
expanded = hidden_states.expand(...)
noise = torch.randn_like(expanded)
```

Likely lowering:

```text
Expand → RandomNormalLike
```

This directly links to `ORT-RSS-002`.

### Model-level mitigation

Add audit rule:

```text
Reject or guard:
  rand_like(expand(...))
  randn_like(expand(...))
  rand_like(dynamic_mask.float())
  randn_like(dynamic_mask.float())
```

---

# 3. Cross-Layer Mechanism Chains

## 3.1 RoPE chain

```text
position_ids runtime input
→ torch.max(position_ids)+1 or position_ids.shape
→ dynamic RoPE frequency update
→ inv_freq expand by batch
→ matmul with position ids
→ cos/sin or polar
→ ONNX Expand/MatMul/Cos/Sin
→ ORT materialization
```

## 3.2 GQA/MQA chain

```text
KV states [B, H_kv, S, D]
→ hidden_states[:, :, None, :, :]
→ expand(B, H_kv, n_rep, S, D)
→ reshape(B, H_q, S, D)
→ ONNX Expand
→ ORT materializes repeated KV
```

## 3.3 MoE routing chain

```text
hidden_states [B, S, H]
→ router_logits [B, S, E]
→ topk runtime indices [B*S, K]
→ one_hot [B*S, K, E]
→ group_mask zeros_like/scatter
→ expand to score_mask [B*S, E]
→ masked_fill
→ full-buffer materialization
```

## 3.4 Dense attention mask chain

```text
B, Q, KV dynamic lengths
→ torch.arange(Q), torch.arange(KV)
→ broadcasted index tensors
→ mask predicate
→ [B, 1, Q, KV] bool mask
→ torch.where to float mask
→ ORT shape-dependent allocation/fill
```

---

# 4. Recommended Patch Plan

## 4.1 ORT-side patches

### Patch 1: Output byte budget guard for materializing kernels

Target kernels:

```text
Expand
ConstantOfShape
Tile
Range
Pad
Scatter/Gather variants where output shape is runtime-derived
RandomUniformLike
RandomNormalLike
```

Common helper:

```cpp
Status CheckOutputByteBudget(const TensorShape& shape,
                             size_t elem_size,
                             const SessionOptions& options,
                             const char* op_name) {
  auto elems = shape.Size();
  auto bytes = SafeInt<size_t>(elems) * elem_size;
  ORT_RETURN_IF_NOT(bytes <= options.max_materialized_tensor_bytes,
                    op_name, " output exceeds materialized tensor byte budget");
  return Status::OK();
}
```

### Patch 2: RuntimeShapeBudgetVerifier

Add a graph/session verification pass that recognizes:

```text
runtime-derived shape → shape-materializing op
```

Pattern list:

```text
Shape/Gather/Concat/Cast/Identity/Add/Mul
→ Expand / ConstantOfShape / Tile / Range / Pad / Scatter / Random*Like
```

### Patch 3: Generator op cap

`Random*Like` must not blindly inherit arbitrary input shape.

```text
max_generator_output_bytes
max_generator_elements
```

### Patch 4: Diagnostics

Add structured logs:

```json
{
  "op": "Expand",
  "output_shape": [1, 65536, 65536],
  "output_bytes": 17179869184,
  "input_shape": [1],
  "runtime_shape_derived": true,
  "budget_exceeded": true
}
```

---

## 4.2 Hugging Face-side patches

### Patch 1: ONNX export safety config

```python
@dataclass
class OnnxExportSafetyConfig:
    max_batch: int = 8
    max_sequence_length: int = 8192
    max_kv_length: int = 8192
    max_dense_mask_elements: int = 268_435_456
    max_moe_routing_elements: int = 268_435_456
    max_rope_elements: int = 134_217_728
    max_repeat_kv_elements: int = 268_435_456
```

### Patch 2: common materialization guard

```python
def check_materialized_elements(name: str, elements: int, limit: int):
    if torch.onnx.is_in_onnx_export() and elements > limit:
        raise RuntimeError(
            f"{name} would materialize {elements} elements during ONNX export/runtime; "
            f"limit={limit}. Use bounded export profile or fused attention/MoE path."
        )
```

### Patch 3: RoPE guard

```python
def check_rope_export_budget(batch, seq_len, rotary_dim, limit):
    elems = batch * seq_len * rotary_dim
    check_materialized_elements("RoPE cos/sin", elems, limit)
```

### Patch 4: dense mask guard

```python
def check_dense_mask_export_budget(batch, q_len, kv_len, limit):
    elems = batch * q_len * kv_len
    check_materialized_elements("dense attention mask", elems, limit)
```

### Patch 5: MoE routing guard

```python
def check_moe_routing_export_budget(tokens, top_k, num_experts, limit):
    elems = tokens * top_k * num_experts
    check_materialized_elements("MoE routing one_hot", elems, limit)
```

### Patch 6: repeat_kv guard

```python
def check_repeat_kv_export_budget(batch, heads, seq_len, head_dim, limit):
    elems = batch * heads * seq_len * head_dim
    check_materialized_elements("repeat_kv expanded KV", elems, limit)
```

---

# 5. Proposed Upstream-Friendly PR Titles

## ORT

```text
Add output-size guards for shape-materializing CPU generator and broadcast kernels
```

```text
Add RuntimeShapeBudgetVerifier for dynamic shape-dependent materialization
```

```text
Add diagnostics for large runtime-derived tensor allocations
```

## Hugging Face Transformers

```text
Add bounded ONNX export guards for dynamic RoPE, dense attention masks, and MoE routing masks
```

```text
Avoid dense MoE routing masks in ONNX export paths
```

```text
Guard dense attention mask materialization during tracing/export
```

---

# 6. Severity Table

| Finding | Severity | Layer | Direct ORT Link |
|---|---:|---|---|
| ORT-RSS-001 | High | ORT CPU EP | `Expand` eager materialization |
| ORT-RSS-002 | High | ORT CPU EP | `Random*Like` full-shape output write |
| ORT-RSS-003 | High | ORT CPU EP | `ConstantOfShape` full-buffer fill |
| ORT-PERF-001 | Medium | ORT framework | `OpKernel::Compute()` virtual dispatch |
| ORT-OPT-001 | Medium/High | ORT optimizer | runtime anti-folding chain |
| HF-ORT-001 | High | HF model | dynamic RoPE → `Expand` |
| HF-ORT-002 | Medium/High | HF model | `repeat_kv()` → `Expand` |
| HF-ORT-003 | High | HF model | MoE `one_hot/zeros_like/expand` |
| HF-ORT-004 | High | HF utilities | dense attention mask → `Expand/Where` |
| HF-ORT-005 | Low | HF generation/model | RNG `rand_like` candidate weak |

---

# 7. Final Engineering Judgment

The strongest end-to-end vulnerability primitive is not a single exotic operator. It is a repeated design pattern:

```text
runtime input controls shape
+ framework uses lazy/broadcast abstraction
+ ONNX preserves dynamic shape
+ ORT CPU kernel materializes contiguous output
+ no explicit output byte budget exists at the kernel boundary
```

The three highest-priority chains are:

```text
1. Dense attention mask:
   B × Q × KV materialization

2. DeepSeek-style MoE routing:
   B × S × K × E one-hot/mask materialization

3. Dynamic RoPE / repeat_kv:
   B × S × D and B × S × H × D expansion
```

The correct architectural fix is split across both layers:

```text
HF side:
  prevent unbounded dynamic export shapes and avoid dense routing/mask materialization

ORT side:
  enforce runtime materialized-output byte budgets before context->Output()
```

The key invariant to enforce:

```text
Every runtime-derived shape that reaches a materializing kernel must pass an explicit byte-budget check before allocation.
```

---

# 8. Source Index

## ONNX Runtime

```text
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/tensor/expand.cc
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/generator/random.h
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/generator/random.cc
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/generator/constant_of_shape.cc
https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/framework/op_kernel.h
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/graph_transformer_utils.cc
```

## Hugging Face Transformers

```text
https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py
https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py
```
