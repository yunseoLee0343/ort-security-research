# ORT Constant Folding / Materialization / QDQ Analysis

## 0. Goal

Trace why ORT's existing constant-folding output-size limit
(`optimization.constant_folding_max_output_size_in_bytes`, PR #28055)
does not protect against runtime-shape-driven `Expand` / `ConstantOfShape`
materialization, and characterize the QDQ propagation chain that places a
`DequantizeLinear` immediately before `Expand` in quantized Llama graphs.

---

## 1. `ConstantFolding::ApplyImpl()` — when folding fires

ORT's constant folding gate can be modeled as:

```
ConstantFoldable(n) =
  node_exists
  ∧ AllowConstantFolding(n)
  ∧ IsSupportedProvider(n)
  ∧ IsOperationDeterministic(domain, op)
  ∧ !ContainsSubgraph(n)
  ∧ AllNodeInputsAreConstant(graph, n, ...)
  ∧ EstimatedOutputSize(n) <= configured_limit
  ∧ CPU kernel exists
  ∧ kernel->Compute() succeeds
  ∧ actual_output_size <= configured_limit
```

The critical condition for this research: **`AllNodeInputsAreConstant`**.

When the expand shape or the ConstantOfShape shape tensor is a graph input
(runtime tensor), `AllNodeInputsAreConstant` is false → the node is excluded
from constant folding → the output-size limit is never evaluated → the node
reaches the execution kernel with no prior rejection.

### Folding stages (for reference)

| Stage | What happens |
|---|---|
| Pre-check | `AllowConstantFolding`, provider compat, determinism, subgraph-free |
| Output-size pre-check | `EstimatedOutputSizeInBytes` vs. `configured_limit` |
| Kernel creation | temporary CPU EP kernel created |
| Compute | `kernel->Compute()` under try/catch |
| Post-check | actual allocated tensor bytes vs. `configured_limit` |

All five stages are bypassed when the shape input is runtime-derived.

---

## 2. CPU Materialization Kernels

### 2.1 `Expand<T>::Compute` — `expand.cc`

```cpp
const auto* shape_tensor = context->Input<Tensor>(1);
// ... broadcast legality check ...
TensorShape output_tensor_shape(output_shape);
auto* output_tensor = context->Output(0, output_tensor_shape);  // ← allocation
auto* output_data = output_tensor->MutableData<T>();
// ... memcpy-based fill of full output buffer ...
```

Key observations:
- `SafeInt` prevents arithmetic overflow in dimension accumulation and copy lengths.
- There is **no byte-budget check** before `context->Output()`.
- A legally broadcast-compatible but very large shape proceeds to physical allocation.

### 2.2 `ConstantOfShape::Compute` — `constant_of_shape.cc`

```cpp
Tensor* output_tensor = nullptr;
ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor));
auto output_data = output_tensor->MutableDataRaw();
const auto size = output_tensor->Shape().Size();
FilloutOutput(..., output_data, narrow<size_t>(size));
```

`std::fill_n` writes every element sequentially — this is where page faults
commit physical RSS. No byte-budget check exists in `PrepareCompute()`.

### 2.3 Contrast with `Tile::Compute` — `tile.cc`

```cpp
constexpr int64_t kMaxTileOutputBytes = int64_t{4} * 1024 * 1024 * 1024;
// per-axis and total product checked against kMaxTileOutputBytes
// → ORT_RETURN_IF_NOT before any allocation
```

`Tile` rejects oversized outputs before calling `context->Output()`.
`Expand` and `ConstantOfShape` do not have an equivalent guard.

---

## 3. CPUAllocator and OS Memory Behavior

`CPUAllocator::Alloc()` delegates to `::operator new` (or `mi_posix_memalign`
in mimalloc builds). No per-allocation budget exists at the allocator level.

On Linux with `vm.overcommit_memory=1`, a large `::operator new` can succeed
even if physical RAM is insufficient — the allocation returns virtual address
space. Physical RSS commits only when the kernel writes each page:
- `Expand`: during `memcpy`
- `ConstantOfShape`: during `std::fill_n`

With Transparent Huge Pages (THP) enabled, sequential writes can trigger
2 MiB huge-page allocation or direct compaction pressure, amplifying the
latency and memory pressure beyond the raw byte count.

ORT has no `madvise(MADV_NOHUGEPAGE)` or similar hints for large transient
tensors in the materialization kernels.

---

## 4. Hugging Face Production Trigger Paths

### 4.1 `repeat_kv()` — Llama / Qwen / Phi-3

```text
src/transformers/models/llama/modeling_llama.py
```

```python
hidden_states = hidden_states[:, :, None, :, :].expand(batch, H_kv, n_rep, S, D)
return hidden_states.reshape(batch, H_q, S, D)
```

PyTorch eager uses a stride-0 view (no allocation). ONNX lowering emits:
```
Unsqueeze → Expand → Reshape
```

ORT CPU `Expand` materializes this as a full contiguous buffer. For
`B=8, S=32768, H_q=32, D=128, fp16`, KV expansion ≈ 4 GiB.

The expand shape `[B, H_kv, n_rep, S, D]` is a runtime graph input → constant
folding does not fire → the kernel-level check is the only enforcement point.

### 4.2 Dynamic RoPE — Llama / DeepSeek-V2

```python
inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
```

`position_ids.shape[0]` is a runtime value. The expand target is not a
compile-time constant → same bypass as `repeat_kv`.

### 4.3 DeepSeek-V2 MoE routing masks

```python
group_mask = torch.zeros_like(group_scores)        # → ConstantOfShape / Fill
group_mask.scatter_(1, group_idx, 1)
score_mask = group_mask.unsqueeze(-1).expand(...)  # → Expand
```

`zeros_like` lowers to `ConstantOfShape`; the expand shape depends on runtime
routing logits → both kernels fire with no budget check.

### 4.4 Dense attention masks — `masking_utils.py`

```python
attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
```

Prefill with `Q = KV = S` produces `O(B × S²)` materialization.
At `B=1, S=16384, fp32`: ≈ 1 GiB.

---

## 5. QDQ Propagation — the `DQ → Unsqueeze → Expand` Path

`qdq_propagation.cc` allows Q/DQ pairs to propagate through these ops:
```
MaxPool, Reshape, Transpose, Squeeze, Unsqueeze, Slice
```

In a quantized Llama 3.1 graph, QDQ propagation can place a
`DequantizeLinear` immediately before `Expand`, producing:

```
DequantizeLinear → Unsqueeze → Expand → Reshape → MatMul
```

(Case A in the graph scanner.) If QDQ propagates further, the pattern becomes:

```
Unsqueeze → QuantizeLinear → DequantizeLinear → Expand → Reshape → MatMul
```

In both cases the `Expand` input is runtime-derived — constant folding
does not fire, and there is no kernel-level budget check.

The QDQ propagation logic (`PropagateDQForward`, `PropagateQBackward`) does
not check whether the downstream consumer of the propagated edge is a
materialization-risk op. Adding such a check is a secondary hardening option,
but the kernel-level guard is the necessary primary fix.

---

## 6. Prior ORT PR Context

| PR | Change | Effect on this finding |
|---|---|---|
| #28055 | `constant_folding_max_output_size_in_bytes` | Protects folding-time materializations only; no effect on runtime-shape paths |
| #22888 | QDQ redundant-Clip removal | Establishes `IsClipMadeRedundantByQ()`; QDQ layout-adapter propagation design |
| #28178 | N-D weight quantizer adds `ConstantOfShape` helper subgraph | Shows quantizer generates shape-dependent graph topology at export time |
| #3041 | `ConstantOfShape` shape error in MaskRCNN export | Historical precedent for runtime shape → `ConstantOfShape` conflicts |

---

## 7. Patch Direction

The correct fix mirrors `Tile::Compute()`:

**Kernel-level guard (primary):**
- `Expand<T>::Compute()`: check `output_tensor_shape.Size() * sizeof(T) <= limit` before `context->Output()`
- `ConstantOfShape::PrepareCompute()`: same check before `ctx->Output()`

**Session-option hook (follow-up):**
- Expose `session.expand_max_output_bytes` (analogous to `constant_folding_max_output_size_in_bytes`) for user-configurable budgets.

**QDQ fence (optional hardening):**
- In `qdq_propagation.cc`, suppress propagation when the downstream consumer is a materialization-risk op (`Expand`, `ConstantOfShape`, `NonZero`, `Range`).

The kernel guard is the only fix that protects runtime-shape paths regardless
of graph optimization level or whether the model was adversarially crafted.

→ [`../patches/expand_output_size_guard.md`](../patches/expand_output_size_guard.md)
