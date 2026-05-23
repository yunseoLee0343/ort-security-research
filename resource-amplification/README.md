# ort-resource-amplification

Root-cause analysis of an ONNX Runtime CPU EP resource amplification bug,
traced from a synthetic DoS PoC through production GQA model paths to a concrete patch.

---

## The Bug

ORT's CPU `Expand` kernel has no pre-allocation byte-budget guard.
It calls `context->Output(0, output_tensor_shape)` unconditionally —
allocating and fully writing a contiguous output buffer for any shape,
regardless of size.

`Tile::Compute()` already enforces a 4 GiB cap via `kMaxTileOutputBytes`.
`Expand` and `ConstantOfShape` do not.

```
compile-time: "shape is broadcast-compatible"   ← ORT accepts
runtime:       Expand materializes full buffer   ← no budget check
               memcpy writes every byte          ← RSS spike
```

Because the shape is runtime-derived (graph input, not initializer),
constant folding never fires — the guard added in PR #28055
(`constant_folding_max_output_size_in_bytes`) is bypassed entirely.

---

## Research Chain

### 1. MSRC PoC — synthetic model, confirmed DoS

`poc/poc_dos.py` generates a 0.46 KB ONNX model that forces 10,198 MB
peak RSS via `Expand → RandomUniformLike`.
The shape tensor is a graph input; ORT passes all validation before allocating.

→ [`poc/poc_dos.py`](poc/poc_dos.py)  
→ [`poc/reproduction_output.txt`](poc/reproduction_output.txt)  
→ [`analysis/msrc_cpu_ep_resource_amplification.md`](analysis/msrc_cpu_ep_resource_amplification.md)

### 2. HF Audit — production models are also affected

The same materialization pattern appears in every GQA/MQA model shipped on
Hugging Face. `repeat_kv()`, used by Llama, Qwen, Phi-3, and others,
expands KV heads before attention:

```python
# PyTorch eager: stride-0 broadcast view (zero cost)
hidden_states = hidden_states[:, :, None, :, :].expand(B, H_kv, n_rep, S, D)
return hidden_states.reshape(B, H_q, S, D)
```

ONNX lowering emits `Unsqueeze → Expand → Reshape`.
ORT CPU EP executes the `Expand` as a full eager `memcpy` — no lazy path exists.

For a typical inference call (`B=8, S=32768, H_q=32, D=128, fp16`):
```
KV expansion ≈ 2 × B × S × H_q × D × 2 bytes ≈ 4 GiB
```

This is not a synthetic attack. It is the default GQA inference path,
triggered by normal sequence lengths with no attacker involvement.

Dynamic RoPE, DeepSeek-V2 MoE routing masks, and dense attention masks in
`masking_utils.py` are additional production trigger paths in the same class.

→ [`analysis/hf_ort_resource_amplification_audit.md`](analysis/hf_ort_resource_amplification_audit.md)

### 3. QDQ / Constant-Folding Analysis — why the existing guard doesn't fire

ORT's constant folding (PR #28055) added `constant_folding_max_output_size_in_bytes`
to cap folding-time materializations. It has no effect here because:

- `repeat_kv`'s expand shape is a **runtime graph input**,
  so `AllNodeInputsAreConstant` is false → constant folding never runs.
- QDQ propagation (`qdq_propagation.cc`) allows Q/DQ pairs to cross
  `Unsqueeze`, placing a `DequantizeLinear` immediately before `Expand`
  in quantized Llama 3.1:

  ```
  DequantizeLinear → Unsqueeze → Expand → Reshape → MatMul
  ```

  A graph scanner confirms this path is present in exported QDQ Llama graphs.

`Tile` already blocks unbounded materialization at the kernel level,
independent of optimization settings. `Expand` and `ConstantOfShape` need
the same treatment.

→ [`analysis/constant_folding_qdq_analysis.md`](analysis/constant_folding_qdq_analysis.md)

### 4. Patch — kernel-level byte-budget guard

Add the same guard to `Expand<T>::Compute()` and `ConstantOfShape::PrepareCompute()`
that `Tile::Compute()` already has:

```cpp
// expand.cc — before context->Output()
static constexpr int64_t kMaxExpandOutputBytes = int64_t{4} * 1024 * 1024 * 1024;
const auto output_bytes =
    SafeInt<int64_t>(output_tensor_shape.Size()) * static_cast<int64_t>(sizeof(T));
ORT_RETURN_IF_NOT(output_bytes <= kMaxExpandOutputBytes, ...);
```

The kernel-level guard fires regardless of whether the shape is compile-time
constant or runtime-derived, closing the gap left by PR #28055.

→ [`patches/expand_output_size_guard.md`](patches/expand_output_size_guard.md)

---

## Repository Layout

```
analysis/
  msrc_cpu_ep_resource_amplification.md   MSRC submission report
  hf_ort_resource_amplification_audit.md  Production model trigger paths (GQA, RoPE, MoE, attention masks)
  constant_folding_qdq_analysis.md        Why constant folding doesn't fire; QDQ propagation chain

patches/
  expand_output_size_guard.md             Pre-allocation byte-budget guard for Expand + ConstantOfShape

poc/
  poc_dos.py                              Synthetic reproducer (generates v16_dos.onnx)
  reproduction_output.txt                 Observed output: 0.46 KB model → 10,198 MB peak RSS
```

---

## Status

| Item | Status |
|---|---|
| Synthetic PoC | Confirmed — 0.46 KB model → 10,198 MB peak RSS |
| Production trigger | Confirmed — `repeat_kv` `Expand` in GQA models (Llama, Qwen, Phi-3) |
| QDQ propagation path | Identified — DQ→Unsqueeze→Expand→Reshape→MatMul in quantized Llama |
| Patch | Proposed — byte-budget guard in `Expand<T>::Compute()` + `ConstantOfShape::PrepareCompute()` |
| MSRC | Filed 2026-02-10 · no public fix yet |
| Upstream PR | Pending |
