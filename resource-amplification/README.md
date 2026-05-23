# ort-resource-amplification

Root-cause analysis and patch for an ONNX Runtime CPU EP resource amplification bug:
a sub-KB schema-valid model forces GiB-scale physical memory commitment at inference time.

---

## The Theme

ORT validates graphs and plans resources at compile / graph-resolve time.
A sub-KB ONNX model can pass all static validation, then force GiB-scale
memory commitment at `sess.run()` time — **the verification step never sees
the allocation cost**.

---

## CPU EP Resource Amplification (DoS)

**Layer:** CPU Execution Provider kernel (`expand.cc`)  
**Filed:** MSRC, 2026-02-10

A sub-KB ONNX model passes all schema and shape validation,
then forces 8–10 GB of physical memory commitment at `sess.run()` time
via `Expand → RandomUniformLike`.

`Expand` constructs a large tensor as a *logical broadcast view*
(no physical allocation yet).
`RandomUniformLike` must write a full output buffer of the same shape,
converting the abstract view into unavoidable physical side effects.

The attacker controls shape via a tiny runtime-input tensor;
the verification step never sees the allocation cost.

**Key finding:** `Tile` already has a configurable byte-ceiling guard;
`Expand` and `ConstantOfShape` do not. The proposed patch closes this gap.

→ [`patches/expand_output_size_guard.md`](patches/expand_output_size_guard.md)  
→ [`poc/poc_dos.py`](poc/poc_dos.py) — 0.46 KB model → 10,198 MB peak RSS

---

## Related

The `Graph::Resolve` cleanup ordering bug (a separate graph-layer issue) lives in its own repo:
→ [ort-graph-cleanup-order](../ort-graph-cleanup-order)

---

## Repository Structure

```
analysis/
  msrc_cpu_ep_resource_amplification.md   MSRC submission report
  hf_ort_resource_amplification_audit.md  Extended HF model path audit
  constant_folding_qdq_analysis.md        Constant-folding / QDQ boundary analysis
  llama_graph_risk_scan.txt               LLaMA 3.2 ONNX static risk scan

patches/
  expand_output_size_guard.md             Pre-allocation byte-limit guard in Expand<T>::Compute()

poc/
  poc_dos.py                              Reproducer (generates v16_dos.onnx)
  reproduction_output.txt                 Observed output
```

---

## Status

| Item | Status |
|---|---|
| Finding | Confirmed — PoC reproduces 10 GB allocation from 0.46 KB model |
| Patch | Proposed — byte-ceiling guard in `Expand<T>::Compute()` |
| Upstream | MSRC filed 2026-02-10; no public fix yet |
