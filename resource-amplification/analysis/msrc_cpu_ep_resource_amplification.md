# Report: ONNX Runtime (CPU EP) — Sub-KB ONNX Model Forces Run-Time Physical Memory Commit via Expand + RandomUniformLike (Resource Amplification DoS)

## Title

ONNX Runtime (CPU EP): Sub-KB ONNX model forces run-time physical memory commit via Expand + RandomUniformLike, causing deterministic resource amplification (DoS)

---

## Summary

In ONNX Runtime’s CPU Execution Provider, a very small ONNX model (≈0.5KB) can expand a scalar into a multi-GiB _logical_ tensor view via `Expand`, then force full materialization via `RandomUniformLike`, which performs sequential writes over the entire output buffer. This produces a fast, reproducible spike in process RSS (peak +8.4GB in my environment) and significant CPU time consumption.

The key point is not “the model creates a big tensor,” but that **dynamic shape handling combined with run-time memory planning decisions creates a resource amplification primitive**: the attacker-controlled artifact is tiny, but the runtime deterministically consumes disproportionate physical resources (memory/CPU). This is reproducible using only default ONNX Runtime behavior (no custom ops, monkeypatching, hooks, or external dependencies).

---

## Security Impact

**Impact type:** Denial of Service (DoS) — memory exhaustion + CPU time exhaustion  
**Severity (hosted inference / multi-tenant assumption):** High ~ Critical

**Threatened deployments:** BYOM (bring-your-own-model), model upload/evaluation services, dynamic session creation pipelines, and any trust boundary where untrusted or semi-trusted ONNX graphs are executed.

**Expected outcomes:**

- Process OOM kill / service crash
- Container eviction (Kubernetes)
- Node-level memory pressure (reclaim/swap) and knock-on latency
- Noisy neighbor effects impacting adjacent workloads
- Cascading SLA/SLO violations and service availability incidents

> This report is not a “performance complaint.” It concerns a structural risk where **untrusted programs (ONNX graphs) can deterministically force run-time physical resource commitment**. Because the model appears small at load/validation time but explodes at execution time, it creates a **verification–execution gap** requiring platform-level mitigations (policy/quotas/guards), not just “operators should set limits.”

---

## Affected Component

ONNX Runtime **CPUExecutionProvider** execution of graphs containing the operator sequence:

- `Expand`: constructs a large broadcasted _logical view_ without immediately allocating a large backing buffer.
- `RandomUniformLike`: generates a tensor with the same shape as its input, requiring full output allocation and a write across the entire buffer (forcing physical commit).
- `ReduceSum` / `Where`: prevents dead-code elimination and provides scalar outputs proving the payload executed.

---

## Attack Preconditions / Threat Model

Any environment that executes untrusted/semi-trusted ONNX graphs, including:

- User model upload/evaluation endpoints (BYOM, model marketplaces)
- Internal automation pipelines executing externally sourced models
- Multi-tenant inference services (request-scoped dynamic session/graph creation)
- Batch pipelines where the graph may be attacker-controlled

---

## Proof of Concept Overview

**PoC file:** `poc_dos.py` (attached)  
**Model generated:** `v16_dos.onnx` created at runtime (≈0.5KB)  
**Logical tensor shape:** `[32768, 32768]` float32 (≈4GiB logical size)  
**Trigger:** `RandomUniformLike` fully materializes the `Expand` view, forcing page commits via sequential writes.

---

## Reproduction Steps

1. Install dependencies:
   - `onnx`, `onnxruntime`, `numpy`, `psutil`
2. Run on Linux:
   - `python poc_dos.py`
3. Observe:
   - ONNX model file size is sub-KB
   - RSS peak rises sharply during `sess.run()`
   - Significant wall time / CPU time is consumed

---

## Observed Results (Representative Run)

**Model file size:** 0.456 KB  
**Baseline RSS:** 1800.41 MB  
**RSS after session init:** 1800.41 MB  
**Peak RSS during run:** 10198.77 MB  
**Delta (after init → peak):** +8398.36 MB  
**Wall time:** 12.59 s  
**CPU time (user/system):** 7.08 s / 4.04 s  
**Output:** Scalar results exist, confirming the graph was not optimized away.

This demonstrates that a <0.5KB input can force >8.4GB of physical memory commitment and meaningful CPU consumption. In “file bytes → RSS bytes” terms, this is on the order of **tens of millions** in amplification.

---

## Root Cause / Technical Explanation

- `Expand` can represent a large broadcasted tensor as a **logical view** according to broadcast semantics, often without allocating a full backing buffer at that stage.
- `RandomUniformLike` must create an output tensor of the same shape as the input. In practice this requires:
  1. allocating the full output buffer, and
  2. writing across the full buffer to populate random values, which forces physical memory commitment.
- Therefore, an attacker controls **shape / element count**, which becomes an input to memory allocation and commitment decisions. `RandomUniformLike` acts as a **forced materialization boundary** converting a logical shape into unavoidable physical side effects (large commit + write).
- The graph includes a runtime-bound anti-fold chain and a reduction so that the payload is not eliminated by constant folding / graph optimization.

---

## Why This Matters Beyond “Just OOM”

In cloud and multi-tenant inference, memory pressure is not a single-process event:

- A single request/model can trigger cache eviction, reclaim/swap, and node-level pressure, degrading other workloads (noisy neighbor).
- While this PoC demonstrates DoS, a deterministic memory pressure/churn primitive can **increase risk and exploit reliability** if other vulnerabilities exist in the same stack (e.g., stale buffer reuse, uninitialized padding exposure).
- **Important:** This report does not claim an information leak occurs here.
  The point is that resource amplification primitives materially increase the platform’s security and reliability risk in multi-tenant environments, strengthening the case for explicit run-time guards/quotas.

---

## Deployment-Credible Cloud Scenario

Assume a managed inference endpoint (or an internal platform) allows ONNX model uploads and performs primarily schema/opset validation, then runs ONNX Runtime on shared CPU nodes.

An attacker uploads a tiny model that passes validation but triggers multi-GiB allocation/commit at execution time, causing worker OOM, container eviction, and node resource pressure, destabilizing adjacent workloads and producing an availability incident.

---

## Suggested Fixes / Mitigations

These mitigations are incremental; any single item can substantially reduce risk:

1. **Run-time memory budget / quota**
   - Enforce a maximum alloc/commit byte budget per session/run for untrusted graphs.

2. **Shape policy limits**
   - Apply element/byte ceilings for sensitive ops (`RandomUniformLike`, `ConstantOfShape`, `Tile`, large `Reshape`, etc.).

3. **Operator-level guard**
   - Before `RandomUniformLike` allocates/writes, validate output size and fail fast with a clear error when exceeding policy.

4. **Operational guidance**
   - Require/strongly recommend container memory/CPU limits and request timeouts for untrusted model execution.

---

## Key Insight (Design Perspective) — **Please Read**

This issue is not “one operator bug.” It exposes a structural boundary where dynamic shape execution interacts with run-time memory planning:

**Symbolic Shape → Memory Lifetime Inference → Lazy→Eager Materialization → (Async/Deferred Execution Gap) → Unprovable Safety / Amplification Risk**

- Symbolic shapes are not merely “descriptive metadata.” In practice they are **inputs into memory lifetime inference and allocation decisions**.
- The transition from `Expand` (lazy logical view) to `RandomUniformLike` (eager full materialization) is where abstract computation turns into **physical side effects** (large commit + write). This transition effectively acts as a **security boundary**.
- When this boundary occurs at run time, static validation can pass while execution cost explodes — a **verification–execution gap**.
- In general, deferred/asynchronous execution models can separate “logical completion” from “physical completion,” creating regions where safety/resource bounds cannot be proven purely statically, and must be enforced by run-time policy.

> **This insight holds even within the PoC’s DoS scope** and explains why **run-time policies (budgets/guards)** are required in multi-tenant settings.

---

## Attachments

- `poc_dos.py` (instrumented reproduction script; prints normalized report)
- `v16_dos.onnx` is generated at runtime by the PoC (sub-KB)

---

## Disclosure / Timeline

- Report date: 2026-02-10
- Vendor contact: MSRC
