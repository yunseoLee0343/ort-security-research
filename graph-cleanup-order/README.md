# ort-graph-cleanup-order

**Root-cause analysis and patch for `Graph::Resolve` cleanup ordering bug in ONNX Runtime.**

---

## The Bug

`Graph::Resolve` calls `CleanUnusedInitializersAndNodeArgs` to remove dead tensors after topology resolution. That function determines liveness by scanning the `InputDefs` / `ImplicitInputDefs` of every node currently in the graph — **including nodes that are already unreachable from any graph output**.

Because unreachable nodes are still present when the scan runs, their referenced `NodeArgs` are incorrectly retained as "live". Conversely, a `NodeArg` whose only real consumer is an unreachable node can be removed while that node still holds a pointer to it, triggering an assertion failure.

The ordering invariant that was assumed but never enforced:

```
✗ current:  TopologicalSort → TypeShapeInference → CleanUnusedInitializersAndNodeArgs
✓ required: TopologicalSort → TypeShapeInference → PruneUnreachableNodes → CleanUnusedInitializersAndNodeArgs
```

**Affected issues:** #10677 (assertion), #7641, #14694 (implicit input loss in `If`/`Loop`/`Scan` subgraphs).

---

## The Fix

A new method `Graph::PruneUnreachableNodes()` is inserted into `finalize_func` inside `Graph::Resolve`, immediately before `CleanUnusedInitializersAndNodeArgs`. It:

1. **Backward DFS** from producer nodes of every graph output — marks the reachable set.
2. **Fixed-point subgraph walk** — for every live `If`/`Loop`/`Scan` node, pulls in outer-scope producers referenced via `ImplicitInputDefs` (edges not visible to the DFS).
3. **Removes dead nodes** — skips nodes that produce a graph output (double-check) and nodes with no registered schema (custom ops may have unknown side effects).

See [`patches/graph_resolve_prune_before_cleanup.patch`](patches/graph_resolve_prune_before_cleanup.patch) for the full diff (107 lines across `graph.h` and `graph.cc`).

---

## Repository Layout

```
patches/
  graph_resolve_prune_before_cleanup.patch   ← unified diff, apply with: patch -p1 < ...
analysis/
  graph_resolve_cleanup_order_design.md      ← PR #27141 background, design rationale
docs/
  patch_rationale.md                         ← comparison vs. original PR; bug-by-bug breakdown
```

---

## Background: PR #27141

An earlier attempt ([PR #27141](https://github.com/microsoft/onnxruntime/pull/27141), closed Jan 2026) introduced the same concept but had four divergences from current `main`:

| Issue | Original PR | This rework |
|---|---|---|
| Lambda call syntax | `(graph)` — wrong | `(graph)` → correct `graph.PruneUnreachableNodes()` |
| `gsl::span` type | `gsl::span<const Node* const>` — removed in ORT | `gsl::make_span(roots)` |
| Build guard | No `ORT_RETURN_IF_ERROR` wrapper | Wrapped correctly |
| Custom-op safety | No `node.Op() == nullptr` guard | Guard added |

The rework is rebased on the `a1fc916` commit of `microsoft/onnxruntime` (`main`, May 2026) and passes the four-check review above.

---

## Applying the Patch

```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
patch -p1 < /path/to/patches/graph_resolve_prune_before_cleanup.patch
```

Tests to run after applying:

```bash
# Graph resolve unit tests
./build.sh --config Debug --build_shared_lib --parallel
ctest -R GraphTest --output-on-failure
```

---

## Status

| Item | Status |
|---|---|
| Root-cause confirmed | Yes — `CleanUnusedInitializersAndNodeArgs` runs before dead nodes are removed |
| Patch implemented | Yes — `PruneUnreachableNodes()` in `graph.cc` + declaration in `graph.h` |
| Rebased on current `main` | Yes — base commit `a1fc916` |
| Unit tests written | Pending |
| Upstream PR submitted | Pending |
