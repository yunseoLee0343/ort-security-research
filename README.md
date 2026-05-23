# ONNX Runtime Security Research

Two independent findings in [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime),
each rooted in a boundary where compile-time decisions diverge from runtime behavior.

---

## resource-amplification/

**CPU EP resource amplification (DoS)**

A sub-KB schema-valid ONNX model passes all static validation, then forces
10 GB of physical memory commitment at `sess.run()` time via `Expand → RandomUniformLike`.
The kernel layer has no pre-allocation size guard — the verification step never sees the cost.

- `Expand` builds a large tensor as a logical broadcast view (no allocation yet)
- `RandomUniformLike` must write a full output buffer of that shape → unavoidable physical side effects
- `Tile` already has a byte-ceiling guard; `Expand` and `ConstantOfShape` do not

→ [resource-amplification/README.md](resource-amplification/README.md)

**Status:** MSRC filed 2026-02-10 · PoC confirmed · patch proposed · no public fix yet

---

## graph-cleanup-order/

**`Graph::Resolve` cleanup ordering bug**

`Graph::Resolve` calls `CleanUnusedInitializersAndNodeArgs` to prune dead tensors,
but does so before unreachable nodes have been removed from the graph.
Because unreachable nodes are still present when the liveness scan runs,
their referenced `NodeArgs` are incorrectly classified as live — or a `NodeArg`
is removed while its producer node still exists, triggering an assertion.

Fix: `PruneUnreachableNodes()` runs a backward-DFS from graph outputs immediately
before `CleanUnusedInitializersAndNodeArgs`, establishing correct reachability
before any resource cleanup occurs.

→ [graph-cleanup-order/README.md](graph-cleanup-order/README.md)

**Status:** Root-caused · patch reworked against ORT `main` (a1fc916) · upstream PR pending

---

## Common Pattern

Both findings share the same structural root:

```
compile-time assumption          runtime reality
────────────────────────────────────────────────────────────────
"shape is broadcast-valid"    →  Expand materializes full GiB buffer
"liveness scan sees all nodes" → unreachable nodes inflate the live set
```

A later phase assumes the earlier phase left a consistent state — but it doesn't.
