# ONNX Runtime PR #27141 — Content

- PR: [[Graph] Ensure node pruning precedes resource cleanup in Resolve](https://github.com/microsoft/onnxruntime/pull/27141)
- Repository: `microsoft/onnxruntime`
- Author: `yunseoLee0343`
- State: `closed`
- Merged: `false`
- Base branch: `main`
- Head branch: `fix/cleanup-order-issue`
- Base SHA: `727db0d3dc9f7dc5958891d80c1073ef7190f316`
- Head SHA: `65a2339e37e326cdd7707857c57ed5edd3f55b61`
- Created: `2026-01-25T14:48:01Z`
- Updated: `2026-01-26T00:51:05Z`
- Closed: `2026-01-26T00:51:05Z`
- Commits: `1`
- Changed files: `2`
- Additions: `75`
- Deletions: `0`

---

## Summary

This PR introduces **Graph Output-based Backward Reachability Analysis** during the `Graph::Resolve` phase. It addresses structural inconsistencies where `CleanUnusedInitializersAndNodeArgs` would prematurely remove `NodeArgs` or `Initializers` that were still "semantically" required by unreachable nodes, or conversely, fail to protect initializers used only within control-flow subgraphs.

## Motivation & Problem Statement

The existing cleanup logic relied on local syntactic scans of NodeArgs. This led to several failure modes:

* **Assertion Failures**: NodeArgs were removed while their producer nodes remained in the graph, violating the producer-consumer invariant (**Issue #10677**).
* **Implicit Input Misidentification**: Initializers used exclusively within `If`, `Loop`, or `Scan` subgraphs were incorrectly flagged as "unused" and removed (**Issue #7641, #14694**).
* **Semantic Shift**: Removing optional inputs that are part of the model's external contract changed the model's interface unexpectedly.

## Design Principle

> **"Node reachability provides a stable foundation for determining resource liveness."**

The liveness of a `NodeArg` or `Initializer` must be a derived property of node reachability. By pruning unreachable nodes first, we ensure that the remaining graph is semantically sound before any resource cleanup occurs.

## Key Changes

1. **Backward DFS from Roots**: Identifies all nodes reachable from graph outputs using a robust DFS traversal.
2. **Fixed-point Subgraph Propagation**: Conservatively marks all nodes within subgraphs and their implicit dependencies as live to prevent accidental deletion of control-flow resources.
3. **Side-effect Preservation**: Ensures nodes with side effects (checked via `OpSchema`) are never pruned, maintaining model integrity.
4. **Strategic Integration**: Placed in `Graph::Resolve` immediately after `BuildConnections`, providing a clean state for subsequent optimizations.

## Related Issues

* **Fixes #10677**: Prevents `initializer_node_arg != nullptr` assertion by pruning nodes before resources.
* **Fixes #7641, #14694**: Properly preserves implicit inputs within control-flow subgraphs.
* Improves constant folding predictability by ensuring a consistent graph state.

---

## Comment

Author comment:

```text
@microsoft-github-policy-service agree
```
