# Patch: Prune unreachable nodes before resource cleanup in Graph::Resolve

**Files:** `include/onnxruntime/core/graph/graph.h`, `onnxruntime/core/graph/graph.cc`  
**Branch:** `fix/graph-resolve-prune-before-cleanup`  
**Fixes:** #10677, #7641, #14694  
**Status:** Rework of closed PR #27141 — applies cleanly to current `main` (HEAD `a1fc916`)

---

## Root Cause

`Graph::Resolve` calls `CleanUnusedInitializersAndNodeArgs` to remove dead
initializers and NodeArgs. Liveness is determined by scanning `InputDefs` and
`ImplicitInputDefs` of every node currently in the graph — but this includes
**unreachable nodes** (nodes with no path to any graph output).

Two failure modes result:

**#10677 — Assertion failure**
An unreachable node keeps a NodeArg "live" during cleanup, so the NodeArg
survives. On the next Resolve cycle the node may be absent but the initializer
remains in `name_to_initial_tensor_`. `CleanUnusedInitializersAndNodeArgs`
then calls `GetNodeArg(name)` and hits:
```
ORT_ENFORCE(initializer_node_arg != nullptr, "Cannot find NodeArgs for [", name, "]");
```

**#7641 / #14694 — Implicit input loss**
An initializer used only inside an `If` / `Loop` / `Scan` subgraph is not
visible to the outer graph's NodeArg scan. The outer cleanup removes it;
the subgraph later fails to find it.

---

## Fix

Add `Graph::PruneUnreachableNodes()`, called in `finalize_func` inside
`Graph::Resolve`, **before** `CleanUnusedInitializersAndNodeArgs`.

The function:
1. Backward DFS from every graph output producer → builds `live` node set
2. Fixed-point expansion → pulls in outer-scope producers that supply implicit
   inputs to live `If`/`Loop`/`Scan` subgraph nodes
3. Removes nodes outside `live`, with two conservative guards:
   - skip if `NodeProducesGraphOutput` (safety double-check)
   - skip if `node.Op() == nullptr` (custom ops / unknown side effects)

After pruning, `CleanUnusedInitializersAndNodeArgs` only sees reachable nodes
so its NodeArg liveness scan is sound.

---

## Diff

```diff
diff --git a/include/onnxruntime/core/graph/graph.h b/include/onnxruntime/core/graph/graph.h
index 92d9105..8b4f09e 100644
--- a/include/onnxruntime/core/graph/graph.h
+++ b/include/onnxruntime/core/graph/graph.h
@@ -1920,6 +1920,14 @@ class Graph {  // NOLINT(clang-analyzer-optin.performance.Padding): preserve exi
   // Iterate this Graph instance and all subgraphs, calling the provided function for each.
   common::Status ForThisAndAllSubgraphs(const std::vector<Graph*>& subgraphs, std::function<Status(Graph&)> func);
 
+  // Prune nodes unreachable from graph outputs before resource cleanup.
+  // Must be called before CleanUnusedInitializersAndNodeArgs so that liveness
+  // is derived from node reachability, not syntactic NodeArg scans.
+  // Fixes: assertion in CleanUnusedInitializersAndNodeArgs when a NodeArg is
+  // removed while its producer node still exists (issue #10677), and implicit
+  // input loss for If/Loop/Scan subgraphs (issues #7641, #14694).
+  Status PruneUnreachableNodes();
+
   // Clear all unused initializers and NodeArgs
   void CleanUnusedInitializersAndNodeArgs(const std::unordered_set<std::string>* initializer_names_to_preserve = nullptr);
 
diff --git a/onnxruntime/core/graph/graph.cc b/onnxruntime/core/graph/graph.cc
index 6f42174..3ffd71c 100644
--- a/onnxruntime/core/graph/graph.cc
+++ b/onnxruntime/core/graph/graph.cc
@@ -3683,6 +3683,75 @@ Status Graph::PerformTypeAndShapeInferencing(const ResolveOptions& options) {
   return Status::OK();
 }
 
+Status Graph::PruneUnreachableNodes() {
+  // Step 1: collect producer nodes of every graph output as DFS roots.
+  InlinedVector<const Node*> roots;
+  roots.reserve(GetOutputs().size());
+  for (const auto* output : GetOutputs()) {
+    if (const Node* producer = GetProducerNode(output->Name()); producer != nullptr) {
+      roots.push_back(producer);
+    }
+  }
+
+  // Step 2: backward DFS — mark every node reachable from a graph output.
+  InlinedHashSet<const Node*> live;
+  live.reserve(static_cast<size_t>(NumberOfNodes()));
+  ReverseDFSFrom(
+      gsl::make_span(roots),
+      [&live](const Node* n) { live.insert(n); },
+      /*leave=*/nullptr);
+
+  // Step 3: fixed-point — pull in outer-scope producers consumed as implicit
+  // inputs by nodes inside live If/Loop/Scan subgraphs.
+  // Needed because those producers are not connected via graph edges visible
+  // to the DFS above, yet their removal would break subgraph execution.
+  bool changed = true;
+  while (changed) {
+    changed = false;
+    for (const auto& node : Nodes()) {
+      if (!live.count(&node) || !node.ContainsSubgraph()) {
+        continue;
+      }
+      for (const auto& sg : node.GetSubgraphs()) {
+        for (const auto& sg_node : sg->Nodes()) {
+          for (const auto* implicit : sg_node.ImplicitInputDefs()) {
+            // The implicit input may be produced in the outer graph.
+            if (const Node* p = GetProducerNode(implicit->Name()); p != nullptr) {
+              if (live.insert(p).second) {
+                changed = true;
+              }
+            }
+          }
+        }
+      }
+    }
+  }
+
+  // Step 4: remove dead nodes.
+  // Preserve nodes that produce graph outputs (double-check — DFS should have
+  // caught them, but NodeProducesGraphOutput is cheap insurance).
+  // Preserve nodes with no registered schema: custom ops or ops loaded from a
+  // minimal build may have side effects unknown to us.
+  InlinedVector<NodeIndex> to_remove;
+  for (const auto& node : Nodes()) {
+    if (live.count(&node)) {
+      continue;
+    }
+    if (NodeProducesGraphOutput(node)) {
+      continue;
+    }
+    if (node.Op() == nullptr) {
+      continue;
+    }
+    to_remove.push_back(node.Index());
+  }
+  for (NodeIndex idx : to_remove) {
+    RemoveNode(idx);
+  }
+
+  return Status::OK();
+}
+
 Status Graph::Resolve(const ResolveOptions& options) {
   if (parent_graph_) {
     // Resolve must start at the top level graph in-order to handle outer scope
@@ -3737,6 +3806,7 @@ Status Graph::Resolve(const ResolveOptions& options) {
             // this can happen to ResolveContext.inputs_and_initializers during CleanUnusedInitializersAndNodeArgs.
             graph.resolve_context_.Clear();
 
+            ORT_RETURN_IF_ERROR(graph.PruneUnreachableNodes());
             graph.CleanUnusedInitializersAndNodeArgs(options.initializer_names_to_preserve);
             graph.GraphResolveNeeded(false);
```

---

## What Changed vs. PR #27141

| | PR #27141 (closed) | This rework |
|---|---|---|
| Base commit | `727db0d` | `a1fc916` (current `main`) |
| Call in lambda | `PruneUnreachableNodes()` — won't compile in lambda | `graph.PruneUnreachableNodes()` ✓ |
| span construction | bare `roots` — type mismatch | `gsl::make_span(roots)` ✓ |
| Build guard | `ORT_EXTENDED_MINIMAL_BUILD` (too wide) | `ORT_MINIMAL_BUILD` only (matches `Resolve`) ✓ |
| Custom op safety | not guarded | `node.Op() == nullptr` guard ✓ |
