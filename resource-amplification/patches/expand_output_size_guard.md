# Patch: Pre-allocation byte-budget guard for `Expand` and `ConstantOfShape`

`Tile::Compute()` already caps output at 4 GiB via `kMaxTileOutputBytes`.
`Expand` and `ConstantOfShape` have no equivalent guard — they call
`context->Output()` unconditionally regardless of output size.

The patch below adds the same guard to both kernels.

---

## `onnxruntime/core/providers/cpu/tensor/expand.cc`

```diff
--- a/onnxruntime/core/providers/cpu/tensor/expand.cc
+++ b/onnxruntime/core/providers/cpu/tensor/expand.cc
@@ ... @@ Status Expand<T>::Compute(OpKernelContext* context) const {
   // ... broadcast legality check ...

   TensorShape output_tensor_shape(output_shape);
+
+  // Guard against unbounded eager materialization.
+  // ORT CPU Expand has no lazy/strided path: it calls memcpy to fill the full
+  // output buffer. For runtime-derived shapes (e.g. repeat_kv expand in GQA
+  // models), the output can reach GiB scale with no prior ORT-level rejection.
+  // Tile::Compute() already enforces kMaxTileOutputBytes; mirror that here.
+  static constexpr int64_t kMaxExpandOutputBytes = int64_t{4} * 1024 * 1024 * 1024;
+  const auto output_bytes =
+      SafeInt<int64_t>(output_tensor_shape.Size()) * static_cast<int64_t>(sizeof(T));
+  ORT_RETURN_IF_NOT(output_bytes <= kMaxExpandOutputBytes,
+                    "Expand output (", static_cast<int64_t>(output_bytes),
+                    " bytes) exceeds the ", kMaxExpandOutputBytes,
+                    "-byte limit. Use a smaller shape or raise the session limit.");
+
   auto* output_tensor = context->Output(0, output_tensor_shape);
   auto* output_data = output_tensor->MutableData<T>();
```

---

## `onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h`

`PrepareCompute()` is the allocation site for both `ConstantOfShape` and
`ConstantOfShape` variants. The guard belongs here so all specializations
inherit it.

```diff
--- a/onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h
+++ b/onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h
@@ ... @@ Status PrepareCompute(OpKernelContext* ctx, Tensor** output_tensor) const {
   const auto span = shape_tensor->template DataAsSpan<int64_t>();
   TensorShape output_shape(span);

+  // Guard mirrors Tile::Compute() kMaxTileOutputBytes.
+  // ConstantOfShape calls std::fill_n over the entire output, committing
+  // physical RSS for every element. A runtime-supplied shape tensor of
+  // [65536, 65536] yields 16 GiB with no prior rejection.
+  static constexpr int64_t kMaxConstantOfShapeOutputBytes =
+      int64_t{4} * 1024 * 1024 * 1024;
+  const int64_t output_bytes =
+      SafeInt<int64_t>(output_shape.Size()) *
+      static_cast<int64_t>(ctx->Input<Tensor>(0)->DataType()->Size());
+  ORT_RETURN_IF_NOT(output_bytes <= kMaxConstantOfShapeOutputBytes,
+                    "ConstantOfShape output (", output_bytes,
+                    " bytes) exceeds the ", kMaxConstantOfShapeOutputBytes,
+                    "-byte limit.");
+
   (*output_tensor) = ctx->Output(0, output_shape);
```

---

## Why a session-option hook is the right long-term fix

The hardcoded 4 GiB constant matches `Tile` for consistency, but the
correct production design follows `optimization.constant_folding_max_output_size_in_bytes`
(added in PR #28055): expose `session.expand_max_output_bytes` so operators
can enforce a user-configurable budget instead of a binary fail/pass.

The kernel-level guard is the necessary first step — it closes the gap
regardless of graph optimization level or whether the shape is
runtime-derived (where constant folding never fires).
