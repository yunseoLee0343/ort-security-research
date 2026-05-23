아래 diff는 **ORT checkout 루트에 그대로 적용하는 풀 패키지**다. 구성은 ① Case A synthetic graph 생성, ② quantized Llama 3.1 ONNX에서 실제 `DequantizeLinear -> Unsqueeze -> Expand -> Reshape -> MatMul` 스캔, ③ 실제 ORT profiler 실행, ④ 필요 시 `Expand<T>::Compute()`의 `context->Output()` 직전 실제 pre-allocation 계측까지 포함한다. ORT `Expand`는 현재 `output_tensor_shape` 생성 직후 `context->Output(0, output_tensor_shape)`를 호출하므로, 계측 위치는 allocation 직전이 맞다.  QDQ propagation은 `Unsqueeze/Reshape/Transpose/Squeeze/Slice`를 통과 가능한 op로 다루고 edge에 Q/DQ pair를 실제 삽입하므로, scanner는 그 source path를 그대로 겨냥한다.  

````diff
diff --git a/onnxruntime/core/providers/cpu/tensor/expand.cc b/onnxruntime/core/providers/cpu/tensor/expand.cc
index 6d299282f3..f5a74c3a11 100644
--- a/onnxruntime/core/providers/cpu/tensor/expand.cc
+++ b/onnxruntime/core/providers/cpu/tensor/expand.cc
@@ -3,11 +3,64 @@
 
 #include "expand.h"
 #include <cmath>
+#include <cstdlib>
+#include <fstream>
+#include <mutex>
+#include <sstream>
 #include <core/common/safeint.h>
 
 namespace onnxruntime {
 
+namespace {
+
+template <typename Dims>
+std::string DimsToJsonArray(const Dims& dims) {
+  std::ostringstream os;
+  os << "[";
+  for (size_t i = 0; i < dims.size(); ++i) {
+    if (i != 0) {
+      os << ",";
+    }
+    os << dims[i];
+  }
+  os << "]";
+  return os.str();
+}
+
+std::mutex& ExpandProbeLogMutex() {
+  static std::mutex mutex;
+  return mutex;
+}
+
+template <typename InputDims, typename OutputDims>
+void LogExpandPreAllocationProbe(const char* path,
+                                 const InputDims& input_shape,
+                                 const OutputDims& output_shape,
+                                 size_t element_size,
+                                 size_t requested_bytes) {
+  if (path == nullptr || path[0] == '\0') {
+    return;
+  }
+
+  std::lock_guard<std::mutex> lock(ExpandProbeLogMutex());
+  std::ofstream out(path, std::ios::app);
+  if (!out.good()) {
+    return;
+  }
+
+  out << "{"
+      << "\"event\":\"ExpandPreAllocation\","
+      << "\"phase\":\"before_context_Output\","
+      << "\"input_shape\":" << DimsToJsonArray(input_shape) << ","
+      << "\"output_shape\":" << DimsToJsonArray(output_shape) << ","
+      << "\"element_size\":" << element_size << ","
+      << "\"requested_bytes\":" << requested_bytes
+      << "}\n";
+}
+
+}  // namespace
+
 #define REG_EXPAND_KERNEL(TYPE)                                                    \
   ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                        \
       Expand,                                                                      \
@@ -66,6 +119,14 @@ Status Expand<T>::Compute(OpKernelContext* context) const {
   }
 
   TensorShape output_tensor_shape(output_shape);
+  if (const char* expand_probe_log_path = std::getenv("ORT_EXPAND_PROBE_LOG");
+      expand_probe_log_path != nullptr && expand_probe_log_path[0] != '\0') {
+    const size_t output_num_elements = onnxruntime::narrow<size_t>(output_tensor_shape.Size());
+    const size_t requested_bytes = SafeInt<size_t>(output_num_elements) * sizeof(T);
+    LogExpandPreAllocationProbe(expand_probe_log_path, input_shape, output_shape, sizeof(T),
+                                requested_bytes);
+  }
+
   auto* output_tensor = context->Output(0, output_tensor_shape);
   auto* output_data = output_tensor->MutableData<T>();
   auto* output_dims = output_shape.data();
diff --git a/tools/qdq_expand_probe/README.md b/tools/qdq_expand_probe/README.md
new file mode 100644
index 0000000000..8bd2af1ed0
--- /dev/null
+++ b/tools/qdq_expand_probe/README.md
@@ -0,0 +1,141 @@
+# QDQ / Expand Case A Probe
+
+This package is for producing non-mock evidence for the following graph:
+
+```text
+DequantizeLinear
+  -> Unsqueeze
+  -> Expand
+  -> Reshape
+  -> MatMul
+```
+
+The target real-model interpretation is Llama 3.1 GQA/MQA `repeat_kv`:
+
+```text
+[B, H_kv, S, D]
+  -> Unsqueeze
+  -> Expand [B, H_kv, n_rep, S, D]
+  -> Reshape [B, H_q, S, D]
+```
+
+where:
+
+```text
+H_q = H_kv * n_rep
+```
+
+## Files
+
+```text
+tools/qdq_expand_probe/
+  README.md
+  requirements.txt
+  make_case_a_graph.py
+  profile_onnx.py
+  scan_llama31_qdq_expand.py
+  run_package.sh
+```
+
+The optional ORT source instrumentation is in:
+
+```text
+onnxruntime/core/providers/cpu/tensor/expand.cc
+```
+
+It is gated by:
+
+```bash
+export ORT_EXPAND_PROBE_LOG=/absolute/path/to/expand_probe.jsonl
+```
+
+If the environment variable is unset, the source patch is inert.
+
+## Install Python dependencies
+
+From the ONNX Runtime repository root:
+
+```bash
+python -m venv .venv-qdx
+source .venv-qdx/bin/activate
+pip install -r tools/qdq_expand_probe/requirements.txt
+```
+
+## Generate the synthetic Case A graph
+
+```bash
+mkdir -p artifacts/qdq_expand_probe
+
+python tools/qdq_expand_probe/make_case_a_graph.py \
+  --out artifacts/qdq_expand_probe/case_a_dq_unsqueeze_expand_reshape_matmul.onnx \
+  --batch 1 \
+  --num-key-value-heads 2 \
+  --num-key-value-groups 4 \
+  --seq 128 \
+  --head-dim 16
+```
+
+Expected graph:
+
+```text
+x_uint8
+  -> DequantizeLinear
+  -> Unsqueeze
+  -> Expand
+  -> Reshape
+  -> MatMul
+```
+
+The `expand_shape` and `reshape_shape` are graph inputs, not constants. The runtime feed supplies:
+
+```text
+expand_shape  = [B, H_kv, n_rep, S, D]
+reshape_shape = [B, H_q, S, D]
+```
+
+## Run real ORT profiling on the synthetic graph
+
+```bash
+python tools/qdq_expand_probe/profile_onnx.py \
+  --model artifacts/qdq_expand_probe/case_a_dq_unsqueeze_expand_reshape_matmul.onnx \
+  --case-a \
+  --outdir artifacts/qdq_expand_probe/case_a_profile \
+  --batch 1 \
+  --num-key-value-heads 2 \
+  --num-key-value-groups 4 \
+  --seq 128 \
+  --head-dim 16 \
+  --iters 5 \
+  --require-expand
+```
+
+Outputs:
+
+```text
+artifacts/qdq_expand_probe/case_a_profile/
+  ort_profile_*.json
+  expand_events.json
+  memory_trace.jsonl
+  run_summary.json
+```
+
+This is not a mocked profile. The script calls `onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"])`
+with ORT profiling enabled and fails if no `Expand` kernel event is found when `--require-expand` is set.
+
+## Run with direct ORT source instrumentation
+
+Build a local ORT wheel after applying this diff:
+
+```bash
+./build.sh --config RelWithDebInfo --build_wheel --parallel --skip_tests
+pip install --force-reinstall build/Linux/RelWithDebInfo/dist/*.whl
+```
+
+Then run:
+
+```bash
+python tools/qdq_expand_probe/profile_onnx.py \
+  --model artifacts/qdq_expand_probe/case_a_dq_unsqueeze_expand_reshape_matmul.onnx \
+  --case-a \
+  --outdir artifacts/qdq_expand_probe/case_a_profile_instrumented \
+  --expand-probe-log artifacts/qdq_expand_probe/expand_probe.jsonl \
+  --require-expand
+```
+
+The source-level probe writes one JSON line immediately before `context->Output()` in CPU `Expand<T>::Compute()`:
+
+```json
+{"event":"ExpandPreAllocation","phase":"before_context_Output","input_shape":[1,2,1,128,16],"output_shape":[1,2,4,128,16],"element_size":4,"requested_bytes":65536}
+```
+
+## Scan a real quantized Llama 3.1 ONNX graph
+
+Use an already exported and QDQ-format quantized Llama 3.1 ONNX model:
+
+```bash
+python tools/qdq_expand_probe/scan_llama31_qdq_expand.py \
+  --model /path/to/llama3_1_qdq.onnx \
+  --out-json artifacts/qdq_expand_probe/llama31_qdq_expand_scan.json \
+  --out-md artifacts/qdq_expand_probe/llama31_qdq_expand_scan.md \
+  --require-case-a
+```
+
+The scanner is graph-only and uses `onnx.load(..., load_external_data=False)` so large external weight files are not loaded.
+
+Confirmed Case A means the same `Expand` node satisfies:
+
+```text
+DequantizeLinear -> Unsqueeze -> Expand -> Reshape -> ... -> MatMul
+```
+
+The downstream `MatMul` may be reached through layout-only nodes such as `Transpose`, because the key path often uses:
+
+```text
+Expand -> Reshape -> Transpose -> MatMul
+```
+
+## One-command synthetic package run
+
+```bash
+tools/qdq_expand_probe/run_package.sh
+```
+
+To also scan a real quantized Llama 3.1 ONNX:
+
+```bash
+LLAMA31_QDQ_ONNX=/path/to/llama3_1_qdq.onnx tools/qdq_expand_probe/run_package.sh
+```
diff --git a/tools/qdq_expand_probe/requirements.txt b/tools/qdq_expand_probe/requirements.txt
new file mode 100644
index 0000000000..25aaaf1766
--- /dev/null
+++ b/tools/qdq_expand_probe/requirements.txt
@@ -0,0 +1,4 @@
+numpy
+onnx
+onnxruntime
+psutil
diff --git a/tools/qdq_expand_probe/make_case_a_graph.py b/tools/qdq_expand_probe/make_case_a_graph.py
new file mode 100755
index 0000000000..9d573bc284
--- /dev/null
+++ b/tools/qdq_expand_probe/make_case_a_graph.py
@@ -0,0 +1,185 @@
+#!/usr/bin/env python3
+
+import argparse
+from pathlib import Path
+
+import numpy as np
+import onnx
+from onnx import TensorProto, helper
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Create a real ONNX Case A graph: DQ -> Unsqueeze -> Expand -> Reshape -> MatMul."
+    )
+    parser.add_argument("--out", required=True, help="Output ONNX path.")
+    parser.add_argument("--batch", type=int, default=1)
+    parser.add_argument("--num-key-value-heads", type=int, default=2)
+    parser.add_argument("--num-key-value-groups", type=int, default=4)
+    parser.add_argument("--seq", type=int, default=128)
+    parser.add_argument("--head-dim", type=int, default=16)
+    parser.add_argument("--opset", type=int, default=17)
+    parser.add_argument("--ir-version", type=int, default=10)
+    return parser.parse_args()
+
+
+def make_scalar_initializer(name: str, elem_type: int, value):
+    return helper.make_tensor(
+        name=name,
+        data_type=elem_type,
+        dims=[],
+        vals=[value],
+    )
+
+
+def make_model(
+    *,
+    batch: int,
+    num_key_value_heads: int,
+    num_key_value_groups: int,
+    seq: int,
+    head_dim: int,
+    opset: int,
+    ir_version: int,
+) -> onnx.ModelProto:
+    num_attention_heads = num_key_value_heads * num_key_value_groups
+
+    nodes = []
+    initializers = []
+
+    x = helper.make_tensor_value_info(
+        "x_uint8",
+        TensorProto.UINT8,
+        ["batch", num_key_value_heads, "seq", head_dim],
+    )
+
+    expand_shape = helper.make_tensor_value_info(
+        "expand_shape",
+        TensorProto.INT64,
+        [5],
+    )
+
+    reshape_shape = helper.make_tensor_value_info(
+        "reshape_shape",
+        TensorProto.INT64,
+        [4],
+    )
+
+    out = helper.make_tensor_value_info(
+        "out",
+        TensorProto.FLOAT,
+        ["batch", num_attention_heads, "seq", head_dim],
+    )
+
+    initializers.append(make_scalar_initializer("x_scale", TensorProto.FLOAT, 0.02))
+    initializers.append(make_scalar_initializer("x_zero_point", TensorProto.UINT8, 128))
+
+    initializers.append(
+        helper.make_tensor(
+            name="unsqueeze_axes",
+            data_type=TensorProto.INT64,
+            dims=[1],
+            vals=[2],
+        )
+    )
+
+    rng = np.random.default_rng(0)
+    matmul_weight = rng.standard_normal((head_dim, head_dim), dtype=np.float32)
+    initializers.append(
+        helper.make_tensor(
+            name="matmul_weight",
+            data_type=TensorProto.FLOAT,
+            dims=[head_dim, head_dim],
+            vals=matmul_weight.flatten().tolist(),
+        )
+    )
+
+    nodes.append(
+        helper.make_node(
+            "DequantizeLinear",
+            inputs=["x_uint8", "x_scale", "x_zero_point"],
+            outputs=["x_fp32"],
+            name="case_a_DequantizeLinear",
+        )
+    )
+
+    nodes.append(
+        helper.make_node(
+            "Unsqueeze",
+            inputs=["x_fp32", "unsqueeze_axes"],
+            outputs=["x_unsqueezed"],
+            name="case_a_Unsqueeze_repeat_kv_axis",
+        )
+    )
+
+    nodes.append(
+        helper.make_node(
+            "Expand",
+            inputs=["x_unsqueezed", "expand_shape"],
+            outputs=["x_expanded"],
+            name="case_a_Expand_repeat_kv_materialization",
+        )
+    )
+
+    nodes.append(
+        helper.make_node(
+            "Reshape",
+            inputs=["x_expanded", "reshape_shape"],
+            outputs=["x_repeated"],
+            name="case_a_Reshape_repeat_kv",
+        )
+    )
+
+    nodes.append(
+        helper.make_node(
+            "MatMul",
+            inputs=["x_repeated", "matmul_weight"],
+            outputs=["out"],
+            name="case_a_attention_like_MatMul",
+        )
+    )
+
+    graph = helper.make_graph(
+        nodes=nodes,
+        name="case_a_dq_unsqueeze_expand_reshape_matmul_graph",
+        inputs=[x, expand_shape, reshape_shape],
+        outputs=[out],
+        initializer=initializers,
+    )
+
+    model = helper.make_model(
+        graph,
+        opset_imports=[helper.make_operatorsetid("", opset)],
+        producer_name="qdq_expand_case_a_probe",
+    )
+    model.ir_version = ir_version
+
+    metadata = {
+        "probe": "case_a_dq_unsqueeze_expand_reshape_matmul",
+        "batch": str(batch),
+        "num_key_value_heads": str(num_key_value_heads),
+        "num_key_value_groups": str(num_key_value_groups),
+        "num_attention_heads": str(num_attention_heads),
+        "seq": str(seq),
+        "head_dim": str(head_dim),
+        "expand_shape": str([batch, num_key_value_heads, num_key_value_groups, seq, head_dim]),
+        "reshape_shape": str([batch, num_attention_heads, seq, head_dim]),
+    }
+    for key, value in metadata.items():
+        entry = model.metadata_props.add()
+        entry.key = key
+        entry.value = value
+
+    onnx.checker.check_model(model)
+    return model
+
+
+def main() -> None:
+    args = parse_args()
+    out_path = Path(args.out)
+    out_path.parent.mkdir(parents=True, exist_ok=True)
+
+    model = make_model(
+        batch=args.batch,
+        num_key_value_heads=args.num_key_value_heads,
+        num_key_value_groups=args.num_key_value_groups,
+        seq=args.seq,
+        head_dim=args.head_dim,
+        opset=args.opset,
+        ir_version=args.ir_version,
+    )
+    onnx.save(model, out_path)
+
+    print(f"wrote: {out_path}")
+    print("expected_case_a_path:")
+    print("  DequantizeLinear -> Unsqueeze -> Expand -> Reshape -> MatMul")
+    print("runtime_expand_shape:")
+    print(f"  {[args.batch, args.num_key_value_heads, args.num_key_value_groups, args.seq, args.head_dim]}")
+    print("runtime_reshape_shape:")
+    print(f"  {[args.batch, args.num_key_value_heads * args.num_key_value_groups, args.seq, args.head_dim]}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/tools/qdq_expand_probe/profile_onnx.py b/tools/qdq_expand_probe/profile_onnx.py
new file mode 100755
index 0000000000..e503774437
--- /dev/null
+++ b/tools/qdq_expand_probe/profile_onnx.py
@@ -0,0 +1,264 @@
+#!/usr/bin/env python3
+
+import argparse
+import json
+import os
+import shutil
+import sys
+import threading
+import time
+from pathlib import Path
+from typing import Any, Dict, List
+
+import numpy as np
+import onnxruntime as ort
+import psutil
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description="Run real ORT profiling and extract Expand kernel events.")
+    parser.add_argument("--model", required=True, help="ONNX model path.")
+    parser.add_argument("--outdir", required=True, help="Output artifact directory.")
+    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
+    parser.add_argument("--iters", type=int, default=5)
+    parser.add_argument("--require-expand", action="store_true")
+
+    parser.add_argument("--case-a", action="store_true", help="Generate feeds for the synthetic Case A graph.")
+    parser.add_argument("--batch", type=int, default=1)
+    parser.add_argument("--num-key-value-heads", type=int, default=2)
+    parser.add_argument("--num-key-value-groups", type=int, default=4)
+    parser.add_argument("--seq", type=int, default=128)
+    parser.add_argument("--head-dim", type=int, default=16)
+
+    parser.add_argument(
+        "--feed-npz",
+        default=None,
+        help="NPZ file containing real-model feeds. Use this for a real Llama 3.1 ONNX profile.",
+    )
+    parser.add_argument(
+        "--expand-probe-log",
+        default=None,
+        help="Path for ORT_EXPAND_PROBE_LOG. Requires the optional expand.cc source patch and local ORT build.",
+    )
+    return parser.parse_args()
+
+
+def json_default(value: Any):
+    if isinstance(value, np.ndarray):
+        return value.tolist()
+    if isinstance(value, np.generic):
+        return value.item()
+    return str(value)
+
+
+def make_case_a_feeds(args: argparse.Namespace) -> Dict[str, np.ndarray]:
+    h_q = args.num_key_value_heads * args.num_key_value_groups
+    rng = np.random.default_rng(0)
+
+    x_uint8 = rng.integers(
+        low=0,
+        high=255,
+        size=(args.batch, args.num_key_value_heads, args.seq, args.head_dim),
+        dtype=np.uint8,
+    )
+
+    expand_shape = np.array(
+        [
+            args.batch,
+            args.num_key_value_heads,
+            args.num_key_value_groups,
+            args.seq,
+            args.head_dim,
+        ],
+        dtype=np.int64,
+    )
+
+    reshape_shape = np.array(
+        [
+            args.batch,
+            h_q,
+            args.seq,
+            args.head_dim,
+        ],
+        dtype=np.int64,
+    )
+
+    return {
+        "x_uint8": x_uint8,
+        "expand_shape": expand_shape,
+        "reshape_shape": reshape_shape,
+    }
+
+
+def load_npz_feeds(path: str) -> Dict[str, np.ndarray]:
+    loaded = np.load(path, allow_pickle=False)
+    return {name: loaded[name] for name in loaded.files}
+
+
+def validate_feeds(sess: ort.InferenceSession, feeds: Dict[str, np.ndarray]) -> None:
+    required_inputs = {inp.name for inp in sess.get_inputs()}
+    provided_inputs = set(feeds)
+    missing = sorted(required_inputs - provided_inputs)
+    if missing:
+        raise SystemExit(f"missing required model inputs: {missing}")
+
+
+def extract_expand_events(profile_path: Path) -> List[Dict[str, Any]]:
+    with profile_path.open("r", encoding="utf-8") as f:
+        events = json.load(f)
+
+    expand_events: List[Dict[str, Any]] = []
+    for event in events:
+        args = event.get("args", {}) or {}
+        name = event.get("name", "")
+        op_name = args.get("op_name", "")
+        if op_name == "Expand" or "Expand" in name:
+            expand_events.append(
+                {
+                    "name": name,
+                    "cat": event.get("cat"),
+                    "ts": event.get("ts"),
+                    "dur_us": event.get("dur"),
+                    "op_name": op_name,
+                    "provider": args.get("provider"),
+                    "input_type_shape": args.get("input_type_shape"),
+                    "output_type_shape": args.get("output_type_shape"),
+                    "thread_scheduling_stats": args.get("thread_scheduling_stats"),
+                }
+            )
+
+    return expand_events
+
+
+def summarize_memory_trace(rows: List[Dict[str, int]]) -> Dict[str, int]:
+    if not rows:
+        return {}
+    base = rows[0]["rss_bytes"]
+    peak = max(row["rss_bytes"] for row in rows)
+    end = rows[-1]["rss_bytes"]
+    return {
+        "rss_base_bytes": base,
+        "rss_peak_bytes": peak,
+        "rss_end_bytes": end,
+        "rss_peak_delta_bytes": peak - base,
+        "samples": len(rows),
+    }
+
+
+def main() -> None:
+    args = parse_args()
+    model_path = Path(args.model)
+    outdir = Path(args.outdir)
+    outdir.mkdir(parents=True, exist_ok=True)
+
+    if args.expand_probe_log:
+        probe_path = Path(args.expand_probe_log)
+        probe_path.parent.mkdir(parents=True, exist_ok=True)
+        if probe_path.exists():
+            probe_path.unlink()
+        os.environ["ORT_EXPAND_PROBE_LOG"] = str(probe_path)
+
+    so = ort.SessionOptions()
+    so.enable_profiling = True
+    so.profile_file_prefix = str(outdir / "ort_profile")
+
+    sess = ort.InferenceSession(
+        str(model_path),
+        sess_options=so,
+        providers=args.providers,
+    )
+
+    if args.case_a:
+        feeds = make_case_a_feeds(args)
+    elif args.feed_npz:
+        feeds = load_npz_feeds(args.feed_npz)
+    else:
+        raise SystemExit("provide either --case-a or --feed-npz")
+
+    validate_feeds(sess, feeds)
+
+    feed_manifest = {
+        name: {
+            "shape": list(value.shape),
+            "dtype": str(value.dtype),
+            "nbytes": int(value.nbytes),
+        }
+        for name, value in feeds.items()
+    }
+
+    memory_rows: List[Dict[str, int]] = []
+    stop = False
+    proc = psutil.Process(os.getpid())
+
+    def sample_rss() -> None:
+        t0 = time.time()
+        while not stop:
+            memory_rows.append(
+                {
+                    "t_ms": int((time.time() - t0) * 1000),
+                    "rss_bytes": int(proc.memory_info().rss),
+                }
+            )
+            time.sleep(0.005)
+
+    sampler = threading.Thread(target=sample_rss)
+    sampler.start()
+
+    outputs_manifest = []
+    try:
+        for iteration in range(args.iters):
+            outputs = sess.run(None, feeds)
+            outputs_manifest.append(
+                {
+                    "iteration": iteration,
+                    "outputs": [
+                        {
+                            "shape": list(output.shape),
+                            "dtype": str(output.dtype),
+                            "nbytes": int(output.nbytes),
+                        }
+                        for output in outputs
+                    ],
+                }
+            )
+            print(json.dumps(outputs_manifest[-1], default=json_default))
+    finally:
+        stop = True
+        sampler.join()
+
+    profile_path = Path(sess.end_profiling())
+    copied_profile_path = outdir / profile_path.name
+    if profile_path.resolve() != copied_profile_path.resolve():
+        shutil.copy2(profile_path, copied_profile_path)
+        profile_path = copied_profile_path
+
+    expand_events = extract_expand_events(profile_path)
+
+    expand_events_path = outdir / "expand_events.json"
+    with expand_events_path.open("w", encoding="utf-8") as f:
+        json.dump(expand_events, f, indent=2, default=json_default)
+
+    memory_trace_path = outdir / "memory_trace.jsonl"
+    with memory_trace_path.open("w", encoding="utf-8") as f:
+        for row in memory_rows:
+            f.write(json.dumps(row) + "\n")
+
+    summary = {
+        "model": str(model_path),
+        "profile": str(profile_path),
+        "providers": args.providers,
+        "iters": args.iters,
+        "feeds": feed_manifest,
+        "outputs": outputs_manifest,
+        "expand_event_count": len(expand_events),
+        "expand_events_path": str(expand_events_path),
+        "memory_trace_path": str(memory_trace_path),
+        "memory_summary": summarize_memory_trace(memory_rows),
+        "expand_probe_log": args.expand_probe_log,
+    }
+
+    if args.expand_probe_log:
+        probe_path = Path(args.expand_probe_log)
+        summary["expand_probe_log_exists"] = probe_path.exists()
+        summary["expand_probe_log_size_bytes"] = probe_path.stat().st_size if probe_path.exists() else 0
+
+    summary_path = outdir / "run_summary.json"
+    with summary_path.open("w", encoding="utf-8") as f:
+        json.dump(summary, f, indent=2, default=json_default)
+
+    print(f"profile: {profile_path}")
+    print(f"expand_events: {expand_events_path}")
+    print(f"run_summary: {summary_path}")
+    print(f"expand_event_count: {len(expand_events)}")
+
+    if args.require_expand and not expand_events:
+        print("ERROR: --require-expand was set but no Expand profile event was found.", file=sys.stderr)
+        sys.exit(2)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/tools/qdq_expand_probe/scan_llama31_qdq_expand.py b/tools/qdq_expand_probe/scan_llama31_qdq_expand.py
new file mode 100755
index 0000000000..60590bfe1c
--- /dev/null
+++ b/tools/qdq_expand_probe/scan_llama31_qdq_expand.py
@@ -0,0 +1,367 @@
+#!/usr/bin/env python3
+
+import argparse
+import json
+import sys
+from collections import Counter, defaultdict, deque
+from pathlib import Path
+from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
+
+import onnx
+
+
+QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}
+LAYOUT_OPS = {
+    "Reshape",
+    "Transpose",
+    "Squeeze",
+    "Unsqueeze",
+    "Slice",
+    "Cast",
+    "Identity",
+}
+RISK_OPS = {"Expand"}
+DOWNSTREAM_ALLOWED = LAYOUT_OPS | QDQ_OPS | {"Gather"}
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description=(
+            "Scan a quantized Llama 3.1 ONNX graph for Case A: "
+            "DequantizeLinear -> Unsqueeze -> Expand -> Reshape -> ... -> MatMul."
+        )
+    )
+    parser.add_argument("--model", required=True)
+    parser.add_argument("--out-json", required=True)
+    parser.add_argument("--out-md", required=True)
+    parser.add_argument("--max-downstream-depth", type=int, default=8)
+    parser.add_argument("--infer-shapes", action="store_true")
+    parser.add_argument("--require-case-a", action="store_true")
+    return parser.parse_args()
+
+
+def load_model(path: Path, infer_shapes: bool) -> onnx.ModelProto:
+    model = onnx.load(path, load_external_data=False)
+    if infer_shapes:
+        model = onnx.shape_inference.infer_shapes(model)
+    return model
+
+
+def build_maps(model: onnx.ModelProto):
+    producer = {}
+    consumers = defaultdict(list)
+
+    for node in model.graph.node:
+        for output_name in node.output:
+            if output_name:
+                producer[output_name] = node
+        for input_name in node.input:
+            if input_name:
+                consumers[input_name].append(node)
+
+    return producer, consumers
+
+
+def graph_outputs(model: onnx.ModelProto) -> set:
+    return {output.name for output in model.graph.output}
+
+
+def node_id(node: onnx.NodeProto) -> str:
+    name = node.name if node.name else "<unnamed>"
+    domain = node.domain if node.domain else "ai.onnx"
+    return f"{node.op_type}::{name}::{domain}"
+
+
+def node_to_dict(node: onnx.NodeProto) -> Dict[str, Any]:
+    return {
+        "name": node.name,
+        "op_type": node.op_type,
+        "domain": node.domain,
+        "inputs": list(node.input),
+        "outputs": list(node.output),
+    }
+
+
+def path_to_dict(path: Sequence[onnx.NodeProto]) -> List[Dict[str, Any]]:
+    return [node_to_dict(node) for node in path]
+
+
+def path_to_string(path: Sequence[onnx.NodeProto]) -> str:
+    return " -> ".join(node_id(node) for node in path)
+
+
+def first_data_producer(
+    node: onnx.NodeProto,
+    producer: Dict[str, onnx.NodeProto],
+    input_index: int = 0,
+) -> Optional[onnx.NodeProto]:
+    if len(node.input) <= input_index:
+        return None
+    return producer.get(node.input[input_index])
+
+
+def consumers_of_node(
+    node: onnx.NodeProto,
+    consumers: Dict[str, List[onnx.NodeProto]],
+    output_index: int = 0,
+) -> List[onnx.NodeProto]:
+    if len(node.output) <= output_index:
+        return []
+    return consumers.get(node.output[output_index], [])
+
+
+def all_consumers_of_node(
+    node: onnx.NodeProto,
+    consumers: Dict[str, List[onnx.NodeProto]],
+) -> List[onnx.NodeProto]:
+    out = []
+    for output_name in node.output:
+        out.extend(consumers.get(output_name, []))
+    return out
+
+
+def get_shape_map(model: onnx.ModelProto) -> Dict[str, List[Any]]:
+    shape_map: Dict[str, List[Any]] = {}
+    value_infos = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
+
+    for value_info in value_infos:
+        tensor_type = value_info.type.tensor_type
+        if not tensor_type.HasField("shape"):
+            continue
+
+        dims: List[Any] = []
+        for dim in tensor_type.shape.dim:
+            if dim.HasField("dim_value"):
+                dims.append(dim.dim_value)
+            elif dim.HasField("dim_param"):
+                dims.append(dim.dim_param)
+            else:
+                dims.append("?")
+        shape_map[value_info.name] = dims
+
+    return shape_map
+
+
+def output_shape(node: onnx.NodeProto, shape_map: Dict[str, List[Any]], output_index: int = 0) -> Optional[List[Any]]:
+    if len(node.output) <= output_index:
+        return None
+    return shape_map.get(node.output[output_index])
+
+
+def input_shape(node: onnx.NodeProto, shape_map: Dict[str, List[Any]], input_index: int = 0) -> Optional[List[Any]]:
+    if len(node.input) <= input_index:
+        return None
+    return shape_map.get(node.input[input_index])
+
+
+def find_first_consumer_by_op(
+    node: onnx.NodeProto,
+    consumers: Dict[str, List[onnx.NodeProto]],
+    op_type: str,
+) -> Optional[onnx.NodeProto]:
+    for consumer in all_consumers_of_node(node, consumers):
+        if consumer.op_type == op_type:
+            return consumer
+    return None
+
+
+def find_downstream_path_to_matmul(
+    start: onnx.NodeProto,
+    consumers: Dict[str, List[onnx.NodeProto]],
+    max_depth: int,
+) -> Optional[List[onnx.NodeProto]]:
+    queue = deque([[start]])
+    visited = {node_id(start)}
+
+    while queue:
+        path = queue.popleft()
+        current = path[-1]
+
+        if len(path) - 1 >= max_depth:
+            continue
+
+        for nxt in all_consumers_of_node(current, consumers):
+            nxt_id = node_id(nxt)
+            if nxt_id in visited:
+                continue
+
+            if nxt.op_type == "MatMul":
+                return path + [nxt]
+
+            if nxt.op_type in DOWNSTREAM_ALLOWED:
+                visited.add(nxt_id)
+                queue.append(path + [nxt])
+
+    return None
+
+
+def detect_exact_case_a(
+    expand: onnx.NodeProto,
+    producer: Dict[str, onnx.NodeProto],
+    consumers: Dict[str, List[onnx.NodeProto]],
+    max_downstream_depth: int,
+) -> Optional[List[onnx.NodeProto]]:
+    unsqueeze = first_data_producer(expand, producer, 0)
+    if unsqueeze is None or unsqueeze.op_type != "Unsqueeze":
+        return None
+
+    dq = first_data_producer(unsqueeze, producer, 0)
+    if dq is None or dq.op_type != "DequantizeLinear":
+        return None
+
+    reshape = find_first_consumer_by_op(expand, consumers, "Reshape")
+    if reshape is None:
+        return None
+
+    downstream = find_downstream_path_to_matmul(reshape, consumers, max_downstream_depth)
+    if downstream is None:
+        return [dq, unsqueeze, expand, reshape]
+
+    return [dq, unsqueeze, expand] + downstream
+
+
+def detect_propagated_pre_expand_qdq(
+    expand: onnx.NodeProto,
+    producer: Dict[str, onnx.NodeProto],
+    consumers: Dict[str, List[onnx.NodeProto]],
+    max_downstream_depth: int,
+) -> Optional[List[onnx.NodeProto]]:
+    dq = first_data_producer(expand, producer, 0)
+    if dq is None or dq.op_type != "DequantizeLinear":
+        return None
+
+    q = first_data_producer(dq, producer, 0)
+    if q is None or q.op_type != "QuantizeLinear":
+        return None
+
+    unsqueeze = first_data_producer(q, producer, 0)
+    if unsqueeze is None or unsqueeze.op_type != "Unsqueeze":
+        return None
+
+    reshape = find_first_consumer_by_op(expand, consumers, "Reshape")
+    if reshape is None:
+        return None
+
+    downstream = find_downstream_path_to_matmul(reshape, consumers, max_downstream_depth)
+    if downstream is None:
+        return [unsqueeze, q, dq, expand, reshape]
+
+    return [unsqueeze, q, dq, expand] + downstream
+
+
+def detect_secondary_post_expand_qdq(
+    expand: onnx.NodeProto,
+    consumers: Dict[str, List[onnx.NodeProto]],
+    max_downstream_depth: int,
+) -> List[List[onnx.NodeProto]]:
+    findings: List[List[onnx.NodeProto]] = []
+    queue = deque([[expand]])
+    visited = {node_id(expand)}
+
+    while queue:
+        path = queue.popleft()
+        current = path[-1]
+        if len(path) - 1 >= max_downstream_depth:
+            continue
+
+        for nxt in all_consumers_of_node(current, consumers):
+            nxt_id = node_id(nxt)
+            if nxt_id in visited:
+                continue
+
+            candidate = path + [nxt]
+            if nxt.op_type in QDQ_OPS:
+                findings.append(candidate)
+
+            if nxt.op_type in DOWNSTREAM_ALLOWED:
+                visited.add(nxt_id)
+                queue.append(candidate)
+
+    return findings
+
+
+def expand_context(
+    expand: onnx.NodeProto,
+    producer: Dict[str, onnx.NodeProto],
+    consumers: Dict[str, List[onnx.NodeProto]],
+    shape_map: Dict[str, List[Any]],
+) -> Dict[str, Any]:
+    data_producer = first_data_producer(expand, producer, 0)
+    output_consumers = all_consumers_of_node(expand, consumers)
+
+    return {
+        "expand": node_to_dict(expand),
+        "expand_input_shape": input_shape(expand, shape_map, 0),
+        "expand_output_shape": output_shape(expand, shape_map, 0),
+        "data_input_producer": node_to_dict(data_producer) if data_producer is not None else None,
+        "output_consumers": [node_to_dict(node) for node in output_consumers],
+    }
+
+
+def scan_model(model: onnx.ModelProto, max_downstream_depth: int) -> Dict[str, Any]:
+    producer, consumers = build_maps(model)
+    shape_map = get_shape_map(model)
+    op_histogram = Counter(node.op_type for node in model.graph.node)
+
+    exact_case_a = []
+    propagated_pre_expand = []
+    secondary_post_expand = []
+    expand_nodes = []
+
+    for node in model.graph.node:
+        if node.op_type != "Expand":
+            continue
+
+        expand_nodes.append(expand_context(node, producer, consumers, shape_map))
+
+        exact_path = detect_exact_case_a(node, producer, consumers, max_downstream_depth)
+        if exact_path is not None:
+            exact_case_a.append(
+                {
+                    "kind": "exact_case_a",
+                    "path": path_to_dict(exact_path),
+                    "path_string": path_to_string(exact_path),
+                    "expand_output_shape": output_shape(node, shape_map, 0),
+                }
+            )
+
+        propagated_path = detect_propagated_pre_expand_qdq(node, producer, consumers, max_downstream_depth)
+        if propagated_path is not None:
+            propagated_pre_expand.append(
+                {
+                    "kind": "propagated_pre_expand_qdq",
+                    "path": path_to_dict(propagated_path),
+                    "path_string": path_to_string(propagated_path),
+                    "expand_output_shape": output_shape(node, shape_map, 0),
+                }
+            )
+
+        for secondary_path in detect_secondary_post_expand_qdq(node, consumers, max_downstream_depth):
+            secondary_post_expand.append(
+                {
+                    "kind": "secondary_post_expand_qdq",
+                    "path": path_to_dict(secondary_path),
+                    "path_string": path_to_string(secondary_path),
+                    "expand_output_shape": output_shape(node, shape_map, 0),
+                }
+            )
+
+    return {
+        "summary": {
+            "node_count": len(model.graph.node),
+            "expand_count": op_histogram.get("Expand", 0),
+            "quantize_linear_count": op_histogram.get("QuantizeLinear", 0),
+            "dequantize_linear_count": op_histogram.get("DequantizeLinear", 0),
+            "unsqueeze_count": op_histogram.get("Unsqueeze", 0),
+            "reshape_count": op_histogram.get("Reshape", 0),
+            "matmul_count": op_histogram.get("MatMul", 0),
+            "exact_case_a_count": len(exact_case_a),
+            "propagated_pre_expand_qdq_count": len(propagated_pre_expand),
+            "secondary_post_expand_qdq_count": len(secondary_post_expand),
+        },
+        "exact_case_a": exact_case_a,
+        "propagated_pre_expand_qdq": propagated_pre_expand,
+        "secondary_post_expand_qdq": secondary_post_expand,
+        "expand_nodes": expand_nodes,
+        "op_histogram": dict(op_histogram),
+    }
+
+
+def write_markdown(report: Dict[str, Any], out_path: Path) -> None:
+    lines = []
+    summary = report["summary"]
+
+    lines.append("# Llama 3.1 QDQ / Expand Case A scan\n")
+    lines.append("## Summary\n")
+    for key, value in summary.items():
+        lines.append(f"- `{key}`: `{value}`")
+
+    def write_findings(title: str, key: str) -> None:
+        lines.append(f"\n## {title}\n")
+        findings = report[key]
+        if not findings:
+            lines.append("No findings.\n")
+            return
+        for i, finding in enumerate(findings):
+            lines.append(f"### Finding {i}\n")
+            lines.append("```text")
+            lines.append(finding["path_string"])
+            lines.append("```")
+            lines.append(f"- `expand_output_shape`: `{finding.get('expand_output_shape')}`")
+
+    write_findings("Exact Case A: DequantizeLinear -> Unsqueeze -> Expand -> Reshape -> ... -> MatMul", "exact_case_a")
+    write_findings("Propagated pre-Expand QDQ: Unsqueeze -> QuantizeLinear -> DequantizeLinear -> Expand", "propagated_pre_expand_qdq")
+    write_findings("Secondary post-Expand QDQ", "secondary_post_expand_qdq")
+
+    out_path.parent.mkdir(parents=True, exist_ok=True)
+    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
+
+
+def main() -> None:
+    args = parse_args()
+    model_path = Path(args.model)
+    out_json = Path(args.out_json)
+    out_md = Path(args.out_md)
+
+    model = load_model(model_path, args.infer_shapes)
+    report = scan_model(model, args.max_downstream_depth)
+    report["model"] = str(model_path)
+    report["infer_shapes"] = args.infer_shapes
+
+    out_json.parent.mkdir(parents=True, exist_ok=True)
+    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
+    write_markdown(report, out_md)
+
+    print(f"scan_json: {out_json}")
+    print(f"scan_md: {out_md}")
+    print(json.dumps(report["summary"], indent=2))
+
+    if args.require_case_a and report["summary"]["exact_case_a_count"] == 0:
+        print("ERROR: --require-case-a was set but exact Case A was not found.", file=sys.stderr)
+        sys.exit(2)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/tools/qdq_expand_probe/run_package.sh b/tools/qdq_expand_probe/run_package.sh
new file mode 100755
index 0000000000..0db453366f
--- /dev/null
+++ b/tools/qdq_expand_probe/run_package.sh
@@ -0,0 +1,72 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
+ARTIFACTS="${ROOT}/artifacts/qdq_expand_probe"
+
+mkdir -p "${ARTIFACTS}"
+
+CASE_A_MODEL="${ARTIFACTS}/case_a_dq_unsqueeze_expand_reshape_matmul.onnx"
+CASE_A_PROFILE_DIR="${ARTIFACTS}/case_a_profile"
+
+echo "[1/3] Generate synthetic Case A graph"
+python "${ROOT}/tools/qdq_expand_probe/make_case_a_graph.py" \
+  --out "${CASE_A_MODEL}" \
+  --batch "${BATCH:-1}" \
+  --num-key-value-heads "${H_KV:-2}" \
+  --num-key-value-groups "${N_REP:-4}" \
+  --seq "${SEQ:-128}" \
+  --head-dim "${HEAD_DIM:-16}"
+
+echo "[2/3] Run real ORT profile on synthetic Case A graph"
+PROFILE_ARGS=(
+  python "${ROOT}/tools/qdq_expand_probe/profile_onnx.py"
+  --model "${CASE_A_MODEL}"
+  --case-a
+  --outdir "${CASE_A_PROFILE_DIR}"
+  --batch "${BATCH:-1}"
+  --num-key-value-heads "${H_KV:-2}"
+  --num-key-value-groups "${N_REP:-4}"
+  --seq "${SEQ:-128}"
+  --head-dim "${HEAD_DIM:-16}"
+  --iters "${ITERS:-5}"
+  --require-expand
+)
+
+if [[ -n "${EXPAND_PROBE_LOG:-}" ]]; then
+  PROFILE_ARGS+=(--expand-probe-log "${EXPAND_PROBE_LOG}")
+fi
+
+"${PROFILE_ARGS[@]}"
+
+if [[ -n "${LLAMA31_QDQ_ONNX:-}" ]]; then
+  echo "[3/3] Scan real quantized Llama 3.1 ONNX graph"
+  SCAN_ARGS=(
+    python "${ROOT}/tools/qdq_expand_probe/scan_llama31_qdq_expand.py"
+    --model "${LLAMA31_QDQ_ONNX}"
+    --out-json "${ARTIFACTS}/llama31_qdq_expand_scan.json"
+    --out-md "${ARTIFACTS}/llama31_qdq_expand_scan.md"
+  )
+
+  if [[ "${REQUIRE_CASE_A:-0}" == "1" ]]; then
+    SCAN_ARGS+=(--require-case-a)
+  fi
+
+  if [[ "${INFER_SHAPES:-0}" == "1" ]]; then
+    SCAN_ARGS+=(--infer-shapes)
+  fi
+
+  "${SCAN_ARGS[@]}"
+else
+  echo "[3/3] Skip real Llama scan because LLAMA31_QDQ_ONNX is unset"
+fi
+
+echo
+echo "Artifacts:"
+find "${ARTIFACTS}" -maxdepth 3 -type f | sort
````

적용 후 바로 실행은 이렇게 하면 된다.

```bash
git apply qdq_expand_probe.patch

python -m venv .venv-qdx
source .venv-qdx/bin/activate
pip install -r tools/qdq_expand_probe/requirements.txt

tools/qdq_expand_probe/run_package.sh
```

real Llama 3.1 QDQ ONNX까지 한 번에 검사하려면:

```bash
LLAMA31_QDQ_ONNX=/path/to/llama3_1_qdq.onnx \
REQUIRE_CASE_A=1 \
tools/qdq_expand_probe/run_package.sh
```

source-level `Expand` pre-allocation 로그까지 필요하면 ORT를 로컬 빌드한 뒤:

```bash
./build.sh --config RelWithDebInfo --build_wheel --parallel --skip_tests
pip install --force-reinstall build/Linux/RelWithDebInfo/dist/*.whl

EXPAND_PROBE_LOG=artifacts/qdq_expand_probe/expand_probe.jsonl \
tools/qdq_expand_probe/run_package.sh
```

성공 판정은 이 3개 파일로 한다.

```text
artifacts/qdq_expand_probe/case_a_profile/expand_events.json
artifacts/qdq_expand_probe/case_a_profile/run_summary.json
artifacts/qdq_expand_probe/llama31_qdq_expand_scan.json
```

`llama31_qdq_expand_scan.json`에서 `exact_case_a_count > 0`이면, 실제 quantized Llama 3.1 그래프에서 원하는 edge case가 잡힌 것이다.
