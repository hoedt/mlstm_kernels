from pathlib import Path

from mlstm_kernels.benchmark_utils.benchmarks.inference_kernel_benchmarks import (
    create_inference_kernel_benchmark,
)
from mlstm_kernels.benchmark_utils.param_handling import BenchmarkConfig
from mlstm_kernels.benchmark_utils.run_benchmark import run_and_record_benchmarks
from mlstm_kernels.benchmark_utils.utils import setup_output_folder

from dacite import from_dict
from omegaconf import OmegaConf


def _head_dim_benchmark(output_folder: Path, half_qkdim=False, batch_size: int = 1):
    ### head dimension benchmark 7B
    head_dims_v = [64, 128, 256, 512, 1024, 2048]
    embedding_dim = 4096
    num_heads = [embedding_dim // head_dim for head_dim in head_dims_v]
    if half_qkdim:
        head_dims_qk = [head_dim // 2 for head_dim in head_dims_v]
        bench_name = "head_dim_half_qk_7B"
    else:
        head_dims_qk = head_dims_v
        bench_name = "head_dim_7B"

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  num_heads: {num_heads}
  head_dim_qk: {head_dims_qk}
  head_dim_v: {head_dims_v}
fixed_params:
  batch_size: {batch_size}
  rep: 2500
  warmup: 500

x_axis_param: "head_dim_v"

kernel_specs:
  - kernel_name: "step_fused_triton"
    dtype: bfloat16
  - kernel_name: "step_fused_triton"
    dtype: float32
  - kernel_name: "step_triton"
    dtype: bfloat16
  - kernel_name: "step_triton"
    dtype: float32
  - kernel_name: "step_torch_autograd"
    dtype: bfloat16
    use_torch_compile: True
  - kernel_name: "step_torch_autograd"
    dtype: float32
    use_torch_compile: True
  - kernel_name: "step_torch_autograd"
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "step_torch_autograd"
    dtype: float32
    use_torch_compile: False

benchmark_name: {bench_name}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_inference_kernel_benchmark, output_folder)


def _batch_size_benchmark(
    output_folder: Path,
    num_heads: int = 8,
    head_dim_qk: int = 256,
    head_dim_v: int = 512,
):
    bench_name_params = f"nh_{num_heads}_hdqk_{head_dim_qk}_hdv_{head_dim_v}"

    ### batch size benchmark 7B
    cfg_yaml = f"""
vary_type: sequence
vary_params:
  batch_size: [1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048]
fixed_params:
  num_heads: {num_heads}
  head_dim_qk: {head_dim_qk}
  head_dim_v: {head_dim_v}
  rep: 2500
  warmup: 500

x_axis_param: "batch_size"

kernel_specs:
  - kernel_name: "step_fused_triton"
    dtype: bfloat16
  - kernel_name: "step_fused_triton"
    dtype: float32
  - kernel_name: "step_triton"
    dtype: bfloat16
  - kernel_name: "step_triton"
    dtype: float32
  - kernel_name: "step_torch_autograd"
    dtype: bfloat16
    use_torch_compile: True
  - kernel_name: "step_torch_autograd"
    dtype: float32
    use_torch_compile: True
  - kernel_name: "step_torch_autograd"
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "step_torch_autograd"
    dtype: float32
    use_torch_compile: False


benchmark_name: "batch_size_7B--{bench_name_params}"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_inference_kernel_benchmark, output_folder)


def run_multiple_benchmarks(output_dir: str = "./outputs_kernel_benchmarks"):
    output_folder = setup_output_folder(output_dir)

    _batch_size_benchmark(output_folder, num_heads=8, head_dim_qk=256, head_dim_v=512)

    _head_dim_benchmark(output_folder, half_qkdim=False, batch_size=8)
    _head_dim_benchmark(output_folder, half_qkdim=True, batch_size=8)


if __name__ == "__main__":
    run_multiple_benchmarks()
