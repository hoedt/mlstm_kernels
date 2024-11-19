import logging

from mlstm_kernels.torch.chunkwise.native import mlstm_chunkwise__native_custbw
from mlstm_kernels.torch.parallel.native_stablef import mlstm_parallel__native_stablef_custbw

import pytest
import torch

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-torch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_native_chunkwise_torch_vs_native_parrallel_stablef_fp32(
    test_session_folder, test_output_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_custbw,
        target_fn=mlstm_chunkwise__native_custbw,
        baseline_name="native_parallel_stablef_custbw",
        target_name="native_chunkwise_custbw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1e-4,
        rtol_fw=1e-3,
        atol_fwbw=1e-3,
        rtol_fwbw=1e-2,
        vmax=1e-4,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        # save_output_tensors_dir=str(test_output_folder),
    )
