#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging

import pytest
import torch

from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk
from mlstm_kernels.torch.parallel.native_stablef import (
    mlstm_parallel__native_stablef_custbw,
)

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-triton_xl_chunk"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_triton_chunkwise_xl_chunk_vs_native_parallel_stablef_fp32(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_custbw,
        target_fn=mlstm_chunkwise__xl_chunk,
        baseline_name="native_parallel_stablef_custbw",
        target_name="triton_chunkwise_xl_chunk",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1e-2,
        rtol_fw=1.3e-6,
        atol_fwbw=42e-1,  # we need to increase this tolerance for vecF.grad (max diff val 0.267...)
        rtol_fwbw=0.5,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        save_output_tensors_dir=str(test_output_folder),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
def test_state_passing(mlstm_state_passing_test, state_passing_qkvif):
    num_chunks = state_passing_qkvif[0].shape[2] // 64  # <- chunk size = 64

    mlstm_state_passing_test(
        kernel_fn=mlstm_chunkwise__xl_chunk,
        q=state_passing_qkvif[0],
        k=state_passing_qkvif[1],
        v=state_passing_qkvif[2],
        igate_preact=state_passing_qkvif[3],
        fgate_preact=state_passing_qkvif[4],
        num_chunks=num_chunks,
        rtol=2e-3,
        atol=2e-3,
        device="cuda",
    )
