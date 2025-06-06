#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import triton

from ....triton.chunkwise.limit_chunk import mlstm_chunkwise__recurrent_fw_C_kernel
from ....triton.kernel_param_heuristics import get_head_dim_block_size
from ....utils.kernels import is_power_of_2
from ...utils import torch2triton_dtype


def mlstm_chunkwise__recurrent_fw_C(
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    matC_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states: torch.Tensor = None,  # (B, NH, (NC + 1)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaMinter_initial: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1))
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    NC = NUM_CHUNKS
    L = CHUNK_SIZE

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    siz_b_DHQK = get_head_dim_block_size(head_dim=DHQK, min_block_size=64)
    siz_b_DHHV = get_head_dim_block_size(head_dim=DHHV, min_block_size=64)

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    USE_INITIAL_STATE = matC_initial is not None
    if USE_INITIAL_STATE:
        assert vecN_initial is not None and scaMinter_initial is not None
        str_matCinitial_B_NH = matC_initial.stride(1)
        str_matCinitial_DHQK = matC_initial.stride(2)
        str_matCinitial_DHHV = matC_initial.stride(3)
        str_vecNinitial_B_NH = vecN_initial.stride(1)
        str_vecNinitial_DHQK = vecN_initial.stride(2)
        str_scaMinterinitial_B_NH = scaMinter_initial.stride(1)
    else:
        str_matCinitial_B_NH = 0
        str_matCinitial_DHQK = 0
        str_matCinitial_DHHV = 0
        str_vecNinitial_B_NH = 0
        str_vecNinitial_DHQK = 0
        str_scaMinterinitial_B_NH = 0

    matC_states = (
        torch.empty(
            B, NH, (NC + 1) * DHQK, DHHV, device=matK.device, dtype=torch.float32
        )
        if matC_states is None
        else matC_states
    )
    vecN_states = (
        torch.empty(B, NH, (NC + 1) * DHQK, device=matK.device, dtype=torch.float32)
        if vecN_states is None
        else vecN_states
    )
    scaMinter_states = (
        torch.empty(B, NH, (NC + 1), device=matK.device, dtype=torch.float32)
        if scaMinter_states is None
        else scaMinter_states
    )

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    mlstm_chunkwise__recurrent_fw_C_kernel[grid](
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        matC_states=matC_states,
        vecN_states=vecN_states,
        scaMinter_states=scaMinter_states,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaMinter_initial,
        str_matK_B_NH=matK.stride(1),
        str_matK_S=matK.stride(2),
        str_matK_DHQK=matK.stride(3),
        str_matV_B_NH=matV.stride(1),
        str_matV_S=matV.stride(2),
        str_matV_DHHV=matV.stride(3),
        str_vecBI_B_NH=vecB.stride(1),
        str_vecBI_NC=vecB.stride(2),
        str_vecBI_L=vecB.stride(3),
        str_matCstates_B_NH=matC_states.stride(1),
        str_matCstates_NCDHQK=matC_states.stride(2),
        str_matCstates_DHHV=matC_states.stride(3),
        str_vecNstates_B_NH=vecN_states.stride(1),
        str_vecNstates_NCDHQK=vecN_states.stride(2),
        str_scaMinterstates_B_NH=scaMinter_states.stride(1),
        str_scaMinterstates_NC=scaMinter_states.stride(2),
        str_matCinitial_B_NH=str_matCinitial_B_NH,
        str_matCinitial_DHQK=str_matCinitial_DHQK,
        str_matCinitial_DHHV=str_matCinitial_DHHV,
        str_vecNinitial_B_NH=str_vecNinitial_B_NH,
        str_vecNinitial_DHQK=str_vecNinitial_DHQK,
        str_scaMinterinitial_B_NH=str_scaMinterinitial_B_NH,
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        DTYPE=torch2triton_dtype(matK.dtype),
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matC_states, vecN_states, scaMinter_states
