#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
import torch
import triton

from ....triton.chunkwise.xl_chunk_siging import (
    mlstm_siging_chunkwise__parallel_bw_dQ_kernel,
)
from ....triton.kernel_param_heuristics import get_head_dim_block_size
from ....utils.kernels import is_power_of_2
from ...utils import torch2triton_dtype


def mlstm_siging_chunkwise__parallel_bw_dQ(
    ## Forward arguments
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecA: torch.Tensor,  # (B, NH, NC, L)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    ## Backward arguments
    matCstate_all: torch.Tensor,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecNstate_all: torch.Tensor,  # (B, NH, (NC+1) * DHQK)
    vecN_out: torch.Tensor,  # (B, NH, S) # vecN_combine
    matDeltaH_out: torch.Tensor,  # (B, NH, S, DHHV)
    matDeltaC_states: torch.Tensor,  # (B, NH, (NC+1) * DHQK, DHHV)
    ## Other arguments
    qk_scale: float = None,
    normalize: bool = True,
    chunk_size: int = 64,
    siz_b_LQ: int = 32,
    siz_b_LKV: int = 32,
    siz_b_DHQK: int | None = None,
    siz_b_DHHV: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:  # matDeltaQ (B, NH, S, DHQK)
    """This function defines the grid and block sizes for the kernel launch and calls the kernel.
    chunk parallel size:        siz_b_LQ
    chunk loop size:            siz_b_LKV
    head dim parallel size:     siz_b_DHQK
    head dim loop size:         siz_b_DHHV
    """
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[-1]

    assert (
        S % chunk_size == 0
    ), f"Sequence length {S} must be divisible by chunk size {chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = (
        get_head_dim_block_size(head_dim=DHQK, min_block_size=128)
        if siz_b_DHQK is None
        else siz_b_DHQK
    )
    siz_b_DHHV = (
        get_head_dim_block_size(head_dim=DHHV, min_block_size=64)
        if siz_b_DHHV is None
        else siz_b_DHHV
    )

    assert siz_b_LQ <= L, "siz_b_LQ must be less than or equal to chunk size L"
    assert siz_b_LKV <= L, "siz_b_LKV must be less than or equal to chunk size L"
    assert siz_b_LKV <= siz_b_LQ, "siz_b_LKV must be less than or equal to siz_b_LQ"
    assert siz_b_LQ % siz_b_LKV == 0, "siz_b_LQ must be divisible by siz_b_LKV"

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_LQ = triton.cdiv(L, siz_b_LQ)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if (siz_b_DHQK >= 64 or siz_b_DHHV >= 64) else 2

    matDeltaQ = torch.empty(B, NH, S, DHQK, device=matQ.device, dtype=output_dtype)

    grid = (num_b_DHQK, num_b_LQ, NC * B * NH)

    mlstm_siging_chunkwise__parallel_bw_dQ_kernel[grid](
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matCstate_all=matCstate_all,
        vecNstate_all=vecNstate_all,
        vecN_out=vecN_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        matDeltaQ=matDeltaQ,
        qk_scale=qk_scale,
        str_matQK_B_NH=matQ.stride(1),
        str_matQK_S=matQ.stride(2),
        str_matQK_DHQK=matQ.stride(3),
        str_matHV_B_NH=matV.stride(1),
        str_matHV_S=matV.stride(2),
        str_matHV_DHHV=matV.stride(3),
        str_vecABI_B_NH=vecB.stride(1),
        str_vecABI_NC=vecB.stride(2),
        str_matCstate_B_NH=matCstate_all.stride(1),
        str_matCstate_NCDHQK=matCstate_all.stride(2),
        str_matCstate_DHHV=matCstate_all.stride(3),
        str_vecNstate_B_NH=vecNstate_all.stride(1),
        str_vecN_B_NH=vecN_out.stride(1),
        str_vecN_S=vecN_out.stride(2),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_LQ=siz_b_LQ,
        siz_b_LKV=siz_b_LKV,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        NORMALIZE=normalize,
        DTYPE=torch2triton_dtype(matQ.dtype),
        OUTPUT_DTYPE=torch2triton_dtype(output_dtype),
        EPS=eps,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matDeltaQ
