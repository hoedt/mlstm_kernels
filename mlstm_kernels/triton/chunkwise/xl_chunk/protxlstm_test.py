import math
from typing import Optional

import torch


def chunkwise_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,  # B, NH, S, DH
    values: torch.Tensor,  # B, NH, S, DH
    igate_preact: torch.Tensor,  # B, NH, S
    fgate_preact: torch.Tensor,  # B, NH, S
    initial_C: Optional[torch.Tensor] = None,  # B, NH, DH, DH
    initial_n: Optional[torch.Tensor] = None,  # B, NH, DH, 1
    initial_m: Optional[torch.Tensor] = None,  # B, NH, 1, 1
    chunk_size: int = 64,  # optimize this
    return_last_state: bool = False,
    eps: float = 1e-6,
):
    B, NH, S, DH = queries.shape
    NS, CS = S // chunk_size, chunk_size
    _dtype, _device = queries.dtype, queries.device

    # form chunks
    q = queries.view(B, NH, NS, CS, DH)
    k = keys.view(B, NH, NS, CS, DH) / math.sqrt(DH)
    v = values.view(B, NH, NS, CS, DH)

    # forget gates
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact).view(B, NH, NS, CS)
    log_fgates_acc = log_fgates.cumsum(dim=3)
    igate_preact = igate_preact.view(B, NH, NS, CS)

    log_fgates_rep = log_fgates[:, :, :, :, None].repeat(1, 1, 1, 1, CS)
    log_fg_matrix = torch.tril(log_fgates_rep, diagonal=-1)
    log_prod_fg_matrix = torch.cumsum(log_fg_matrix, dim=3)
    
    loggates = (igate_preact + log_prod_fg_matrix[:, :, :, -1]).unsqueeze(-1)
    m_loc, _ = torch.max(loggates, dim=3, keepdim=True)
    loggates = loggates - m_loc

    kv = k.transpose(-1, -2) @ (v * (loggates).exp())
    ksum = (k * (loggates).exp()).sum(dim=-2)
    C = torch.zeros((B, NH, NS + 1, DH, DH), device=kv.device, dtype=kv.dtype)
    n = torch.zeros((B, NH, NS + 1, DH, 1), device=kv.device, dtype=kv.dtype)
    if initial_C is not None:
        C[:, :, 0] = initial_C
    if initial_n is not None:
        n[:, :, 0] = initial_n

    m = torch.zeros((B, NH, NS + 1, 1, 1), device=kv.device, dtype=kv.dtype)
    if initial_m is not None:
        m[:, :, 0] = initial_m

    for i in range(1, NS + 1):
        m[:, :, i] = torch.maximum(
            log_fgates_acc[:, :, i - 1, -1, None, None] + m[:, :, i - 1],
            m_loc[:, :, i - 1],
        )
        C[:, :, i] = (
            C[:, :, i - 1].clone()
            * (
                log_fgates_acc[:, :, i - 1, -1, None, None]
                + m[:, :, i - 1]
                - m[:, :, i]
            ).exp()
            + kv[:, :, i - 1] * (m_loc[:, :, i - 1] - m[:, :, i]).exp()
        )
        n[:, :, i] = (
            n[:, :, i - 1].clone()
            * (
                log_fgates_acc[:, :, i - 1, None, -1:]
                + m[:, :, i - 1]
                - m[:, :, i]
            ).exp()
            + ksum[:, :, i - 1, :, None] * (m_loc[:, :, i - 1] - m[:, :, i]).exp()
        )
    # return C, n, m

    log_fg_matrix = log_prod_fg_matrix - torch.triu(
        torch.full([1, 1, 1, CS, CS], float("inf")).to(q), diagonal=1
    )

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact[:, :, :, :, None].transpose(
        -2, -1
    )  # (B, NH, NS, CS, CS)
    D_max, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)

    stab = torch.maximum(D_max, m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None])
    inter_C = (
        q * (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()
    ) @ C[:, :, :-1]
    inter_n = (
        q * (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()
    ) @ n[:, :, :-1, :]

    # D matrix stabilization
    log_D_matrix_stabilized = log_D_matrix - stab  # (B, NH, NS, CS, CS)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, NS, CS, CS)

    # combination matrix C
    qk_matrix = q @ k.transpose(-2, -1)  # (B, NH, NS, CS, CS)
    E_matrix = qk_matrix * D_matrix  # (B, NH, NS, CS, CS)

    normalizer = torch.maximum(
        (E_matrix.sum(dim=-1, keepdim=True) + inter_n).abs(),
        torch.exp(-stab),
    )  # (B, NH, NS, CS, 1)

    E_matrix_normalized = E_matrix / (normalizer + eps)

    # retrieved values
    intra = E_matrix_normalized @ v  # (B, NH, S, DH)
    inter = inter_C / (normalizer + eps)

    if return_last_state:
        return (intra + inter).view((B, NH, S, DH)), (C[:, :, -1], n[:, :, -1], m[:, :, -1])
    else:
        return (intra + inter).view((B, NH, S, DH))


if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 2048, 16
    q = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    k = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    v = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    i = torch.randn(B, H, S, device="cuda", requires_grad=True)
    f = torch.randn(B, H, S, device="cuda", requires_grad=True)
    c_prev = torch.randn(B, H, D, D, device="cuda", requires_grad=True)
    n_prev = torch.randn(B, H, D, device="cuda", requires_grad=True)
    m_prev = torch.randn(B, H, device="cuda", requires_grad=True) - float("inf")

    h_ref, (c_final, n_final, m_final) = chunkwise_simple(
        q, k, v, i, f, c_prev, n_prev.unsqueeze(-1), m_prev[..., None, None], 
        chunk_size=1024, return_last_state=True
    )

    from mlstm_kernels.torch.chunkwise.native.fw import mlstm_chunkwise_fw
    h, n, m, _, (c_all, n_all, m_all) = mlstm_chunkwise_fw(
        q, k, v, i, f, c_prev, n_prev, m_prev, 
        chunk_size=1024, return_all_states=True
    )

    h_ref64 = chunkwise_simple(
        q.double(), k.double(), v.double(), i.double(), f.double(), 
        c_prev.double(), n_prev.unsqueeze(-1).double(), m_prev[..., None, None].double(), 
        chunk_size=1024
    )

    h64, _, _, _, _ = mlstm_chunkwise_fw(
        q.double(), k.double(), v.double(), i.double(), f.double(), 
        c_prev.double(), n_prev.double(), m_prev.double(), 
        chunk_size=1024
    )

    qk_scale = D ** -.5
    torch.testing.assert_close(h, h_ref)
    torch.testing.assert_close(qk_scale * c_all[:, :, -D:, :], c_final)
    torch.testing.assert_close(qk_scale * n_all[:, :, -D:, None], n_final)
    torch.testing.assert_close(m_all[:, :, -1:, None], m_final)
    print(" === all good === ")
