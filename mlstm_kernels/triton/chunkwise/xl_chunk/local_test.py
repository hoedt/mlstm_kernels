if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 64, 16
    q = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    k = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    v = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    i = torch.randn(B, H, S, device="cuda", requires_grad=True)
    f = torch.randn(B, H, S, device="cuda", requires_grad=True)
    c_prev = torch.randn(B, H, D, D, device="cuda", requires_grad=True)
    n_prev = torch.randn(B, H, D, device="cuda", requires_grad=True)
    m_prev = torch.randn(B, H, device="cuda", requires_grad=True)

    from mlstm_kernels.torch.chunkwise.native.fw import mlstm_chunkwise_fw
    h, l, m, _, (c_all, n_all, m_all) = mlstm_chunkwise_fw(
        q, k, v, i, f, c_prev, n_prev, m_prev,
        chunk_size=32, return_all_states=True, eps=1e-7
    )

    h_grad = torch.randn_like(h)
    c_grad = torch.randn_like(c_prev)
    n_grad = torch.randn_like(n_prev)
    m_grad = torch.randn_like(m_prev)

    # intermezzo
    # from mlstm_kernels.torch.chunkwise.native.bw import _mlstm_chunkwise__parallel_bw_dQKV, mlstm_chunkwise__recurrent_bw_dC
    # b = torch.nn.functional.logsigmoid(f.view(B, H, 2, 32)).cumsum(dim=-1)
    # dq_intra_ref, dk_intra_ref, dv_intra_ref, aux_ref = _mlstm_chunkwise__parallel_bw_dQKV(
    #     q, k, v, b, i.view(B, H, 2, 32), m, l, c_all[:, :, :-D, :], m_all[:, :, :-1], h_grad,
    #     chunk_size=32, num_chunks=2
    # )
    # dc_ref, dn_ref = mlstm_chunkwise__recurrent_bw_dC(
    #     q, b, m_all, m, h_grad, l, aux_ref, c_grad, n_grad,
    #     chunk_size=32, num_chunks=2
    # )

    # from mlstm_kernels.torch.chunkwise.triton_xl_chunk.bw import mlstm_chunkwise__parallel_bw_dV, mlstm_chunkwise__parallel_bw_dQ, mlstm_chunkwise__recurrent_bw_dC
    # dv_intra, aux_tmp = mlstm_chunkwise__parallel_bw_dV(
    #     q, k, v, i, b, l, m, h_grad,
    #     chunk_size=32,
    #     siz_b_LQ=16, siz_b_LKV=16,
    #     siz_b_DHQK=16, siz_b_DHHV=16
    # )
    # dq_intra, aux = mlstm_chunkwise__parallel_bw_dQ(
    #     q, k, v, i, b, c_all, n_all, m_all, l, m, aux_tmp, h_grad,
    #     chunk_size=32,
    #     siz_b_LQ=16, siz_b_LKV=16,
    #     siz_b_DHQK=16, siz_b_DHHV=16
    # )
    # dc, dn = mlstm_chunkwise__recurrent_bw_dC(
    #     q, f, m_all, m, h_grad, l, aux, c_grad, n_grad,
    #     chunk_size=32, save_states_every_nth_chunk=1
    # )

    # torch.testing.assert_close(aux, aux_ref, atol=1e-4, rtol=1.3e-6)
    # torch.testing.assert_close(dv_intra, dv_intra_ref, atol=5e-5, rtol=1.3e-6)
    # torch.testing.assert_close(dn, dn_ref)
    # torch.testing.assert_close(dc, dc_ref)
    # end intermezzo

    from mlstm_kernels.torch.chunkwise.native.bw import mlstm_chunkwise_bw
    dq_ref, dk_ref, dv_ref, di_ref, df_ref, dc_ref, dn_ref, dm_ref = mlstm_chunkwise_bw(
        q, k, v, i, f, c_prev, n_prev, m_prev, c_all, n_all, m_all,
        vecL_out=l, vecM_out=m,
        matDeltaH=h_grad, matDeltaC_last=c_grad, vecDeltaN_last=n_grad, scaDeltaM_last=m_grad,
        CHUNK_SIZE=32, EPS=1e-7
    )

    from mlstm_kernels.torch.chunkwise.triton_xl_chunk.bw import mlstm_chunkwise_bw
    dq, dk, dv, di, df, dc, dn, dm = mlstm_chunkwise_bw(
        q, k, v, i, f, c_prev, n_prev, m_prev, c_all, n_all, m_all,
        vecL_out=l, vecM_out=m,
        matDeltaH_out=h_grad, matDeltaC_last=c_grad, vecDeltaN_last=n_grad, scaDeltaM_last=m_grad,
        chunk_size=32, chunk_size_inter=16, chunk_size_intra=32,
        siz_b_L_loop=16, siz_b_L_parallel=16, eps=1e-7
    )

    torch.testing.assert_close(dq, dq_ref)
    torch.testing.assert_close(dk, dk_ref)
    torch.testing.assert_close(dv, dv_ref)
    torch.testing.assert_close(di, di_ref)
    torch.testing.assert_close(df, df_ref)
    torch.testing.assert_close(dc, dc_ref)
    torch.testing.assert_close(dn, dn_ref)
    torch.testing.assert_close(dm, dm_ref)
    print(" === all good === ")
