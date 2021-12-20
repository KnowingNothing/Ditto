from .. import _ffi_api


def share_axis_analysis(state1, state2):
    """Perform the share relationship analysis.

    Args:
        state1 (OpHyperState): the first op state
        state2 (OpHyperState): the second op state

        op1 and op2 may not have direct access relationship,
        i.e., the possible topology is
        op1-->op?-->op?-->...-->op2.
        So in this analysis, the share relationship should
        be propagated along the producer-consumer chain.

    Example:
        GEMM1 C[i, l] = A[i, k] * B[k ,l]
        ReLU  D[i', l'] = ReLU(C[i', l'])
        GEMM2 F[i'', j] = D[i, l''] * E[l'', j]

        op1 is GEMM1, op2 is GEMM2

        The result is [(i, i''), (l, l'')]

    Returns:
        List[Tuple(IterVar, IterVar)]
    """
    # Note: the IterVar in result shared_paris should use
    # the same naming format as OpHyperState
    # 'Si' for the i-th spatial axis, 'Ri' for the i-th reduce axis
    iters_dict = state1.get_all_iters_dict()
    iters_dict.update(state2.get_all_iters_dict())
    share_axis_pairs = _ffi_api.ShareAxisAnalysis(state1.op, state2.op)
    share_iters_pairs = [
        (iters_dict[x[0]], iters_dict[x[1]]) for x in share_axis_pairs
    ]
    return share_iters_pairs
