import numpy as np
from typing import Callable
import itertools

cached_indices: dict[int, (np.array, np.array)] = dict()


def make_inds_parity_exc(maxexcitation: int):
    """Get Hamiltonian indices of states whose excitation number is even/odd and at most M"""
    even = []
    odd = []
    for i, comb in enumerate(itertools.product(range(maxexcitation + 1), repeat=5)):
        m = sum(comb)
        if m > maxexcitation:
            continue
        if m % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    return np.array(even), np.array(odd)


def get_inds_parity_exc(maxexcitation: int):
    if maxexcitation in cached_indices:
        return cached_indices[maxexcitation]

    even, odd = make_inds_parity_exc(maxexcitation)
    cached_indices[maxexcitation] = (even, odd)
    return even, odd


def make_kron_submatrix_indices(maxexcitation: int, inds: np.array):
    """Create the multidimensional index arrays used for partial kronecker.
    Each of the three subspaces must contain states 0 1 ... maxexcitation"""
    N = maxexcitation + 1  # states per bit, size of a factor matrix

    div4 = np.floor_divide(inds, N**4)
    inds -= div4 * N**4
    div3 = np.floor_divide(inds, N**3)
    inds -= div3 * N**3
    div2 = np.floor_divide(inds, N**2)
    inds -= div2 * N**2
    div1 = np.floor_divide(inds, N)
    inds -= div1 * N
    rowsA = div4[:, None]
    colsA = div4[None, :]
    rowsB = div3[:, None]
    colsB = div3[None, :]
    rowsC = div2[:, None]
    colsC = div2[None, :]
    rowsD = div1[:, None]
    colsD = div1[None, :]
    rowsE = inds[:, None]
    colsE = inds[None, :]

    return (
        rowsA,
        colsA,
        rowsB,
        colsB,
        rowsC,
        colsC,
        rowsD,
        colsD,
        rowsE,
        colsE,
    )


type SubmatrixKron = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
]
cached_kron: dict[int, tuple[SubmatrixKron, SubmatrixKron]] = dict()


def make_kron_submatrix_func(maxexcitation: int) -> tuple[SubmatrixKron, SubmatrixKron]:
    """Function factory. Function description:
    For calculating the kronecker product of A B C D E without explicitly creating the full product.
    Instead, result is split into even/odd. The two are also truncated based on maxexcitation.
    A B C D E must be square and of size maxexcitation+1"""

    global cached_kron
    if maxexcitation in cached_kron:
        return cached_kron[maxexcitation]

    even, odd = get_inds_parity_exc(maxexcitation)
    (
        rowsAeven,
        colsAeven,
        rowsBeven,
        colsBeven,
        rowsCeven,
        colsCeven,
        rowsDeven,
        colsDeven,
        rowsEeven,
        colsEeven,
    ) = make_kron_submatrix_indices(maxexcitation, even)
    (
        rowsAodd,
        colsAodd,
        rowsBodd,
        colsBodd,
        rowsCodd,
        colsCodd,
        rowsDodd,
        colsDodd,
        rowsEodd,
        colsEodd,
    ) = make_kron_submatrix_indices(maxexcitation, odd)

    def kron_submatrix_even(A, B, C, D, E):
        return (
            A[rowsAeven, colsAeven]
            * B[rowsBeven, colsBeven]
            * C[rowsCeven, colsCeven]
            * D[rowsDeven, colsDeven]
            * E[rowsEeven, colsEeven]
        )

    def kron_submatrix_odd(A, B, C, D, E):
        return (
            A[rowsAodd, colsAodd]
            * B[rowsBodd, colsBodd]
            * C[rowsCodd, colsCodd]
            * D[rowsDodd, colsDodd]
            * E[rowsEodd, colsEodd]
        )

    cached_kron[maxexcitation] = (kron_submatrix_even, kron_submatrix_odd)

    return kron_submatrix_even, kron_submatrix_odd


if __name__ == "__main__":
    make_kron_submatrix_indices(2, np.array([1, 2, 3, 9, 26, 27, 28]))
    pass
