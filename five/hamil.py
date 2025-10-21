# Here I will try to expand the two transmon Hamiltonian in the charge basis
import numpy as np
import cupy as cp
import scipy.linalg as spalg
import itertools
from five.util import make_kron_submatrix_func, get_inds_parity_exc


def excitation_trunc_indices(max_excitation: int):
    """Return Hamiltonian indices of states with too high excitation"""
    N = max_excitation + 1  # states per bit
    indices = []
    for a, b, c, d, e in itertools.product(range(N), repeat=5):
        if a + b + c + d + e > max_excitation:
            indices.append(a + b * N + c * N**2 + c * N**3 + e * N**4)
    return indices


type idx_map_type = dict[tuple, int]
cached_idx_maps: dict[int, tuple[idx_map_type, idx_map_type]] = (
    dict()
)  # global variable not the most elegant


def make_excitation_idx_map(max_excitation: int):
    even_map: dict[tuple, int] = {}
    odd_map: dict[tuple, int] = {}
    e = 0
    o = 0
    # the order we use is the one after filtering itertools.product output
    for state in itertools.product(range(max_excitation + 1), repeat=5):
        m = sum(state)
        if m > max_excitation:
            continue
        if m % 2 == 0:
            even_map[state] = e
            e += 1
        else:
            odd_map[state] = o
            o += 1
    return even_map, odd_map


def get_excitation_idx_map(max_excitation: int):
    """Returns two index maps, each for even/odd.
    Index this map with 5-tuples to get the hamiltonian index of that state"""
    global cached_idx_maps
    if max_excitation not in cached_idx_maps:
        cached_idx_maps[max_excitation] = make_excitation_idx_map(max_excitation)
    return cached_idx_maps[max_excitation]


def eig_even_odd(
    Ec1,
    Ec2,
    Ec3,
    Ec4,
    Ec5,
    Ej1,
    Ej2,
    Ej3,
    Ej4,
    Ej5,
    E12,
    E23,
    E13,
    E34,
    E45,
    E35,
    only_energy=False,
    M=20,
    C=30,
):
    """
    M: largest total excitation number of a state to keep in space
    Returns:
        eigenvalues and eigenvectors in bare basis
    """

    nstates = np.arange(-C, C + 1, step=1)
    ndiag = np.square(nstates)
    vals1, vecs1 = spalg.eigh_tridiagonal(ndiag * 4 * Ec1, -np.ones(2 * C) * Ej1 / 2)
    vals2, vecs2 = spalg.eigh_tridiagonal(ndiag * 4 * Ec2, -np.ones(2 * C) * Ej2 / 2)
    vals3, vecs3 = spalg.eigh_tridiagonal(ndiag * 4 * Ec3, -np.ones(2 * C) * Ej3 / 2)
    vals4, vecs4 = spalg.eigh_tridiagonal(ndiag * 4 * Ec4, -np.ones(2 * C) * Ej4 / 2)
    vals5, vecs5 = spalg.eigh_tridiagonal(ndiag * 4 * Ec5, -np.ones(2 * C) * Ej5 / 2)

    ndiag = np.diag(nstates)
    n1 = vecs1.T @ ndiag @ vecs1  # change into transmon bare basis
    n2 = vecs2.T @ ndiag @ vecs2
    n3 = vecs3.T @ ndiag @ vecs3
    n4 = vecs4.T @ ndiag @ vecs4
    n5 = vecs5.T @ ndiag @ vecs5

    N = M + 1  # number of states per transmon (at this stage)
    n1 = n1[:N, :N]  # truncate to NxN
    n2 = n2[:N, :N]
    n3 = n3[:N, :N]
    n4 = n4[:N, :N]
    n5 = n5[:N, :N]

    even, odd = get_inds_parity_exc(M)
    # calling kron() for diagonal matrices is slow compared to this
    D1 = np.repeat(vals1[:N], N**4)
    D2 = np.repeat(np.tile(vals2[:N], N), N**3)
    D3 = np.repeat(np.tile(vals3[:N], N**2), N**2)
    D4 = np.repeat(np.tile(vals4[:N], N**3), N)
    D5 = np.tile(vals5[:N], N**4)

    Dsum = D1 + D2 + D3 + D4 + D5
    Deven = np.diag(Dsum[even])
    Dodd = np.diag(Dsum[odd])

    ID = np.eye(N, N)
    # t = perf_counter()
    kron_submat_even, kron_submat_odd = make_kron_submatrix_func(M)
    even12 = 4 * E12 * kron_submat_even(n1, n2, ID, ID, ID)
    even23 = 4 * E23 * kron_submat_even(ID, n2, n3, ID, ID)
    even13 = 4 * E13 * kron_submat_even(n1, ID, n3, ID, ID)
    even34 = 4 * E34 * kron_submat_even(ID, ID, n3, n4, ID)
    even45 = 4 * E45 * kron_submat_even(ID, ID, ID, n4, n5)
    even35 = 4 * E35 * kron_submat_even(ID, ID, n3, ID, n5)

    odd12 = 4 * E12 * kron_submat_odd(n1, n2, ID, ID, ID)
    odd23 = 4 * E23 * kron_submat_odd(ID, n2, n3, ID, ID)
    odd13 = 4 * E13 * kron_submat_odd(n1, ID, n3, ID, ID)
    odd34 = 4 * E34 * kron_submat_odd(ID, ID, n3, n4, ID)
    odd45 = 4 * E45 * kron_submat_odd(ID, ID, ID, n4, n5)
    odd35 = 4 * E35 * kron_submat_odd(ID, ID, n3, ID, n5)
    # print(perf_counter() - t)
    # t = perf_counter()

    if only_energy:
        H_even = cp.asarray(Deven + even12 + even23 + even13 + even34 + even45 + even35)
        vals_even = cp.linalg.eigvalsh(H_even)
        vals_even = cp.asnumpy(vals_even)

        H_odd = cp.asarray(Dodd + odd12 + odd23 + odd13 + odd34 + odd45 + odd35)
        vals_odd = cp.linalg.eigvalsh(H_odd)
        vals_odd = cp.asnumpy(vals_odd)

        return np.sort(np.concatenate([vals_odd, vals_even]))

    H_even = cp.asarray(Deven + even12 + even23 + even13 + even34 + even45 + even35)
    # print(H_even.shape)
    vals_even, vecs_even = cp.linalg.eigh(H_even)
    vals_even = cp.asnumpy(vals_even)
    vecs_even = cp.asnumpy(vecs_even)

    # clear gpu mem?

    H_odd = cp.asarray(Dodd + odd12 + odd23 + odd13 + odd34 + odd45 + odd35)
    # print(H_odd.shape)
    vals_odd, vecs_odd = cp.linalg.eigh(H_odd)
    vals_odd = cp.asnumpy(vals_odd)
    vecs_odd = cp.asnumpy(vecs_odd)
    index_map_even, index_map_odd = get_excitation_idx_map(M)
    # print(perf_counter() - t)
    # t = perf_counter()
    return vals_even, vals_odd, vecs_even, vecs_odd, index_map_even, index_map_odd


if __name__ == "__main__":
    pass
