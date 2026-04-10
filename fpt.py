"""
Synopsis: Generic tools for free probability theory of rectangular random
Gaussian matrices.
Author: Elvis Dohmatob, Arjun Subramonian
"""
import os
import warnings
import argparse

import pickle
import numpy as np

import sympy as sp
from sympy import (Function, symbols, Matrix, matrices, MatrixSymbol,
                   simplify, simplify, factor, Eq, pretty_print)
from sympy.matrices.expressions.matexpr import MatrixElement

import free_proba_utils


def find_duplicates(q_inv, Q):
    """Gather entries of given matrix into equivalence classes of pairs of
    indices with same value q_inv[i, j]."""
    assert q_inv.shape == Q.blocks.shape
    nb = q_inv.shape[0]
    classes = {}
    for i in range(nb):
        for j in range(nb):
            term = simplify(q_inv[i, j])
            shape = Q.blocks[i, j].shape
            if len(term.free_symbols) > 0:
                key = term, shape
                classes[key] = classes.get(key, []) + [(i, j)]
    return dict((term, indices) for (term, _), indices in classes.items())


def remove_duplicates(G, expr, classes):
    if not classes:
        return expr
    for indices in classes.values():
        i0, j0 = indices[0]
        for i, j in indices[1:]:
            expr = expr.subs(G[i, j], G[i0, j0])
    return simplify(expr)

def construct_fixed_point_equations(G, diff_inv, row_idx, col_idx,
                                    classes=None):
    expr = diff_inv[row_idx, col_idx]
    expr = remove_duplicates(G=G, expr=expr, classes=classes)
    eqs = [Eq(G[row_idx, col_idx], expr)]

    # Gather subtree of fixed-point equations contraining the target expr
    prev_terms = set()
    while True:
        flag = False
        for a in expr.atoms(MatrixElement):
            if (a.args[1], a.args[2]) not in prev_terms:
                eqs.append(Eq(G[a.args[1], a.args[2]],
                              diff_inv[a.args[1], a.args[2]]))
                expr = expr.subs(G[a.args[1], a.args[2]],
                                 diff_inv[a.args[1], a.args[2]])
                prev_terms.add((a.args[1], a.args[2]))
                flag = True
        if not flag:
            break
    tmp = list(set(eqs[1:]))
    if eqs[0] in tmp:
        tmp.remove(eqs[0])
    eqs = eqs[:1] + tmp

    eqs = [factor(simplify(eq)) for eq in eqs]
    return expr, eqs


def calc(Q, row_idx: int=None, col_idx: int=None, symmetric: bool=False,
         variances: dict=None, subs: dict={}, use_sagemath: bool=False,
         verbose: int=1, display=pretty_print):
    """
    Parameters
    ----------
    Q: BlockMatrix
        Linear pencil for target rational expression (expr)
    row_idx: int (required)
        Row index of expr within the given linear pencil Q
    col_idx: int (required)
        Column index of expr within the given linear pencil Q
    variances: dict (required)
        For every random matrix Z, Z1, Z2, W, ..., appearing in expr, this dict
        should contain a key-value pair. Typical examples of such pairs include:

        Z: 1 / (n * lambd), Z1: 1 / (n * lambd),  Z2: 1 / (n * lambda), W: 1 / d
    use_sagemath: bool (optional)
        If True, SageMath backend will be used for computing the inverse of
        scalar matrices (it its lightspeed faster than Sympy!)
    subs: dict (optional)
        Dictionary of variable substitutions to do at the end of the analysis
        to make the final results more compact / insightful. Typical examples
        include:

        (input dim) d: n * phi, (network width) m: n * phi,

        where n is the sample size. More advanced include:

        (dataset1 size) n1: n * p1, (dataset2 size) n2: n * p2, etc.
    """
    if None in [row_idx, col_idx]:
        raise ValueError("Both row_idx and col_idx required in function call!")
    if variances is None:
        raise ValueError("A dictionary must be specified for variances")
    random_matrices = list(variances.keys())

    nb = Q.blocks.shape[0]
    if verbose:
        print("Q = ")
        display(Q)
        print("Size of pencil: %i x %i" % (nb, nb))
        print()
        print("Variance of entries of each random matrix:")
        for k, v in variances.items():
            print("\t%s: %s" % (k, v))

    free_symbols = list(Q.free_symbols)
    free_symbols = dict(zip(map(str, free_symbols), free_symbols))
    random_matrices = list(random_matrices)
    rands = []
    for rmat in random_matrices:
        if isinstance(rmat, str) and rmat in free_symbols:
            rands.append(free_symbols[rmat])
        else:
            rands.append(rmat)

    if verbose >= 2:
        for rand in rands:
            if rand not in free_symbols.values():
                warnings.warn(
                "Random matrix %s doesn't appear in pencil" % rand)
    rands = list(set(rands).intersection(free_symbols.values()))

    matrix_symbols = dict((k, v)
    for k, v in free_symbols.items() if isinstance(v, MatrixSymbol))
    matrix_symbols.update(dict(("%s_T" % k, v.T if v in random_matrices else v)
    for k, v in matrix_symbols.items()))

    print("\nPreparing matrix-to-scalar identification...")
    matrix_to_scalar_map = dict((v, symbols("x%d" % idx, commutative=True))
    for idx, v in enumerate(matrix_symbols.values()))
    tmp = {}
    for k, v in matrix_symbols.items():
        if k.endswith("_T"):
            if v.T not in random_matrices:
                tmp[v.T] = matrix_to_scalar_map[v]
    matrix_to_scalar_map.update(tmp)

    def _print_identification(exclude_random=False):
        treated = []
        for k, v in matrix_to_scalar_map.items():
            if v in treated:
                continue
            if exclude_random and (k in rands or k.T in rands):
                continue
            print("\t%s --> %s" % (v, k))
            treated.append(v)

    if verbose:
        _print_identification()
    print("\nSplitting the input pencil...")
    Qx, F = free_proba_utils.decompose_Q(Q, rands=rands)
    if verbose:
        print("Qx = ")
        display(Qx)
        print("F = ")
        display(F)

    print("\nScalarizing matrices...")
    q = free_proba_utils.scalarize_block_matrix(Q, matrix_to_scalar_map)
    qx = free_proba_utils.scalarize_block_matrix(Qx, matrix_to_scalar_map)
    f = free_proba_utils.scalarize_block_matrix(F, matrix_to_scalar_map)
    if verbose:
        print("qx = ")
        display(qx)
        print("f = ")
        display(f)


    print("\nChecking that pencil makes sense...")
    if use_sagemath:
        print("We'll use SageMath (for speed) to compute inverse of pencil...)")
        import sage_utils
        q_inv = sage_utils.sympy_inverse_via_sage(q)
    else:
        q_inv = q.inv()
    print("q_inv[%i, %i] = " % (row_idx, col_idx))
    display(q_inv[row_idx, col_idx])
    print("\nWhere")
    _print_identification()
    if symmetric:
        Q_, Qx_ = Q, Qx
    else:
        print("\nSymmetrizing the provided linear pencil (Q)...")
        Q_ = free_proba_utils.symmetrize_block_matrix(Q)
        Qx_ = free_proba_utils.symmetrize_block_matrix(Qx)
        if verbose:
            print("Q_ = ")
            display(Q_)

    print("\nIdentifying indices of entries which must be equal in G...")
    classes = find_duplicates(q_inv, Q)
    if verbose:
        for val, indices in classes.items():
            print("\t%s: %s" % (indices, val))

    # Setup sought-for matrix of normalized traces
    print("\nInferring relevant mask...")
    print()
    mask = np.ones(q.shape, dtype=int)

    # trace of zero matrix is zero
    mask[np.array(q_inv, dtype=object) == 0] = 0

    # trace of non-square matrix is zero (by convention!)
    for i in range(nb):
        for j in range(nb):
            shape = Q.blocks[i, j].shape
            if shape[0] != shape[1]:
                mask[i, j] = 0

    mask = Matrix(mask)
    if verbose:
        display(mask)

    print("\nPreparing G matrix...")
    if symmetric:
        G = np.zeros((nb, nb), dtype=object)
        for i in range(nb):
            for j in range(nb):
                if i <= j:
                    G[i, j] = sp.Symbol("G_{%d, %d}" % (i, j))
                else:
                    G[i, j] = G[j, i]
        G = sp.Matrix(G)
    else:
        G = sp.MatrixSymbol("G", nb, nb).as_explicit()
    G = matrices.dense.matrix_multiply_elementwise(G, mask)
    if symmetric:
        G_ = G
    else:
        G_ = free_proba_utils.symmetrize_matrix(G)

    # Compute R-transform
    print("\nComputing R-transform R = R(G) matrix...")
    r_ = free_proba_utils.R_transform(Qx_, G_, variances=variances)
    r_.simplify()
    if symmetric:
        r = r_
    else:
        r = r_[nb:, :nb]

    if verbose:
        print("r = ")
        display(r)

    # Invert difference of R-transform and constant matrices
    print("\nComputing F - R(G) matrix...")
    diff = f - r
    diff = diff.subs(subs)
    diff = simplify(diff)
    if verbose:
        print("F - R(G) = ")
        display(diff)

    print("\nComputing inverse...")
    if use_sagemath:
        print("(We'll use SageMath for speed...)")
        mapping = {}
        cnt = 0
        for i in range(nb):
            for j in range(nb):
                if G[i, j] != 0:
                    mapping[G[i, j]] = "g%i" % cnt
                    cnt += 1
        reverse_mapping = dict((v, k) for k, v in mapping.items())
        diff_inv = sage_utils.sympy_inverse_via_sage(
            diff.subs(mapping)).subs(reverse_mapping)
        diff_inv = np.array(diff_inv, dtype=object)
    else:
        diff_inv = np.zeros((nb, nb, 1), dtype=object)
        for j in range(nb):
            b = np.zeros(nb, dtype=object)
            b[j] = 1
            b = sp.Matrix(b[:, None])
            diff_inv[:, j] = diff.solve(b, method="LU").applyfunc(sp.factor)
        diff_inv = diff_inv[:, :, 0]
    diff_inv[np.array(G) == 0] = 0
    diff_inv = sp.Matrix(diff_inv)

    if verbose >= 1:
        print("Computing inv(F - R^o)")

    print("\nComputing fixed-point equations...")
    _, eqs = construct_fixed_point_equations(G=G, diff_inv=diff_inv,
                                             row_idx=row_idx,
                                             col_idx=col_idx,
                                             classes=classes)

    print("\nMatricizing equations...")

    print("\nThe fixed-point equations are:\n")
    scalar_to_matrix_map = {v : k for k, v in matrix_to_scalar_map.items()}
    returned_eqs = []
    for eq in eqs:
        if eq.rhs.free_symbols.intersection(set(matrix_to_scalar_map.values())):
            matrix_rhs = simplify(
            free_proba_utils.matricize_expr(eq.rhs, scalar_to_matrix_map))
            returned_eqs.append(Eq(eq.lhs, Function(r'trbar')(matrix_rhs)))
            eq = eq.subs(subs)
            display(returned_eqs[-1])
        else:
            returned_eqs.append(eq)
            display(eq)
        print("\n")

    return returned_eqs
