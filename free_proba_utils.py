"""
Synopsis: Helper functions for FPT calculations (e.g R-transform, etc.).
Author: Elvis Dohmatob, Arjun Subramonian
"""
import sympy as sp
from sympy import (Matrix, Identity, ZeroMatrix, BlockMatrix,
                   block_collapse, inv_quick, simplify, factor, sqrt,
                   MatrixExpr, Pow, Inverse, MatPow, Add, Mul, MatAdd, MatMul, Transpose)
from sympy.matrices.expressions.matexpr import Transpose
import numpy as np


def is_zero(obj):
    # explicit zero matrix expression
    if obj.is_ZeroMatrix:
        return True

    # explicit matrices often have this
    z = getattr(obj, "is_zero_matrix", None)
    if z is not None:
        return z is True

    # scalars
    z = getattr(obj, "is_zero", None)
    if z is not None:
        return z is True
    else:
        raise RuntimeError


def is_scalar_times_identity(M):
    n, m = M.shape
    if n != m:
        return False
    I = Identity(n)
    a = M[0, 0]
    return M / a == I


def matrix_expr_to_scalar(expr, mat_to_scalar):
    """
    Convert one matrix expression into a scalar expression by replacing
    MatrixSymbol objects using mat_to_scalar.

    Examples:
        X      -> x
        X + Y  -> x + y
        X*Y    -> x*y
        X**2   -> x**2
        X.I    -> 1/x
        X.T    -> x
        Identity(n) -> 1
        ZeroMatrix(...) -> 0
    """
    # direct hit
    if expr in mat_to_scalar:
        return mat_to_scalar[expr]

    # atoms / simple cases
    if expr.is_Number or expr.is_Symbol:
        return expr
    if isinstance(expr, Identity):
        return sp.Integer(1)
    if isinstance(expr, ZeroMatrix):
        return sp.Integer(0)
    if isinstance(expr, Inverse):
        return 1 / matrix_expr_to_scalar(expr.arg, mat_to_scalar)
    if isinstance(expr, Transpose):
        return matrix_expr_to_scalar(expr.arg, mat_to_scalar)
    if isinstance(expr, MatPow):
        base = matrix_expr_to_scalar(expr.base, mat_to_scalar)
        exp = expr.exp
        return base**exp
    if isinstance(expr, (MatAdd, sp.Add)):
        return sp.Add(*(matrix_expr_to_scalar(a, mat_to_scalar)
                      for a in expr.args))
    if isinstance(expr, (MatMul, sp.Mul)):
        return sp.Mul(*(matrix_expr_to_scalar(a, mat_to_scalar)
                      for a in expr.args))

    # generic recursive fallback
    if hasattr(expr, "args") and expr.args:
        new_args = [matrix_expr_to_scalar(a, mat_to_scalar) for a in expr.args]
        try:
            return expr.func(*new_args)
        except Exception:
            # last fallback: rebuild as ordinary scalar expression if possible
            return sp.sympify(expr.func(*new_args))

    raise TypeError(f"Don't know how to convert expression: {expr!r}")


def scalarize_block_matrix(Q, mat_to_scalar):
    """
    Convert an n x m block matrix Q into an n x m scalar matrix q
    by replacing matrix symbols with scalar symbols.

    Parameters
    ----------
    Q : BlockMatrix or explicit Matrix of block expressions
    mat_to_scalar : dict
        Example: {X: x, Y: y, Z: z, S: s}

    Returns
    -------
    sympy.Matrix
    """
    if isinstance(Q, BlockMatrix):
        blocks = Q.blocks
    elif isinstance(Q, sp.MatrixBase):
        blocks = Q
    else:
        try:
            blocks = Q.blocks
        except AttributeError:
            raise TypeError("Q must be a BlockMatrix or a Matrix of block expressions.")

    n, m = blocks.shape
    return sp.Matrix(n, m, lambda i, j: sp.simplify(
        matrix_expr_to_scalar(blocks[i, j], mat_to_scalar)
    ))


def decompose_Q(Q, rands):
    """rands should contain all the random quantities (symbolic matrices) that
    appear as blocks of Q."""
    rands = rands + [item.T for item in rands]
    rands = rands + [-item for item in rands]
    rows = []
    for i in range(Q.blocks.shape[0]):
        row = []
        for j in range(Q.blocks.shape[1]):
                block = Q.blocks[i, j]
                # if block in rands:
                if len([rand for rand in rands if rand in block.free_symbols]):
                    item = -block
                else:
                    item = ZeroMatrix(*block.shape)
                row.append(item)
        rows.append(row)
    Qx = BlockMatrix(rows)
    F = block_collapse(Q + Qx)

    return Qx, F


def R_transform(Qx, G, variances={}, rands=None):
    """
    Computes the R-transform of Qx evaluated at G, i.e

        Entrace(G \otimes I - Qx)^{-1}.

    The output is an n0xn0 complex matrix with

      Rij = sum_{k,L} sigma(i,k;L,j) * alpha_k * G[k,L],

    where the scalar sigma(i,k;L,j) is the covariance between the entries of
    block (i, k) and block (L, j) of Qx, and alpha_k is the length of block
    (k, L).

    We work in the setting where sigma(i,k;L,j) in {-var, 0, var}. For example,
    typically, var = 1 / (n * lambd)
    """
    from sympy.stats import Normal, E
    mapping = {}
    rands = list(variances.keys())
    for i, Z in enumerate(rands):
        std = sqrt(variances.get(Z, 1))
        z = Normal("z_%s" % i, 0, std)
        mapping[Z] = z
        mapping[Z.T] = z
    qx = scalarize_block_matrix(Qx, mapping)
    n0 = Qx.blocks.shape[0]
    assert G.shape == (n0, n0)
    R = np.zeros((n0, n0), dtype="object")
    for i in range(n0):
        for j in range(n0):
            for k in range(n0):
                if k == j:
                    continue
                block1 = Qx.blocks[i, k]
                if is_zero(block1):
                    continue
                for L in range(n0):
                    if L == i:
                        continue
                    if (i, k) == (L, j):
                        continue
                    block2 = Qx.blocks[L, j]
                    if is_zero(block2):
                        continue
                    if block2.shape != block1.shape[::-1]:
                        continue
                    cov = E(qx[i, k] * qx[L, j])
                    assert block2 not in [block1, -block1], (block2, block1)
                    shape = Qx.blocks[k, L].shape
                    assert shape[1] == shape[0]
                    m = shape[0]
                    R[i, j] += cov * G[k, L] * m
    return Matrix(R)


def lens(block, random_matrices):
    if is_zero(block):
        return None, None
    shape = block.shape
    free_symbols = list(block.free_symbols)
    if len(free_symbols) == 1:
        free_symbol = free_symbols[0]
        if free_symbol in random_matrices:
            c = sp.sqrt(block[0, 0] / free_symbol[0, 0])
            return free_symbol, c
    return None, None


def R_transform_bis(Q0, G, variances):
    """
    REturns the R-transform of G w.r.t to the random structure in the nb x nb
    block matrix Q0.
    """
    nb = Q0.blocks.shape[0]
    random_matrices = list(variances.keys())
    R = np.zeros((nb, nb), dtype=object)
    for i in range(nb):
        for k in range(nb):
            block = Q0.blocks[i, k]
            free_symbol, c = lens(block, random_matrices)
            if free_symbol is None:
                continue
            for L in range(nb):
                for j in range(nb):
                    other_block = Q0.blocks[L, j]
                    other_free_symbol, other_c = lens(other_block,
                    random_matrices)
                    if other_free_symbol is None:
                        continue
                    if (other_free_symbol == free_symbol
                    and other_block.shape == block.shape[::-1]):
                        alpha = other_block.shape[0]
                        var = c * other_c
                        var *= variances[free_symbol]
                        R[i, j] += var * alpha * G[k, L]
    return sp.Matrix(R)


def symmetrize_matrix(q):
    n, m = q.shape
    q = np.array(q)
    q_ = np.zeros((n + m, n + m), dtype=object)
    q_[:m, -n:] = q
    q_[-n:, :m] = q.T
    return Matrix(q_)


def symmetrize_block_matrix(Q):
    adj = []
    for i in range(Q.T.blocks.shape[0]):
        adj_row = []
        for j in range(Q.blocks.shape[1]):
            adj_row.append(ZeroMatrix(Q.T.blocks[i, 0].shape[0],
            Q.blocks[0, j].shape[1]))
        for j in range(Q.T.blocks.shape[1]):
            adj_row.append(Q.T.blocks[i, j])
        adj.append(adj_row)

    for i in range(Q.blocks.shape[0]):
        adj_row = []
        for j in range(Q.blocks.shape[1]):
            adj_row.append(Q.blocks[i, j])
        for j in range(Q.T.blocks.shape[1]):
            adj_row.append(ZeroMatrix(Q.blocks[i, 0].shape[0],
            Q.T.blocks[0, j].shape[1]))
        adj.append(adj_row)
    return BlockMatrix(adj)


def matricize_expr(expr, conv_map):
    if expr.func == Pow:
        subexpr = matricize_expr(expr.args[0], conv_map)
        if issubclass(type(subexpr), MatrixExpr):
            if expr.args[1] == -1:
                return Inverse(subexpr, -1)
            return MatPow(subexpr, expr.args[1])
        return expr
    elif expr.func == Add:
        mat_args = []
        scal_args = []
        for a in expr.args:
            subexpr = matricize_expr(a, conv_map)
            if issubclass(type(subexpr), MatrixExpr):
                mat_args.append(subexpr)
            else:
                scal_args.append(subexpr)
        if len(mat_args) == 0:
            return expr
        scal_args = [a * Identity(mat_args[0].shape[0]) for a in scal_args]
        return MatAdd(*(mat_args + scal_args))
    elif expr.func == Mul:
        args = [matricize_expr(a, conv_map) for a in expr.args]
        return MatMul(*args)
    elif expr in conv_map:
        new_expr = conv_map[expr]
        if new_expr.func == Transpose:
            return new_expr.args[0]
        return new_expr
    else:
        return expr
