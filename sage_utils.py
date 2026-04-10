"""
Synopsis: Sympy sucks at inverting symbolic matrices!
Author: Elvis Dohmatob (with the crucial help of ChatGPT)
"""

import sympy as sp
from sage.all import SR, QQ, PolynomialRing, matrix
from sage.misc.sage_eval import sage_eval


def sympy_matrix_to_sage(q_sympy, base_field=QQ, use_fraction_field=True,
                         order='lex'):
    """Convert Sympy matrix into SageMath format."""
    if not isinstance(q_sympy, sp.MatrixBase):
        raise TypeError("q_sympy must be a SymPy Matrix")

    symbols = sorted(q_sympy.free_symbols, key=lambda s: s.name)
    names = [s.name for s in symbols]

    if names:
        R = PolynomialRing(base_field, names=names, order=order)
    else:
        R = PolynomialRing(base_field, names=['dummy'], order=order)

    P = R.fraction_field() if use_fraction_field else R

    # Crucial: use generators from the ACTUAL target parent
    if names:
        local_dict = dict(zip(names, P.gens()))
    else:
        local_dict = {}

    def convert_entry(expr):
        s = sp.sstr(sp.sympify(expr))
        return sage_eval(s, locals=local_dict)

    entries = [
        convert_entry(q_sympy[i, j])
        for i in range(q_sympy.rows)
        for j in range(q_sympy.cols)
    ]

    M = matrix(P, q_sympy.rows, q_sympy.cols, entries)
    return R, P, M


def sage_matrix_to_sympy(M_sage):
    """
    Robust SageMath -> SymPy conversion.
    Avoids SR/string coercion when possible.
    """
    return sp.Matrix([
        [
            M_sage[i, j]._sympy_() if hasattr(M_sage[i, j], "_sympy_")
            else sp.sympify(str(M_sage[i, j]).replace("^", "**"))
            for j in range(M_sage.ncols())
        ]
        for i in range(M_sage.nrows())
    ])


def sympy_inverse_via_sage(mat, base_field=QQ, order='lex'):
    """Fast computation of inverse of Sympy matrix using SageMath backend."""
    _, _, mat = sympy_matrix_to_sage(mat, base_field=base_field, order=order,
                                     use_fraction_field=True)
    inv = ~mat
    inv = inv.apply_map(lambda r: 0 if r == 0
        else SR(r.numerator().factor()).collect_common_factors() /
             SR(r.denominator().factor()).collect_common_factors())
    return sage_matrix_to_sympy(inv)
