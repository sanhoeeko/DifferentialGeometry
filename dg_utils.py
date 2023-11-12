import numpy as np
from sympy import *

_global_simplify_option = True


def setSimplify(tf):
    """
    If this program is too slow, probably it is problem of simplify.
    Disable it will not cause big issue.
    :param tf: True: enable simplify. False: disable simplify. 'positive': enable simplify, and discard Abs and sign.
    """
    global _global_simplify_option
    _global_simplify_option = tf


def Simplify(expr):
    if not _global_simplify_option:
        return expr
    else:
        if _global_simplify_option == 'positive':
            expr = simplify(expr)
            expr = expr.replace(Abs, lambda x: x)
            expr = expr.replace(sign, lambda x: 1)
            return simplify(expr)
        else:
            return simplify(expr)


def sqrSimplify(expr):
    if not _global_simplify_option:
        return expr

    def _sqr_simplify(expr):
        expr = Simplify(expr)
        p, rep = posify(simplify(expr * expr))
        return sqrt(p).expand().subs(rep)

    if expr.is_Matrix:
        return expr.applyfunc(_sqr_simplify)
    else:
        return _sqr_simplify(expr)


def normalize(vec):
    if not _global_simplify_option:
        return vec / norm(vec)
    return Simplify(Simplify(vec) / norm(vec))


def norm(vec):
    return sqrSimplify(sqrt(vec.dot(vec)))


def var(varname: str):
    return symbols(varname, positive=True)


def splitTensor(tensor: np.ndarray):
    """
    :param tensor: Array with 3 dims
    :return: two sympy Matrices, split by the first dimension (k dimension of Christoffel symbol)
    """
    mat1, mat2 = tensor
    return Matrix(mat1), Matrix(mat2)


def RotationMatrixX(th):
    return Matrix([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])


def RotationMatrixZ(th):
    return Matrix([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])


def RotationMatrixY(th):
    return Matrix([[cos(th), 0, -sin(th)], [0, 1, 0], [sin(th), 0, cos(th)]])
