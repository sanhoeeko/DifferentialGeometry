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


def det2(mat):
    return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]


def inv2(mat):
    return Matrix([[mat[1, 1], -mat[0, 1]], [-mat[1, 0], mat[0, 0]]]) / det2(mat)


def eigsys2(mat):
    a = mat[0, 0]
    b = mat[0, 1]
    c = mat[1, 0]
    d = mat[1, 1]
    delta = Simplify((a - d) ** 2 + 4 * b * c)
    trace = a + d
    sq = sqrSimplify(sqrt(delta))
    j1 = (trace - sq) / 2
    j2 = (trace + sq) / 2
    s11 = (j1 - d) / c
    s12 = (j2 - d) / c
    eigvects = Matrix([[s11, s12], [1, 1]])
    return (j1, j2), eigvects


def var(varname: str):
    return symbols(varname, positive=True)


def splitTensor(tensor: np.ndarray):
    """
    :param tensor: Array with 3 dims
    :return: two sympy Matrices, split by the first dimension (k dimension of Christoffel symbol)
    """
    mat1, mat2 = tensor
    return Matrix(mat1), Matrix(mat2)


def Lambdify(expr):
    """
    A safe lambdify. It preserves input shape even when the function returns a constant.
    """
    pass


def RotationMatrixX(th):
    return Matrix([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])


def RotationMatrixZ(th):
    return Matrix([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])


def RotationMatrixY(th):
    return Matrix([[cos(th), 0, -sin(th)], [0, 1, 0], [sin(th), 0, cos(th)]])
