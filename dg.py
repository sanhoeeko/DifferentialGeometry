import numpy as np
from sympy import *

_global_simplify_option = True


def setSimplify(tf: bool):
    """
    If this program is too slow, probably it is problem of simplify.
    Disable it will not cause big issue.
    """
    global _global_simplify_option
    _global_simplify_option = tf


def sqrSimplify(expr):
    if not _global_simplify_option:
        return expr

    def _sqr_simplify(expr):
        expr = simplify(expr)
        p, rep = posify(simplify(expr * expr))
        return sqrt(p).expand().subs(rep)

    if expr.is_Matrix:
        return expr.applyfunc(_sqr_simplify)
    else:
        return _sqr_simplify(expr)


def normalize(vec):
    if not _global_simplify_option:
        return vec / norm(vec)
    return simplify(simplify(vec) / norm(vec))


def norm(vec):
    return sqrSimplify(sqrt(vec.dot(vec)))


def splitTensor(tensor: np.ndarray):
    """
    :param tensor: Array with 3 dims
    :return: two sympy Matrices, split by the first dimension (k dimension of Christoffel symbol)
    """
    mat1, mat2 = tensor
    return Matrix(mat1), Matrix(mat2)


class Curve:
    def __init__(self, func_array, variable):
        """
        :param func_array: [A1(t), A2(t), A3(t)]
        :param variable: t, for example above
        """
        self.x = Matrix(func_array)
        self.t = variable
        self.v = diff(self.x, self.t)  # velocity vector
        self.a = diff(self.v, self.t)  # accelerator vector
        self.tg = normalize(self.v)  # tangent vector
        self.n = normalize(self.v.cross(self.a).cross(self.v))  # normal vector
        self.curvature = sqrSimplify(norm(self.v.cross(self.a)) / norm(self.v) ** 3)
        self.b = normalize(self.v.cross(self.a))  # binormal vector
        self.torsion = simplify((diff(self.n, self.t) / norm(self.v)).dot(self.b))

    def getCenterOfCurvature(self):
        return simplify(self.x + self.n / self.curvature)


class Surface:
    def __init__(self, func_array, *variables):
        """
        :param func_array: [A1(u,v), A2(u,v), A3(u,v)]
        :param variable:  u,v for example
        """
        self.x = Matrix(func_array)
        self.u = variables[0]
        self.v = variables[1]
        self.xu = diff(self.x, self.u)
        self.xv = diff(self.x, self.v)
        self.n = normalize(self.xu.cross(self.xv))
        self.I = self.get_first_fundamental()
        self.II = self.get_second_fundamental()
        self.weingarten = self.II @ self.I.inv()
        self.g = simplify(det(self.I))
        self.K = simplify(det(self.weingarten))  # gaussian curvature
        self.lowerG = np.array(self.I)  # the metric tensor is implemented with numpy!
        self.highG = np.array(self.I.inv())
        self.lowerGamma = self.getChristoffel()
        self.Gamma = simplify(np.einsum('kl,lij->kij', self.highG, self.lowerGamma))
        self.has_principal = False

    def principal(self):
        """
        Get principal curvatures and directions.
        This process is too slow, so call it only when you trust the answer exists.
        """
        if not self.has_principal:
            self.principalCurvature, self.principalDirection = self.get_eigen_system()
            self.k1, self.k2 = self.principalCurvature
            self.vk1, self.vk2 = self.principalDirection
            self.has_principal = True

    def parameterize(self, uv_prime: list, new_u, new_v):
        """
        :param uv_prime: substitute u by u_prime(s,t), v by v_prime(s,t), where new_u=s, new_v=t
        """
        return Surface(
            simplify(self.x.subs(self.u, uv_prime[0]).subs(self.v, uv_prime[1])),
            new_u, new_v
        )

    def u_curve(self, u0):
        return self.x.subs(self.u, u0)

    def v_curve(self, v0):
        return self.x.subs(self.v, v0)

    def get_first_fundamental(self):
        lst = [self.xu.dot(self.xu), self.xu.dot(self.xv), self.xv.dot(self.xu), self.xv.dot(self.xv)]
        return simplify(Matrix(lst).reshape(2, 2))

    def get_second_fundamental(self):
        lst = [diff(self.x, self.u, self.u), diff(self.x, self.u, self.v),
               diff(self.x, self.v, self.u), diff(self.x, self.v, self.v)]
        lst = [simplify(y.dot(self.n)) for y in lst]
        return Matrix(lst).reshape(2, 2)

    def get_eigen_system(self):
        eigsys = self.weingarten.eigenvects()
        eig = [eigsys[0][0], eigsys[1][0]]
        vec = [eigsys[0][2], eigsys[1][2]]
        return eig, vec

    def getChristoffel(self):
        r"""
        :return: Christoffel symbol of \Gamma^k_{ij}, and the index sequence is (k,i,j)
        """
        lst = [diff(self.x, self.u, self.u), diff(self.x, self.u, self.v),
               diff(self.x, self.v, self.u), diff(self.x, self.v, self.v)]
        ri = [diff(self.x, self.u), diff(self.x, self.v)]
        ij_k1 = np.array([simplify(y.dot(ri[0])) for y in lst]).reshape((2, 2))
        ij_k2 = np.array([simplify(y.dot(ri[1])) for y in lst]).reshape((2, 2))
        return np.array([ij_k1, ij_k2])

    def getLineOfCurvatureODE(self):
        """
        :return: du/dv = f(u,v)
        """
        # as the product of two symmetric matrices, weingarten is not symmetric!
        a = self.weingarten[0, 0]
        b = self.weingarten[0, 1]
        c = self.weingarten[1, 0]
        d = self.weingarten[1, 1]
        if c == 0:
            du_dv_of_k1 = simplify(b / (d - a))
            du_dv_of_k2 = simplify(b / (d - a))
        else:
            a_d = simplify(a - d)
            delta = simplify(a_d ** 2 + 4 * b * c)
            # note that sqrSimplify is regardless of + and - !
            du_dv_of_k1 = simplify(a_d / (2 * c) + sqrSimplify(sqrt(delta) / (2 * c)))
            du_dv_of_k2 = simplify(a_d / (2 * c) - sqrSimplify(sqrt(delta) / (2 * c)))
        return du_dv_of_k1, du_dv_of_k2

    def getAsymptoteODE(self):
        """
        :return: du/dv = f(u,v)
        """
        L = self.II[0, 0]
        M = self.II[0, 1]
        if L == 0:
            # if L is zero, another asymptote is dv = 0 i.e. v curve
            N = self.II[1, 1]
            return [simplify(-N / (2 * M))]
        sqr_det = sqrSimplify(det(self.II))
        du_dv_1 = simplify((-M + sqr_det) / L)
        du_dv_2 = simplify((-M - sqr_det) / L)
        return [du_dv_1, du_dv_2]
