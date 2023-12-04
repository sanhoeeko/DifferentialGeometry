from dg_utils import *


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
        self.torsion = Simplify((diff(self.n, self.t) / norm(self.v)).dot(self.b))

    def getCenterOfCurvature(self):
        return Simplify(self.x + self.n / self.curvature)

    def arc(self):
        """
        :return: the arc length parameter s
        """
        return integrate(norm(self.v), self.t)

    def parameterize(self, t_as_func_of_s, s):
        """
        :param t_as_func_of_s: t(s)
        """
        return Curve(Simplify(self.x.subs(self.t, t_as_func_of_s)), s)

    def arc_length_parameterize(self, s_name):
        s_t = self.arc()
        t_s = solve(s_t - s_name, self.t)  # derive the inverse function of the arc length
        return self.parameterize(t_s[0], s_name)

    def rotate(self, option: str, angle) -> Curve:
        """
        :params option: string, 'x', 'y' or 'z'.
        """
        option = option.lower()
        if option == 'x':
            mat = RotationMatrixX(angle)
        elif option == 'y':
            mat = RotationMatrixY(angle)
        elif option == 'z':
            mat = RotationMatrixZ(angle)
        else:
            raise ValueError
        return Curve(Simplify(mat @ self.x), self.t)

    def tangent_surface(self, u, v):
        return RuledSurface(self.x.subs(self.t, u), self.tg.subs(self.t, u), u, v)


class Surface:
    def __init__(self, func_array, *variables):
        """
        :param func_array: [A1(u,v), A2(u,v), A3(u,v)]
        :param variable:  u,v for example
        """
        self.x = Matrix(func_array)
        self.u = variables[0]
        self.v = variables[1]
        self.xu = Simplify(diff(self.x, self.u))
        self.xv = Simplify(diff(self.x, self.v))
        self.n = normalize(self.xu.cross(self.xv))
        self.I = self.get_first_fundamental()
        self.II = self.get_second_fundamental()
        self.g = Simplify(det2(self.I))
        self.K = Simplify(det2(self.II) / det2(self.I))  # gaussian curvature
        self.lowerG = np.array(self.I)  # the metric tensor is implemented with numpy!
        self.lowerGamma = Simplify(self.getChristoffel())
        self.has_principal = False
        self.has_weingarten = False

    def get_weingarten(self):
        """
        Although simplify can be skipped, inverse matrix may be a bottleneck.
        So calculate inverse matrix only if you need it.
        """
        if not self.has_weingarten:
            g_inv = Simplify(inv2(self.I))
            self.weingarten = self.II @ g_inv
            self.H = Simplify(trace(self.weingarten) / 2)  # mean curvature
            self.upperG = np.array(g_inv)
            self.Gamma = Simplify(np.einsum('kl,lij->kij', self.upperG, self.lowerGamma))

    def principal(self):
        """
        Get principal curvatures and directions.
        This process is too slow, so call it only when you trust the answer exists.
        """
        if not self.has_principal:
            if not self.has_weingarten:
                self.get_weingarten()
            self.principalCurvature, self.principalDirection = self.get_eigen_system()
            self.k1, self.k2 = self.principalCurvature
            self.vk1, self.vk2 = self.principalDirection
            self.has_principal = True

    def parameterize(self, uv_prime: list, new_u, new_v):
        """
        :param uv_prime: substitute u by u_prime(s,t), v by v_prime(s,t), where new_u=s, new_v=t
        """
        return Surface(
            Simplify(self.x.subs(self.u, uv_prime[0]).subs(self.v, uv_prime[1])),
            new_u, new_v
        )

    def Xu(self, u, v):
        return self.xu.subs(self.u, u).subs(self.v, v)

    def Xv(self, u, v):
        return self.xv.subs(self.u, u).subs(self.v, v)

    def tangentMatrixFunction(self):
        mat = []
        for term in self.xu:
            mat.append(lambdify([self.u, self.v], term))
        for term in self.xv:
            mat.append(lambdify([self.u, self.v], term))

        def func(u, v):
            lst = [f(u, v) for f in mat]
            shape = lst[0].shape
            res = np.stack(lst, np.newaxis)  # the stack is not safe
            return res.reshape((2, 3, *shape))

        return func

    def u_curve(self, u0):
        return self.x.subs(self.u, u0)

    def v_curve(self, v0):
        return self.x.subs(self.v, v0)

    def on_curve(self, u_t, v_t):
        return Simplify(self.x.subs(self.u, u_t).subs(self.v, v_t))

    def addConfinedCurve(self, u_t, v_t, t):
        return ConfinedCurve(self, u_t, v_t, t)

    def on_curve_u_by_v(self, u_v, v=None):
        if v is None:
            return Simplify(self.x.subs(self.u, u_v))
        else:
            return Simplify(self.x.subs(self.u, u_v)).subs(self.v, v)

    def on_curve_v_by_u(self, v_u, u=None):
        if u is None:
            return Simplify(self.x.subs(self.v, v_u))
        else:
            return Simplify(self.x.subs(self.v, v_u)).subs(self.u, u)

    def get_first_fundamental(self):
        lst = [self.xu.dot(self.xu), self.xu.dot(self.xv), self.xv.dot(self.xu), self.xv.dot(self.xv)]
        return Simplify(Matrix(lst).reshape(2, 2))

    def get_second_fundamental(self):
        lst = [diff(self.x, self.u, self.u), diff(self.x, self.u, self.v),
               diff(self.x, self.v, self.u), diff(self.x, self.v, self.v)]
        lst = [Simplify(y.dot(self.n)) for y in lst]
        return Matrix(lst).reshape(2, 2)

    def get_eigen_system(self):
        eigsys = eigsys2(self.weingarten)
        eig = eigsys[0]
        vecs = eigsys[1]
        vec = [vecs[:, 0], vecs[:, 1]]
        return eig, vec

    def getChristoffel(self):
        r"""
        :return: Christoffel symbol of \Gamma^k_{ij}, and the index sequence is (k,i,j)
        """
        lst = [diff(self.x, self.u, self.u), diff(self.x, self.u, self.v),
               diff(self.x, self.v, self.u), diff(self.x, self.v, self.v)]
        ri = [diff(self.x, self.u), diff(self.x, self.v)]
        ij_k1 = np.array([Simplify(y.dot(ri[0])) for y in lst]).reshape((2, 2))
        ij_k2 = np.array([Simplify(y.dot(ri[1])) for y in lst]).reshape((2, 2))
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
            du_dv_of_k1 = Simplify(b / (d - a))
            du_dv_of_k2 = Simplify(b / (d - a))
        else:
            a_d = Simplify(a - d)
            delta = Simplify(a_d ** 2 + 4 * b * c)
            # note that sqrSimplify is regardless of + and - !
            du_dv_of_k1 = Simplify(a_d / (2 * c) + sqrSimplify(sqrt(delta) / (2 * c)))
            du_dv_of_k2 = Simplify(a_d / (2 * c) - sqrSimplify(sqrt(delta) / (2 * c)))
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
            return [Simplify(-N / (2 * M))]
        sqr_det = sqrSimplify(det(self.II))
        du_dv_1 = Simplify((-M + sqr_det) / L)
        du_dv_2 = Simplify((-M - sqr_det) / L)
        return [du_dv_1, du_dv_2]

    def covariantDerivative(self, vector_field, sgn):
        """
        :param vector_field: list or np.ndarray (not sy.Matrix). Make sure that it has correct u, v for variables.
        :param sgn: 1 for contravariant vectors (upper index), and -1 for covariant vectors (lower index).
        :return: 2x2 Matrix
        """
        f = vector_field
        ordinary_derivative = []
        for fj in f:
            ordinary_derivative.append([diff(fj, self.u), diff(fj, self.v)])
        self.get_weingarten()
        appendix = np.einsum('kij,k->ij', self.Gamma, np.asarray(f))
        return Simplify(Matrix(ordinary_derivative) + sgn * Matrix(appendix))

    def getKillingODEs(self):
        killing_components_u = symbols(fr'\zeta_{str(self.u)}', cls=Function)(self.u, self.v)
        killing_components_v = symbols(fr'\zeta_{str(self.v)}', cls=Function)(self.u, self.v)
        killing_vector = [killing_components_u, killing_components_v]
        mat = self.covariantDerivative(killing_vector, sgn=-1)
        res = Simplify(mat + mat.T)
        # extract upper triangular components
        lst = [res[0, 0], res[1, 1], res[0, 1]]
        return lst

    def getOrthogonalFrame(self):
        """
        return a unit orthogonal frame, whose z-axis is the normal vector,
        x-axis is the normalized Xu vector.
        """
        z_vec = self.n
        x_vec = normalize(self.xu)
        y_vec = Simplify(z_vec.cross(x_vec))
        return Matrix([x_vec.T, y_vec.T, z_vec.T]).T


class RuledSurface(Surface):
    def __init__(self, directrix, generatrix, u, v, complicated=True):
        """
        :param directrix: a(u), also called director curve, is the movement of the generatrices.
        :param generatrix: b(u), the direction vectors of straight lines called generatrices.
        :param u: the expression of the ruled surface is: r(u,v) = a(u) + v * b(u)
        """
        self.a_u = Matrix(directrix)
        self.b_u = Matrix(generatrix)
        self.x = self.a_u + v * self.b_u
        self.u = u
        self.v = v
        if not complicated:
            super(RuledSurface, self).__init__(self.x, self.u, self.v)

    def director_curve(self, t) -> Curve:
        return Curve(self.a_u.subs(self.u, t), t)


class ConfinedCurve:
    def __init__(self, surface: Surface, u_t, v_t, t):
        self.u_t = u_t
        self.v_t = v_t
        self.t = t
        self.surface = surface
        self.x = self.surface.on_curve(self.u_t, self.v_t)
        self.dudt = diff(self.u_t, self.t)
        self.dvdt = diff(self.v_t, self.t)
        self.V = Simplify(self.surface.Xu(u_t, v_t) * self.dudt + self.surface.Xv(u_t, v_t) * self.dvdt)  # velocity
        self.T = normalize(self.V)  # the tangent vector of the frame is the tangent vector of the curve
        self.N = self.surface.n.subs(self.surface.u, u_t).subs(self.surface.v, v_t)
        # the normal vector of the frame is the normal vector of the surface
        self.B = Simplify(self.T.cross(self.N))

    def toCurve(self) -> Curve:
        return Curve(self.x, self.t)

    def toRuledSurface(self, u, v) -> RuledSurface:
        return RuledSurface(self.x.subs(self.t, u), self.B.subs(self.t, u), u, v)
