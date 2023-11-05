import math as mh

import numpy as np
import pyvista as pv
from sympy import lambdify

import dg


def arange(start, end, step):
    return np.arange(start, end + step, step)


def pvPlotTube(points):
    spline = pv.Spline(points, 1000)
    spline.plot(
        render_lines_as_tubes=True,
        line_width=10,
    )


def _plotCurve(xfunc, yfunc, zfunc, t_range, eps):
    """
    :param xfunc, yfunc, zfunc: Callable for vector
    :param t_range: list, like [t_start, t_end]
    """
    ts = arange(*t_range, eps)
    xs = xfunc(ts)
    ys = yfunc(ts)
    zs = zfunc(ts)
    points = np.vstack((xs, ys, zs)).T
    pvPlotTube(points)


def _plotSurface(xfunc, yfunc, zfunc, u_range, v_range, eps, plotter=None):
    """
    :param xfunc, yfunc, zfunc: Callable f(u,v) for vector
    :param u_range, v_range: list, like [u_start, u_end]
    """
    if plotter is None:
        plotter = pv.Plotter()
    us = arange(*u_range, eps)
    vs = arange(*v_range, eps)
    U, V = np.meshgrid(us, vs)
    xs = xfunc(U, V)
    ys = yfunc(U, V)
    zs = zfunc(U, V)
    grid = pv.StructuredGrid(xs, ys, zs)
    plotter.add_mesh(grid)
    return plotter


def _plotSurfaceColorfully(xfunc, yfunc, zfunc, color_func, u_range, v_range, eps, color_meaning, plotter=None):
    """
    :param xfunc, yfunc, zfunc: Callable f(u,v) for vector
    :param u_range, v_range: list, like [u_start, u_end]
    """
    if plotter is None:
        plotter = pv.Plotter()
    us = arange(*u_range, eps)
    vs = arange(*v_range, eps)
    U, V = np.meshgrid(us, vs)
    xs = xfunc(U, V)
    ys = yfunc(U, V)
    zs = zfunc(U, V)
    # generate color
    color = color_func(U, V)
    # if the value is unfortunately a constant
    if not hasattr(color, 'shape'):
        color = np.ones(U.shape) * color
    color = color.reshape(-1)
    grid = pv.StructuredGrid(xs, ys, zs)
    # draw
    grid[color_meaning] = color
    plotter.add_mesh(grid, scalars=color_meaning, cmap="jet")
    return plotter


def plotCurve(curve: dg.Curve, t_range=(-1, 1), eps=None, params: dict = None):
    if eps is None:
        eps = (t_range[1] - t_range[0]) / 100
    # substitute parameters
    lst = list(curve.x)
    if params is not None:
        for k in params.keys():
            for i in range(len(lst)):
                lst[i] = lst[i].subs(k, params[k])
    # convert expressions to callables
    xf, yf, zf = [lambdify(curve.t, f) for f in lst]
    # draw
    try:
        _plotCurve(xf, yf, zf, t_range, eps)
    except TypeError:
        print("plotCurve Error: You may forget to eliminate all free parameters in your expression.")


def plotSurface(sur: dg.Surface, u_range=(-1, 1), v_range=(-1, 1), eps=None, params: dict = None, style: str = None):
    """
    :param params: substitute free parameters
    :param style: None|gauss(show gaussian curvature)
    """
    if eps is None:
        ulen = u_range[1] - u_range[0]
        vlen = v_range[1] - v_range[0]
        eps = mh.sqrt(ulen * vlen / 6400)
    # substitute parameters
    lst = list(sur.x)
    if params is not None:
        for k in params.keys():
            for i in range(len(lst)):
                lst[i] = lst[i].subs(k, params[k])
    # convert expressions to callables
    xf, yf, zf = [lambdify([sur.u, sur.v], f, modules='numpy') for f in lst]
    # draw
    try:
        if style is None:
            return _plotSurface(xf, yf, zf, u_range, v_range, eps)
        else:
            if style == 'gauss':
                expr = sur.K
                if params is not None:
                    for k in params.keys():
                        expr.subs(k, params[k])
                cf = lambdify([sur.u, sur.v], expr)
            return _plotSurfaceColorfully(xf, yf, zf, cf, u_range, v_range, eps, 'gaussian curvature')
    except TypeError:
        print("plotSurface Error: You may forget to eliminate all free parameters in your expression.")


def runge_kutta(f, x0, y0, h, n):
    # x0 and y0 are the initial values of x and y
    # h is the step size
    # n is the number of steps
    # Initialize the arrays to store the values of x and y
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    # Assign the initial values to the first elements of the arrays
    x[0] = x0
    y[0] = y0
    # Loop over the steps
    for i in range(n):
        # Calculate the intermediate values of k1, k2, k3, and k4
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(x[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(x[i] + h, y[i] + h * k3)
        # Update the values of x and y using the formula
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # Return the arrays of x and y
    return x, y


def integral_curve(diff_func, x0, y0, t_range):
    """
    t_range must include 0 because this is where the initial point (x0, y0) is defined.
    """
    tot_step = 100
    eps = (t_range[1] - t_range[0]) / tot_step
    forward_step = round(t_range[1] / eps)
    backward_step = round(-t_range[0] / eps)
    fw_x, fw_y = runge_kutta(diff_func, x0, y0, eps, forward_step)
    bw_x, bw_y = runge_kutta(diff_func, x0, y0, -eps, backward_step)
    x = np.hstack((bw_x[::-1], fw_x))
    y = np.hstack((bw_y[::-1], fw_y))
    return x, y


def _plotUVGrid(sur: dg.Surface, u_or_v, u_range, v_range, sample=5, plotter=None, color_str='black'):
    """
    :param u_or_v: String, only 'u' or 'v'
    """
    u0s = np.linspace(*u_range, sample)
    v0s = np.linspace(*v_range, sample)
    lines = []
    if plotter is None:
        plotter = pv.Plotter()
    for u0 in u0s:
        for v0 in v0s:
            if u_or_v == 'u':
                X = lambdify(sur.v, sur.u_curve(u0))
                ts = np.linspace(*u_range, 100)
            elif u_or_v == 'v':
                X = lambdify(sur.u, sur.v_curve(v0))
                ts = np.linspace(*u_range, 100)
            else:
                raise ValueError
            pos = np.zeros((len(ts), 3))
            for i in range(len(ts)):
                pos[i, :] = X(ts[i]).reshape(-1)
            lines.append(pos)
    for points in lines:
        spline = pv.Spline(points, 200)
        plotter.add_mesh(spline, color=color_str, line_width=4)
    return plotter


def integralPlotGrid(sur: dg.Surface, diff_func, u_range, v_range, sample=5, plotter=None, color_str='black'):
    """
    :param diff_func: Sympy expression, f in dv/du = f(u,v)
    """
    u0s = np.linspace(*u_range, sample)
    v0s = np.linspace(*v_range, sample)
    X = lambdify([sur.u, sur.v], sur.x)
    diff_func = lambdify([sur.u, sur.v], diff_func)
    lines = []
    if plotter is None:
        plotter = pv.Plotter()
    for u0 in u0s:
        for v0 in v0s:
            u, v = integral_curve(diff_func, u0, v0, [-1, 1])
            pos = X(u, v)
            lines.append(pos.reshape(3, -1).T)
    for points in lines:
        spline = pv.Spline(points, 200)
        plotter.add_mesh(spline, color=color_str, line_width=4)
    return plotter


def plotLineOfCurvature(sur: dg.Surface, u_range, v_range, sample=5):
    def stretch(interval: list, rate):
        x = np.array(interval)
        center = (x[0] + x[1]) / 2
        return center + (x - center) * rate

    f1, f2 = sur.getLineOfCurvatureODE()
    plotter = plotSurface(sur, stretch(u_range, 2), stretch(v_range, 2))
    plotter = integralPlotGrid(sur, f1, u_range, v_range, sample, color_str='red', plotter=plotter)
    plotter = integralPlotGrid(sur, f2, u_range, v_range, sample, color_str='blue', plotter=plotter)
    return plotter


def plotAsymptote(sur: dg.Surface, u_range, v_range, sample=5):
    def stretch(interval: list, rate):
        x = np.array(interval)
        center = (x[0] + x[1]) / 2
        return center + (x - center) * rate

    plotter = plotSurface(sur, stretch(u_range, 2), stretch(v_range, 2))
    lst = sur.getAsymptoteODE()

    if len(lst) == 2:
        f1, f2 = lst
        plotter = integralPlotGrid(sur, f1, u_range, v_range, sample, color_str='red', plotter=plotter)
        plotter = integralPlotGrid(sur, f2, u_range, v_range, sample, color_str='blue', plotter=plotter)
    elif len(lst) == 1:
        plotter = integralPlotGrid(sur, lst[0], u_range, v_range, sample, color_str='red', plotter=plotter)
        plotter = _plotUVGrid(sur, 'u', stretch(u_range, 2), stretch(v_range, 2), sample,
                              color_str='blue', plotter=plotter)
    return plotter


def plotUVCurve(sur: dg.Surface, u_range, v_range, sample=5):
    plotter = plotSurface(sur, u_range, v_range)
    plotter = _plotUVGrid(sur, 'u', u_range, v_range, sample, color_str='red', plotter=plotter)
    plotter = _plotUVGrid(sur, 'v', u_range, v_range, sample, color_str='blue', plotter=plotter)
    return plotter
