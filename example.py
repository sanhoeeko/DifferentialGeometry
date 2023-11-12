from sympy.abc import *

from dg import *
from visualize import *

setSimplify('positive')
sphere = Surface([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], theta, phi)
curve = ConfinedCurve(sphere, t, t, t)
sur = curve.toRuledSurface(var('u'), var('v'))
plt = plotSurface(sphere, (-4, 4), (-4, 4), style='background')
plt = plotCurve(curve, (-4, 4), plotter=plt, color='purple')
plt = plotSurface(sur, (-4, 4), (-0.5, 0.5), plotter=plt, style='background', color='green')
plotUVCurve(sur, (-2, 2), (-0.5, 0.5), sample=9, plotter=plt)
plt.show()
