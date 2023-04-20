import os
import numpy as np
from qic import Qic
from scipy.optimize import minimize
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, LinkingNumber
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curverzfourier import CurveRZFourier
# Input parameters
r = 0.1 # radius (in meters) for the plasma boundary from the near-axis equilibrium
vmec_input='input.QI_nfp1' # name for vmec input from near-axis
nphi = 32 # toroidal resolution of the surface for the Biot-Savart evaluation
ntheta = 32 # poloidal resolution of the surface for the Biot-Savart evaluation
OUT_DIR = 'output' # output directory name
ncoils = 12 # number of coils per half field period
R1 = 4*r # Minor radius for the initial circular coils
order = 14 # Number of Fourier modes describing each Cartesian component of each coil
LENGTH_WEIGHT = 1e-2 # Weight on the curve lengths in the objective function
MAX_LENGTH_initial = 2.4 # Maximum length of each coil for first optimization
MAX_LENGTH_final = 4.0 # Maximum length of each coil for first optimization
MAXITER_initial = 350 # Number of iterations to perform with initial coil length
MAXITER_final   = 500 # Number of iterations to perform with final coil length
CC_THRESHOLD = 0.1 # Threshold and weight for the coil-to-coil distance penalty in the objective function
CC_WEIGHT = 1000
CURVATURE_THRESHOLD = 7 # Threshold and weight for the curvature penalty in the objective function
CURVATURE_WEIGHT = 1e-6
MSC_THRESHOLD = 7 # Threshold and weight for the mean squared curvature penalty in the objective function
MSC_WEIGHT = 1e-6
# Create near-axis equilibrium and output folder
stel=Qic.from_paper('QI Jorge')
os.makedirs(OUT_DIR, exist_ok=True)
# Generate plasma boundary
stel.to_vmec(os.path.join(OUT_DIR,vmec_input), r=r, ntorMax=50, params={'mpol':6, 'ntor': 16, 'ns_array': [51], 'ftol_array':[1e-14], "niter_array":[20000]})
s = SurfaceRZFourier.from_vmec_input(os.path.join(OUT_DIR,vmec_input), range="half period", nphi=nphi, ntheta=ntheta)
s.to_vtk(os.path.join(OUT_DIR,"surf_init"))
# Initialize coils following the axis
ma = CurveRZFourier((ncoils+1)*2*stel.nfp, len(stel.rc)-1, stel.nfp, True)
ma.rc[:] = stel.rc
ma.zs[:] = stel.zs[1:]
ma.x = ma.get_dofs()
gamma_curves = ma.gamma()
numquadpoints = 15 * order
base_curves = []
for i in range(ncoils):
    curve = CurveXYZFourier(numquadpoints, order)
    angle = (i+0.5)*(2*np.pi)/((2)*stel.nfp*ncoils)
    curve.set("xc(0)", gamma_curves[i+1,0])
    curve.set("xc(1)", np.cos(angle)*R1)
    curve.set("yc(0)", gamma_curves[i+1,1])
    curve.set("yc(1)", np.sin(angle)*R1)
    curve.set("zc(0)", gamma_curves[i+1,2])
    curve.set("zs(1)", R1)
    curve.x = curve.x  # need to do this to transfer data to C++
    base_curves.append(curve)
base_currents = [Current(1)*1e5 for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, os.path.join(OUT_DIR,"curves_init"))
# Define objective function
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
JF_base = (
      Jf
    + CC_WEIGHT * Jccdist
    + CURVATURE_WEIGHT * sum(Jcs)
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
    # + 1e2*LinkingNumber(base_curves)
    )
JF_initial = JF_base + LENGTH_WEIGHT * sum([QuadraticPenalty(Jls[i], MAX_LENGTH_initial) for i in range(len(base_curves))])
JF_final   = JF_base + LENGTH_WEIGHT * sum([QuadraticPenalty(Jls[i], MAX_LENGTH_final)   for i in range(len(base_curves))])
def fun(dofs, initial_or_final='initial'):
    if initial_or_final=='initial': JF = JF_initial
    else: JF = JF_final
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad
# Find coils with initial length
dofs = JF_base.x
res = minimize(fun, dofs, args=('initial'), jac=True, method='L-BFGS-B', options={'maxiter': MAXITER_initial, 'maxcor': 300}, tol=1e-15)
curves_to_vtk(curves, os.path.join(OUT_DIR,"curves_opt_short"))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(os.path.join(OUT_DIR,"surf_opt_short"), extra_data=pointData)
# Find coils with final length
dofs = JF_base.x
res = minimize(fun, dofs, args=('final'), jac=True, method='L-BFGS-B', options={'maxiter': MAXITER_final, 'maxcor': 300}, tol=1e-15)
curves_to_vtk(curves, os.path.join(OUT_DIR,"curves_opt_long"))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(os.path.join(OUT_DIR,"surf_opt_long"), extra_data=pointData)