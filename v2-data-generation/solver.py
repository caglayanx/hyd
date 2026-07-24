"""FEniCSx reference solver for the V2 coupled mech-hydrogen simulator.

Solves the transient stress-assisted hydrogen diffusion PDE on a DOLFINx mesh
of a notched specimen containing a single curved dislocation:

    J        = -D grad(C_L) + (D C_L V_H / (R T)) grad(sigma_h)
    dC_L/dt  = -div(J)

with the Dirichlet boundary condition ``C_L = C_s(P, T)`` applied on the
explicitly tagged notch surface only (NOT the whole outer boundary).

Time integration uses a theta-scheme with a LUMPED mass matrix (row-sum
lumping) to suppress the negative-concentration ringing that a consistent
mass matrix produces at Dirichlet boundaries:

    A c^{n+1} = b ,   A = (1/dt) diag(M_lump) + theta * K_stiff ,
    b = (1/dt) diag(M_lump) c^n - (1 - theta) K_stiff c^n ,

with Dirichlet rows overwritten as identity and RHS set to ``C_s``. Because the
stress-assisted drift term ``+ (D C_L V_H / (R T)) grad(sigma_h)`` makes the
bilinear form non-symmetric, the linear system is solved with GMRES.

The solver also exports the exact nodal gradient of the hydrostatic stress
(``grad_sigma``, shape ``(n_dof, 2)``) and a boolean notch-DOF mask
(``bc_mask``, shape ``(n_dof,)``) for V3 PINN training.

DOLFINx / UFL / PETSc are imported lazily so this module imports cleanly without
the FEniCSx stack. The thermodynamics come from :mod:`hydrogen_thermo`.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import constants as C
import hydrogen_thermo as ht


def _import_dolfinx():
    """Lazily import the DOLFINx v0.11 native stack.

    Returns ``(dolfinx, ufl, PETSc, mpi)``. Raises ImportError with a helpful
    message if the FEniCSx stack is not installed.
    """
    try:
        import dolfinx  # type: ignore[import-not-found]
        import dolfinx.fem  # type: ignore[import-not-found]
        import dolfinx.fem.petsc  # type: ignore[import-not-found]
        import ufl  # type: ignore[import-not-found]
        from petsc4py import PETSc  # type: ignore[import-not-found]
        from dolfinx import mpi  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only with the stack
        raise ImportError(
            "DOLFINx v0.11 is required for HydrogenDiffusionSolver. Install it "
            "via conda (`conda install -c conda-forge fenics-dolfinx`) or apt."
        ) from exc
    return dolfinx, ufl, PETSc, mpi


class HydrogenDiffusionSolver:
    """Transient stress-assisted hydrogen diffusion solver (DOLFINx v0.11).

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Mesh of the notched specimen (2D, plane strain).
    facet_tags : dolfinx.mesh.MeshTags
        Boundary facet tags identifying the notch surface physical group.
    notch_id : int
        Facet tag value identifying the notch surface Dirichlet facets.
    sigma_h : dolfinx.fem.Function
        Prescribed hydrostatic stress field (Pa) from the Mura-based solution.
    P, T : float
        Operating pressure (Pa) and temperature (K); used to evaluate the
        dynamic Dirichlet value ``C_s(P, T)``.
    D : float, optional
        Lattice diffusion coefficient (m^2/s), default ``constants.D_L``.
    V_H : float, optional
        Partial molar volume of hydrogen (m^3/mol), default ``constants.V_H``.
    theta : float, optional
        Theta-scheme parameter (0=explicit, 1=implicit Euler, 0.5=Crank-Nicolson).
    """

    def __init__(self, domain, facet_tags, notch_id: int, sigma_h,
                 P: float, T: float, *, D: float = C.D_L, V_H: float = C.V_H,
                 theta: float = 1.0):
        dolfinx, ufl, PETSc, _mpi = _import_dolfinx()
        self._dolfinx = dolfinx
        self._ufl = ufl
        self._PETSc = PETSc

        self.domain = domain
        self.facet_tags = facet_tags
        self.notch_id = int(notch_id)
        self.sigma_h = sigma_h
        self.P = float(P)
        self.T = float(T)
        self.D = float(D)
        self.V_H = float(V_H)
        self.theta = float(theta)

        # Scalar P1 Lagrange space for the lattice concentration C_L.
        self.V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # Current / previous concentration fields.
        self.c_curr = dolfinx.fem.Function(self.V)
        self.c_prev = dolfinx.fem.Function(self.V)

        # Dynamic Dirichlet value C_s(P, T) as a Constant (updated on demand).
        self._c_s_value = float(ht.surface_concentration(self.T, self.P))
        self.cs_const = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(self._c_s_value))

        # Locate notch-surface DOFs (notch facets only) and build the BC.
        self.bc_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, facet_tags.dim, facet_tags.find(self.notch_id))
        self.bc = dolfinx.fem.dirichletbc(self.cs_const, self.bc_dofs)

        # Bilinear forms and assembled operators.
        self._build_forms()
        self._assemble_lumped_mass()
        self._setup_solver()

    # ------------------------------------------------------------------
    # Weak forms: mass M = integral u v dx, stiffness K from div(J).
    # ------------------------------------------------------------------
    def _build_forms(self):
        ufl = self._ufl
        D, V_H, T = self.D, self.V_H, self.T
        sigma_h = self.sigma_h

        # Mass form (consistent, lumped afterwards).
        self._m_form = self._dolfinx.fem.form(ufl.inner(self.u, self.v) * ufl.dx)

        # Spatial operator S(C, v) = integral D grad C . grad v dx
        #     - integral (D V_H / (R T)) C grad(sigma_h) . grad v dx
        # (from dC/dt = -div(J) -> M dC/dt + S C = 0). The drift term makes S
        # non-symmetric, hence GMRES.
        drift_coeff = D * V_H / (C.R * T)
        a_diff = D * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        a_drift = -drift_coeff * self.u * ufl.dot(ufl.grad(sigma_h), ufl.grad(self.v)) * ufl.dx
        self._k_form = self._dolfinx.fem.form(a_diff + a_drift)

        # Linear form for the RHS action K * c_prev (used via assemble_vector).
        c = self.c_prev
        a_diff_c = D * ufl.inner(ufl.grad(c), ufl.grad(self.v)) * ufl.dx
        a_drift_c = -drift_coeff * c * ufl.dot(ufl.grad(sigma_h), ufl.grad(self.v)) * ufl.dx
        self._k_action_form = self._dolfinx.fem.form(a_diff_c + a_drift_c)

    def _assemble_lumped_mass(self):
        """Row-sum lumped mass matrix -> a PETSc diagonal vector."""
        petsc = self._dolfinx.fem.petsc
        M = petsc.assemble_matrix(self._dolfinx.fem.form(self._m_form), bcs=[])
        M.assemble()
        n = M.size[0]
        # Row-sum lumping: diag_i = sum_j M_ij.
        ones = self._PETSc.Vec().createSeq(n)
        ones.set(1.0)
        ones.assemble()
        lumped = M.createVecLeft()
        M.mult(ones, lumped)
        ones.destroy()
        self._m_lump = lumped  # PETSc Vec holding the diagonal of M_lump
        self._n_dof = n

    def _setup_solver(self):
        """Configure a GMRES KSP for the non-symmetric system."""
        self._ksp = self._PETSc.KSP().create(self._PETSc.Comm.self)
        self._ksp.setType(self._PETSc.KSP.Type.GMRES)
        self._ksp.getPC().setType(self._PETSc.PC.Type.ILU)  # ILU preconditioner
        self._ksp.setTolerances(rtol=1e-10, atol=1e-30, max_it=500)
        self._ksp.setFromOptions()

    # ------------------------------------------------------------------
    # Dynamic state setters (re-evaluate C_s(P, T)).
    # ------------------------------------------------------------------
    def set_temperature(self, T: float) -> None:
        self.T = float(T)
        self._c_s_value = float(ht.surface_concentration(self.T, self.P))
        self.cs_const.value = self._dolfinx.default_scalar_type(self._c_s_value)
        # Drift coefficient depends on T -> rebuild forms.
        self._build_forms()

    def set_pressure(self, P: float) -> None:
        self.P = float(P)
        self._c_s_value = float(ht.surface_concentration(self.T, self.P))
        self.cs_const.value = self._dolfinx.default_scalar_type(self._c_s_value)

    @property
    def c_s(self) -> float:
        return self._c_s_value

    # ------------------------------------------------------------------
    # Time integration: theta-scheme with lumped mass + manual Dirichlet rows.
    # ------------------------------------------------------------------
    def _assemble_stiffness(self):
        """Assemble the (non-symmetric) stiffness matrix K (no BC handling)."""
        K = self._dolfinx.fem.petsc.assemble_matrix(
            self._dolfinx.fem.form(self._k_form), bcs=[])
        K.assemble()
        return K

    def _apply_dirichlet_rows(self, A, b):
        """Overwrite Dirichlet rows of A as identity and set b = C_s on those rows."""
        bc_dofs = np.asarray(self.bc_dofs, dtype=np.int32)
        if bc_dofs.size == 0:
            return
        # Zero the BC rows, set diagonal to 1.
        for dof in bc_dofs:
            A.zeroRows(dof, diag=1.0)
        b.array[bc_dofs] = self._c_s_value

    def step(self, dt: float):
        """Advance one theta-scheme step; return the new concentration Function."""
        petsc = self._dolfinx.fem.petsc
        theta, n = self.theta, self._n_dof

        K = self._assemble_stiffness()
        # System matrix A = (1/dt) diag(M_lump) + theta * K.
        A = K.copy()
        diag = A.getDiagonal()
        diag.array[:] = (1.0 / dt) * self._m_lump.array + theta * diag.array
        A.setDiagonal(diag)
        diag.destroy()

        # RHS b = (1/dt) M_lump c_prev - (1 - theta) K c_prev.
        k_c_prev = petsc.assemble_vector(self._k_action_form)  # K @ c_prev
        k_c_prev.assemble()
        b = self._PETSc.Vec().createSeq(n)
        b.array[:] = (1.0 / dt) * self._m_lump.array * self.c_prev.x.array - (1.0 - theta) * k_c_prev.array
        k_c_prev.destroy()

        # Apply Dirichlet rows (notch surface) -> identity rows, b = C_s.
        self._apply_dirichlet_rows(A, b)

        # Solve A c^{n+1} = b with GMRES.
        x = self._PETSc.Vec().createSeq(n)
        self._ksp.setOperators(A)
        self._ksp.solve(b, x)

        # Guard against negative concentrations (lumped mass should prevent this).
        x_arr = np.asarray(x.array, dtype=np.float64)
        if np.nanmin(x_arr) < -1e-12:
            raise RuntimeError(
                f"Negative c_L (min={float(np.nanmin(x_arr)):.3e}); lumped-mass "
                "formulation should prevent ringing."
            )

        self.c_curr.x.array[:] = x_arr
        self.c_prev.x.array[:] = x_arr

        A.destroy(); K.destroy(); b.destroy(); x.destroy()
        return self.c_curr

    def run(self, n_steps: int, dt: float):
        """Time-step loop; return (c_lattice_history, time_points).

        ``c_lattice_history`` has shape ``(n_steps, n_dof)``; ``time_points``
        has shape ``(n_steps,)``.
        """
        history = np.empty((int(n_steps), self._n_dof), dtype=np.float64)
        times = np.empty(int(n_steps), dtype=np.float64)
        for i in range(int(n_steps)):
            self.step(dt)
            history[i] = np.asarray(self.c_curr.x.array, dtype=np.float64)
            times[i] = (i + 1) * dt
        return history, times

    # ------------------------------------------------------------------
    # Field export for the Zarr v3 writer / V3 PINN.
    # ------------------------------------------------------------------
    def export_fields(self) -> Dict[str, np.ndarray]:
        """Return coords, bc_mask, hydrostatic_stress, grad_sigma as numpy arrays.

        - coords:            (n_dof, 2) float64
        - bc_mask:            (n_dof,)   bool
        - hydrostatic_stress: (n_dof,)   float64
        - grad_sigma:         (n_dof, 2) float64
        """
        dolfinx = self._dolfinx
        ufl = self._ufl

        coords3 = self.V.tabulate_dof_coordinates()
        coords = np.ascontiguousarray(coords3[:, :2], dtype=np.float64)

        bc_mask = np.zeros(self._n_dof, dtype=bool)
        bc_mask[np.asarray(self.bc_dofs, dtype=np.int32)] = True

        sigma_h_arr = np.asarray(self.sigma_h.x.array, dtype=np.float64)

        # Exact nodal gradient of sigma_h via L2 projection into a Vector P1 space.
        Vvec = dolfinx.fem.functionspace(self.domain, ("Lagrange", 1, (2,)))
        g = ufl.TrialFunction(Vvec)
        w = ufl.TestFunction(Vvec)
        a_proj = ufl.inner(g, w) * ufl.dx
        L_proj = ufl.dot(ufl.grad(self.sigma_h), w) * ufl.dx
        problem = dolfinx.fem.petsc.LinearProblem(
            a_proj, L_proj, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
        g_func = problem.solve()
        grad_sigma = np.ascontiguousarray(
            np.asarray(g_func.x.array, dtype=np.float64).reshape(-1, 2))

        return {
            "coords": coords,
            "bc_mask": bc_mask,
            "hydrostatic_stress": sigma_h_arr,
            "grad_sigma": grad_sigma,
        }


__all__ = ("HydrogenDiffusionSolver",)
