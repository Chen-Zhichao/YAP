"""CPU-only complementary equilibrium audit with both cell faces fixed."""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from types import SimpleNamespace

import build_prepared_operators as storage

HELPER_SHA = "b71b24e9eea8a4564fb14a0a052085deb0ec7fd82f33765cc654a76cb15ae626"
SAMPLE_SETS = (201, 361)


def extract_helpers(path, namespace):
    """Reuse only reviewed generic routines, without BI experiment imports."""
    if storage.digest(path) != HELPER_SHA or Path(path).stat().st_size > 128 * 1024:
        raise ValueError("full-nodal helper source differs")
    tree = ast.parse(Path(path).read_text())
    names = {"connectivity", "periodic_free_map", "assemble_sparse", "inelastic_load"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names or len(nodes) != len(names):
        raise ValueError("reviewed full-nodal helper functions absent")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{k: namespace[k] for k in names})


def both_fixed_faces(p, xyz):
    import numpy as np
    from scipy.sparse import diags
    faces = np.flatnonzero((xyz[:, 1] == xyz[:, 1].min()) | (xyz[:, 1] == xyz[:, 1].max()))
    rows = (3 * faces[:, None] + np.arange(3)).ravel()
    keep = np.flatnonzero(np.asarray(p[rows].getnnz(axis=0)).ravel() == 0)
    p = p[:, keep].tocsr()
    counts = np.asarray(p.sum(axis=0)).ravel()
    if not len(counts) or np.any(counts <= 0) or p[rows].nnz:
        raise ValueError("both-face constrained map invalid")
    gram = (p.T @ p).tocsr()
    if (gram - diags(counts)).nnz:
        raise ValueError("periodic map columns are not independent")
    return p, counts


def constraint_map_audit(p, ids, deck):
    import numpy as np
    lookup, dofs = {int(v): i for i, v in enumerate(ids)}, {"UX": 0, "UY": 1, "UZ": 2}
    count, equation_ids = 0, set()
    for row in csv.reader(deck.split("/SOLU", 1)[0].splitlines()):
        if not row:
            continue
        if row[0] == "CP":
            raise ValueError("unexpected coupled target DOFs")
        if row[0] == "D":
            if len(row) != 4 or row[2] not in dofs or float(row[3]) != 0:
                raise ValueError("unexpected fixed target DOF")
            node = int(row[1])
            if node in lookup and p[3*lookup[node] + dofs[row[2]]].nnz:
                raise ValueError("free map violates fixed physical DOF")
            if node not in lookup and not len(ids) < node <= len(ids)+28:
                raise ValueError("unknown fixed target pilot")
        if row[0] != "CE":
            continue
        if len(row) not in (9, 12) or float(row[2]) != 0 or int(row[1]) in equation_ids:
            raise ValueError("unrecognized target constraint")
        equation_ids.add(int(row[1]))
        residual = None
        for i in range(3, len(row), 3):
            node, component, coefficient = int(row[i]), row[i+1], float(row[i+2])
            if component not in dofs or not np.isfinite(coefficient):
                raise ValueError("invalid target constraint component")
            if node in lookup:
                term = coefficient * p[3 * lookup[node] + dofs[component]]
                residual = term if residual is None else residual + term
            elif not len(ids) < node <= len(ids) + 28:
                raise ValueError("unknown target pilot")
        if residual is not None and np.max(abs(residual.data), initial=0.) > 1e-12:
            raise ValueError("periodic free map violates actual target constraint")
        count += 1
    if count != 5250:
        raise ValueError("unexpected actual target constraint count")
    return count


def complement(k, factor, z, residual):
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve
    delta = factor.solve(-residual)
    kz = k @ z
    gram = z.T @ kz
    diagonal = np.diag(gram)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError("invalid retained-space energy")
    scale = 1 / np.sqrt(diagonal)
    balanced = gram * np.outer(scale, scale)
    balanced = .5 * (balanced + balanced.T)
    eigen = np.linalg.eigvalsh(balanced)
    if eigen[0] <= 1e-11 * eigen[-1]:
        raise ValueError("unresolved retained-space energy conditioning")
    alpha = scale * cho_solve(cho_factor(balanced, lower=True), scale * (kz.T @ delta))
    missing = delta - z @ alpha
    energy = .5 * float(missing @ (k @ missing))
    total = .5 * float(delta @ (k @ delta))
    backward = float(np.linalg.norm(k @ delta + residual) / max(np.linalg.norm(residual), 1e-20))
    orthogonal = float(np.linalg.norm(scale * (kz.T @ missing)) / max(np.sqrt(max(2 * total, 0.)), 1e-20))
    if (not all(np.isfinite(v).all() for v in (delta, missing, eigen))
            or not np.isfinite([energy, total, backward, orthogonal]).all()
            or min(energy, total) < -1e-20 or backward > 1e-8 or orthogonal > 1e-7):
        raise ValueError("full-nodal solve/energy projection failed")
    return missing, dict(missing_energy_Nmm=energy, total_relaxation_energy_Nmm=total,
        missing_fraction=energy / total if total > 1e-20 else 0.,
        linear_solve_relative_residual=backward, energy_orthogonality=orthogonal,
        parent_balanced_condition=float(eigen[-1] / eigen[0]))


class FrozenResidualAudit:
    def __init__(self, work, source, ids, xyz, fields, materials, dataset, operator, deck):
        import numpy as np
        from scipy import sparse
        from exp12_cu_material_point import isotropic_stiffness
        from exp12_material_separable_reference import structured_node_id
        from exp12_solid185_bbar import bbar_points
        self.np, self.work, self.ds, self.op = np, work, dataset, operator
        self.source = Path(source) / "exp12_repeated_msi_bi_riesz_candidate.py"
        self.api = extract_helpers(self.source, dict(np=np, sparse=sparse,
            depth=SimpleNamespace(isotropic_stiffness=isotropic_stiffness),
            structured_node_id=structured_node_id, bbar_points=bbar_points))
        self.ids, self.xyz, self.phi = ids, xyz, fields.reshape(len(ids)*3, -1)
        self.cells, self.mids, self.dofs, self.points = self.api.connectivity(ids, xyz, materials, 45, 20, 45)
        p, _ = self.api.periodic_free_map(xyz)
        self.p, counts = both_fixed_faces(p, xyz)
        self.constraint_count = constraint_map_audit(self.p, ids, deck)
        old = self.phi[:, operator.macro_size:]
        self.z = (self.p.T @ old) / counts[:, None]
        if np.max(abs(self.p @ self.z - old)) > 1e-9:
            raise ValueError("old full bubble span is not in target constrained space")
        self.selected = np.flatnonzero(self.mids == 4)
        if len(self.selected) * 8 != dataset.selected_cu_point_count:
            raise ValueError("full-nodal Cu point count differs")
        self.system, self.samples, self.directions = None, [], []

    def sample(self, result_set, temperature, full, plastic, creep, reduced_eigen, reduced_response):
        if result_set not in SAMPLE_SETS:
            return
        np, reference = self.np, self.ds.material_reference()
        print("full-nodal assemble", result_set, temperature, flush=True)
        names = {3: "si", 4: "cu"}
        k, thermal = self.api.assemble_sparse(reference, self.mids, self.dofs, self.points,
            names, temperature, len(self.ids))
        scales = np.r_[self.ds.macro_coordinate_scale, self.ds.bubble_coordinate_scale]
        expected = self.op.reference_stiffness(temperature)
        projected = self.phi.T @ (k @ self.phi)
        k_error = np.linalg.norm((projected-expected) * np.outer(scales, scales)) / max(np.linalg.norm(expected * np.outer(scales, scales)), 1e-20)
        expected_thermal = self.op.reference_thermal_load(temperature)
        f_error = np.linalg.norm(scales * (self.phi.T @ thermal-expected_thermal)) / max(np.linalg.norm(scales * expected_thermal), 1e-20)
        if not np.isfinite([k_error, f_error]).all() or max(k_error, f_error) > 2e-8:
            raise ValueError("full-nodal material/thermal reference parity failed")
        eigen = self.api.inelastic_load(reference, plastic, creep, self.dofs, self.points,
            self.selected, temperature, len(self.ids))
        if np.max(abs(self.phi.T @ eigen-reduced_eigen)) > 1e-8:
            raise ValueError("full-nodal Cu eigenstrain order/sign differs")
        residual = k @ (self.phi @ full) - thermal - eigen
        projected_residual = self.phi.T @ residual
        if not np.isfinite(projected_residual).all() or np.max(abs(projected_residual-reduced_response)) > 1e-8:
            raise ValueError("full-nodal response reconstruction differs")
        kfree = (self.p.T @ k @ self.p).tocsc()
        from scipy.sparse.linalg import splu
        prior = self.system
        same = prior is not None and (prior[0].shape == kfree.shape and np.array_equal(prior[0].indptr, kfree.indptr)
            and np.array_equal(prior[0].indices, kfree.indices) and np.array_equal(prior[0].data, kfree.data))
        factor = prior[1] if same else splu(kfree, permc_spec="MMD_AT_PLUS_A")
        self.system = kfree, factor
        rfree = self.p.T @ residual
        missing, energy = complement(kfree, factor, self.z, rfree)
        self.directions.append(np.asarray(self.p @ missing).reshape(len(self.ids), 3))
        self.samples.append(dict(result_set=result_set, temperature_c=temperature,
            stiffness_recovery_error=float(k_error), thermal_recovery_error=float(f_error),
            retained_bubble_equilibrium_maximum=float(np.max(abs(projected_residual[self.op.macro_size:]))),
            complementary_residual_l2_N=float(np.linalg.norm(rfree)), reused_exact_factor=same, **energy))
        storage.write_json(self.work / "FULL_NODAL_PROGRESS.json", dict(samples=self.samples))

    def finish(self):
        np = self.np
        if tuple(s["result_set"] for s in self.samples) != SAMPLE_SETS:
            raise ValueError("predeclared full-nodal snapshots missing")
        raw = np.stack(self.directions, axis=2)
        singular = np.linalg.svd(raw.reshape(-1, len(SAMPLE_SETS)), compute_uv=False)
        rank = int(np.sum(singular > max(1e-14, 1e-10 * singular[0])))
        np.savez_compressed(self.work / "MISSING_FROZEN_DIRECTIONS.npz", node_ids=self.ids,
            coordinates_mm=self.xyz, raw_directions_mm=raw, singular_values_mm=singular,
            rank=np.asarray(rank))
        storage.write_json(self.work / "FULL_NODAL_COMPLETE.json", dict(
            state="FROZEN_COMPLEMENT_DIAGNOSTIC_NOT_OPERATOR_PASS", samples=self.samples,
            measured_rank=rank, rank_tolerance=dict(relative=1e-10, absolute_mm=1e-14),
            helper_sha256=storage.digest(self.source), constraints=self.constraint_count,
            free_dofs=self.p.shape[1], no_teacher_used_to_construct_directions=True,
            no_constitutive_history_updates=True, no_native_solver=True,
            directions_sha256=storage.digest(self.work / "MISSING_FROZEN_DIRECTIONS.npz")), immutable=True)
