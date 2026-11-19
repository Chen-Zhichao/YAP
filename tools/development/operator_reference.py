"""Owned-input native FE assembly and injected elastic responses; no file I/O.

This selects unchanged numerical definitions from the pinned nonuniform-brick
assembler. The owner must admit resources and authenticate geometry before
calling assemble_target. Output/cache bounds are not peak-memory forecasts.
The response connector requires an owner-supplied scaled-system factor factory.
Neither connector launches a backend or admits Cu history, bow or an operator.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from itertools import product
import math
from types import MappingProxyType, ModuleType, SimpleNamespace as NS

from tools import operator_geometry as height


SOURCE_PINS = MappingProxyType({
    "quadrature": "254b8d815953166afb51b49aefb8f5192e9619b810b88251647e2f1f685b2e85",
    "native": "c05decbf146226f470b25b3bb8b3d336023ab0ee0c5dcf3cd681ba28efa9419c",
    "bbar": "8aee0119f791b9fe113d4e3779b528431e20f0864d34794fdc4bab2f5edb4f6a",
    "material": "35e7016dab8b1fb2cb0a928e7642665d3fc4f00b4f255eb797fb6d6e44b56722",
    "fields": "37d7ca5f2317c130865307cc8527476e6f6bdf9127765c2a95d6503d211cd729",
})
QUADRATURE_FUNCTIONS = frozenset(("tensor_plan", "quadrature_cache", "physical_stiffness",
                                 "copper_maps", "assemble_native"))
BBAR_FUNCTIONS = frozenset(("shape_functions", "natural_shape_gradients", "physical_shape_gradients",
                          "engineering_b_matrix", "bbar_points", "element_linear_operators"))
LIMIT_KEYS = frozenset(("chunk_elements", "max_shape_groups", "cache_cap_bytes", "output_payload_cap_bytes"))
_TOKEN = object()
ELASTIC_REFERENCE_SHA = "9b8324b6771fd5b6f1bc1ad7ead74c32bcf0da13c1219700f76a487d2adcda4c"
ELASTIC_KEYS = frozenset(("kaa", "c", "fm", "ff", "macro_fields", "mechanical_q", "mechanical_u",
    "mechanical_reactions", "thermal_q", "thermal_u", "schur", "P_data", "P_indices", "P_indptr",
    "P_shape", "kff_data", "kff_indices", "kff_indptr", "kff_shape"))
_RESPONSE_TOKEN = object()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _definitions(raw, role, *, functions=(), classes=(), assignments=(), namespace=None):
    """Select exact original definitions; exclude imports and launch entry points."""
    functions, classes, assignments = map(set, (functions, classes, assignments))
    selected, found_functions, found_classes, found_assignments = [], [], [], []
    for node in ast.parse(raw).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
            found_functions.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            selected.append(node)
            found_classes.append(node.name)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in assignments):
            selected.append(node)
            found_assignments.append(node.targets[0].id)
    require(len(found_functions) == len(functions) and set(found_functions) == functions
            and len(found_classes) == len(classes) and set(found_classes) == classes
            and len(found_assignments) == len(assignments) and set(found_assignments) == assignments,
            "exact original FE definitions required: " + role)
    module = ModuleType("_operator_reference_" + role)
    module.__file__ = "<authenticated-FE-" + role + ">"
    vars(module).update(namespace or {})
    # No inherited future annotations: selected dataclasses use actual types and
    # need no registration in the process-global module namespace.
    exec(compile(ast.Module(body=selected, type_ignores=[]), module.__file__, "exec",
                 dont_inherit=True), vars(module))
    return module


def bind_sources(api, raw):
    require(type(raw) is dict and set(raw) == set(SOURCE_PINS), "exact five-source FE closure required")
    for role, pin in SOURCE_PINS.items():
        value = raw[role]
        require(type(value) is bytes and 0 < len(value) <= 256 * 1024
                and hashlib.sha256(value).hexdigest() == pin, "frozen FE source differs: " + role)
    np, sparse = api.np, api.sparse
    names = MappingProxyType({1: "sio2", 2: "lowk", 3: "si", 4: "cu"})
    bbar = _definitions(raw["bbar"], "bbar", functions=BBAR_FUNCTIONS,
        classes={"Hex8BBarPoint"},
        assignments={"Array", "GAUSS_ABSCISSA", "GAUSS_POINTS", "NATURAL_NODE_COORDINATES"},
        namespace=dict(np=np, dataclass=dataclass, product=product))
    material = _definitions(raw["material"], "material", functions={"isotropic_stiffness"},
                            namespace=dict(np=np))
    fields = _definitions(raw["fields"], "fields", functions={"analytic_fields"})
    native = _definitions(raw["native"], "native", functions={"native_reference"},
        classes={"NativeElasticReference"}, namespace=dict(dataclass=dataclass, NAMES=names, require=require))
    primitives = NS(np=np, sparse=sparse, bbar=bbar,
        isotropic_stiffness=material.isotropic_stiffness, analytic_fields=fields.analytic_fields)
    # The original PINS loop reads file modes only. No files are accessed here;
    # its empty diagnostics are replaced below by consumed-source byte proofs.
    quadrature = _definitions(raw["quadrature"], "quadrature", functions=QUADRATURE_FUNCTIONS,
        namespace=dict(math=math, SimpleNamespace=NS, NAMES=names, require=require,
                       PINS={}, load_primitives=lambda: primitives))
    return NS(token=_TOKEN, np=np, sparse=sparse, native=native, quadrature=quadrature,
              primitives=primitives, source_pins=SOURCE_PINS)


def _subdivisions(np, basis):
    y = np.unique(basis["coordinates_mm"][:, 1])
    require(len(y) >= 4 and y[0] == 0. and y[-1] > .0115,
            "origin-based BI/BEOL/Si thickness intervals required")
    indices = []
    for interface in (.0015, .0115):
        found = np.flatnonzero(np.abs(y - interface) <= 1e-14)
        require(len(found) == 1, "target is missing a unique fixed BI/BEOL interface")
        indices.append(int(found[0]))
    first, second = indices
    require(0 < first < second < len(y) - 1, "three positive regional thickness subdivisions required")
    return first, second - first, len(y) - 1 - second


def assemble_target(api, bound, *, target, limits, check):
    """Assemble all physical rows and Cu GPs from one prepared target.

    The original per-shape Bbar kernels, stiffness symmetrization and material
    law are retained. Exact-width shape grouping is used: no coordinate rounding,
    cap increases, congruent-element substitution or point subsampling.
    """
    require(getattr(bound, "token", None) is _TOKEN and bound.np is api.np
            and bound.sparse is api.sparse, "authenticated FE binding and identical numerical API required")
    require(callable(check), "owner resource/expiry guard required")
    check()
    require(type(target) is height.GeometryTarget and type(target.proof) is dict
            and target.proof.get("state") == "HEIGHT_RETARGET_GEOMETRY_CANDIDATE_ONLY_NOT_ADMITTED",
            "explicit prepared geometry target required, not an admitted operator")
    require(type(limits) is dict and set(limits) == LIMIT_KEYS, "explicit finite FE allocation limits required")
    proof, basis, np = target.proof, target.basis, api.np
    expected_gp = proof.get("Cu_GP")
    require(type(expected_gp) is int and expected_gp > 0
            and 8 * int(np.count_nonzero(basis["materials"] == 4)) == expected_gp,
            "full target Cu census differs before assembly")
    subdivisions = _subdivisions(np, basis)
    reference = bound.native.native_reference(target.deck)
    check()
    result = bound.quadrature.assemble_native(api, None, basis, reference,
        subdivisions=subdivisions, coordinate_quantum_mm=None, checkpoint=check, **limits)
    check()
    observed = result["proof"]
    require(observed["Cu_GP"] == expected_gp
            and observed["physical_nodes"] == proof["physical_nodes"]
            and observed["micro_coordinates"] == proof["body_coordinates"]
            and {name: value["Cu_GP"] for name, value in observed["regions"].items()}
                == {name: value["Cu_GP"] for name, value in proof["target_regions"].items()},
            "assembled full physical/regional ledger differs from target")
    # These matrices and arrays were just allocated by the pinned assembler.
    # Prepare their storage before sealing; never normalize borrowed inputs.
    csr_fields = ("k", "kff", "B_global", "B_free")
    for name in csr_fields:
        check()
        matrix = result[name]
        matrix.sum_duplicates()
        matrix.sort_indices()
        require(matrix.has_canonical_format, "newly assembled canonical CSR preparation failed")
        for value in (matrix.data, matrix.indices, matrix.indptr):
            value.setflags(write=False)
        check()
    for value in result.values():
        if type(value) is np.ndarray:
            value.setflags(write=False)
    # Authenticate consumed source bytes without claiming their file identities
    # or modes were verified by this I/O-free adapter.
    observed.pop("dependencies_compiled_from_identity_checked_SHA_pinned_bytes")
    observed.pop("dependency_modes")
    observed.pop("source_sha256")
    observed.update(state="ASSEMBLED_NATIVE_STATIC_UNFACTORED_UNVALIDATED",
        source_bytes_authenticated=True, source_sha256=dict(SOURCE_PINS), file_modes_not_read=True,
        file_identity_verified_here=False, geometry_provenance_verified_here=False,
        caller_resource_admission_required=True, resource_admission_performed_here=False,
        regional_subdivisions=list(subdivisions), stiffness_symmetrization="unchanged original physical_stiffness",
        canonical_CSR_output_fields=list(csr_fields), assembled_static_buffers_readonly=True,
        CSR_storage_operations=["sum_duplicates", "sort_indices"], borrowed_input_buffers_modified=False,
        numerical_definitions_rewritten=False, coordinate_rounding_enabled=False,
        history_executed=False, bow_accuracy_validated=False, assembly_accuracy_admission=False)
    check()
    return NS(assembly=result, reference=reference, proof=observed)


def bind_responses(api, raw):
    """Extract original response algebra, excluding imports, I/O and launching.

    The only adaptations are the factor-factory call, guards, residual recording
    and omission of vendor-only statistics. An inverse AST check accounts for
    every adaptation. The original numerical expressions and gates are retained.
    """
    require(type(raw) is bytes and 0 < len(raw) <= 256 * 1024
            and hashlib.sha256(raw).hexdigest() == ELASTIC_REFERENCE_SHA,
            "exact bounded native elastic-reference source required")
    functions = [node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    require(len(functions) == 1, "one original native reference run function required")
    body = functions[0].body
    anchor = ast.dump(ast.parse("kff,c,kaa = (assembly[name] for name in ('kff','c','kaa'))").body[0])
    starts = [index for index, node in enumerate(body) if ast.dump(node) == anchor]
    require(len(starts) == 1, "one exact native response block required")
    block = deepcopy(body[starts[0]:starts[0] + 7])
    original = ast.dump(ast.Module(body=block, type_ignores=[]))
    require(len(block) == 7 and isinstance(block[-1], ast.Try)
            and ast.dump(block[-1].finalbody[0]) == ast.dump(ast.parse("factor.close()").body[0])
            and len(block[-1].finalbody) == 1 and not block[-1].handlers and not block[-1].orelse,
            "original factor lifetime must be retained")
    factory, protected = block[-2], block[-1]
    require(isinstance(factory, ast.Assign) and ast.dump(factory.targets[0]) == ast.dump(ast.Name(id="factor", ctx=ast.Store()))
            and isinstance(factory.value, ast.Call) and ast.unparse(factory.value.func) == "backend.PardisoFactor",
            "one original backend constructor required")
    constructor = factory.value
    factory.value = ast.parse("factor_factory(scaled)", mode="eval").body
    vendor = protected.body.pop()
    require(isinstance(vendor, ast.Assign) and ast.unparse(vendor.targets[0]) == "factor_report",
            "one final vendor-only diagnostic assignment required")
    injected = []

    def insert(sequence, index, expression):
        node = ast.parse(expression).body[0]
        sequence.insert(index, node)
        injected.append((sequence, node))

    solves = [node for node in protected.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
    require(len(solves) == 1, "one original physical body solve required")
    solve = solves[0]
    require(ast.unparse(solve.body[0].targets[0]) == "answer" and isinstance(solve.body[-1], ast.Return),
            "original physical solve boundaries required")
    insert(solve.body, 0, "check()")
    insert(solve.body, 2, "check()")
    insert(solve.body, len(solve.body) - 1, "record_body_residual(float(error))")
    thermal = [node for node in protected.body if isinstance(node, ast.Assign)
               and ast.unparse(node.targets[0]) == "thermal_q"]
    require(len(thermal) == 1, "one original coupled thermal solve required")
    index = protected.body.index(thermal[0])
    insert(protected.body, index, "check()")
    insert(protected.body, index + 2, "check()")
    insert(protected.body, 0, "check()")
    insert(block, block.index(factory), "check()")
    insert(block, 0, "check()")
    returned = ast.parse("return dict(schur=schur, mechanical_q=mechanical_q, mechanical_u=mechanical_u, "
        "mechanical_reactions=reactions, thermal_q=thermal_q, thermal_u=thermal_u, "
        "coupled_thermal_relative_residual=float(error))").body[0]
    function = ast.FunctionDef(name="responses", args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=name) for name in ("basis", "assembly", "factor_factory", "check", "record_body_residual")],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=block + [returned], decorator_list=[])
    module = ModuleType("_operator_native_elastic_responses")
    vars(module).update(np=api.np, sparse=api.sparse, linalg=api.linalg, require=require)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "<authenticated-native-elastic-responses>", "exec", dont_inherit=True), vars(module))
    for sequence, node in reversed(injected):
        sequence.remove(node)
    factory.value = constructor
    protected.body.append(vendor)
    require(ast.dump(ast.Module(body=block, type_ignores=[])) == original,
            "inverse adaptation must restore every original response statement")
    return NS(token=_RESPONSE_TOKEN, np=api.np, sparse=api.sparse, linalg=api.linalg,
        responses=module.responses, source_sha256=ELASTIC_REFERENCE_SHA,
        inverse_AST_exact=True, guard_insertions=7,
        original_block_AST_sha256=hashlib.sha256(original.encode("utf-8")).hexdigest())


def _immutable_array(np, value, shape, name, check, *, integer=False):
    require(type(value) is np.ndarray and value.shape == shape and not value.flags.writeable
            and (value.flags.c_contiguous or value.flags.f_contiguous), "immutable contiguous native array required: " + name)
    require(value.dtype in (np.dtype("int32"), np.dtype("int64")) if integer
            else value.dtype == np.dtype("float64"), "native dtype required: " + name)
    flat = value.ravel(order="K")
    require(not value.size or np.shares_memory(value, flat), "validation must not copy native arrays")
    for first in range(0, value.size, 131072):
        check()
        require(bool(np.isfinite(flat[first:first + 131072]).all()), "nonfinite native array: " + name)


def _immutable_csr(api, value, shape, name, check):
    np = api.np
    require(api.sparse.isspmatrix_csr(value) and value.shape == shape, "native CSR dimensions required: " + name)
    _immutable_array(np, value.data, value.data.shape, name + ".data", check)
    _immutable_array(np, value.indices, value.data.shape, name + ".indices", check, integer=True)
    _immutable_array(np, value.indptr, (shape[0] + 1,), name + ".indptr", check, integer=True)
    require(value.indptr[0] == 0 and value.indptr[-1] == len(value.data), "native CSR pointer endpoints differ")
    for first in range(0, shape[0], 131072):
        check()
        part = value.indptr[first:min(first + 131072, shape[0]) + 1]
        require(bool(np.all(part[1:] >= part[:-1])), "nonmonotone native CSR pointers")
    for first in range(0, len(value.indices), 131072):
        check()
        part = value.indices[first:first + 131072]
        require(bool(np.all((part >= 0) & (part < shape[1]))), "native CSR index out of range")
    require(value.has_canonical_format, "canonical native CSR required; inputs are not repaired")


class _FactorLease:
    """Preallocated handle proxy; construction failure cleanup belongs to owner."""

    def __init__(self, factory):
        self.factory = factory
        self.handle = None
        self.factory_calls = self.solve_calls = self.close_calls = 0
        self.close_returned_successfully = False

    def acquire(self, scaled):
        self.factory_calls += 1
        self.handle = self.factory(scaled)
        return self

    def solve(self, rhs):
        self.solve_calls += 1
        return self.handle.solve(rhs)

    def close(self):
        self.close_calls += 1
        self.handle.close()
        self.close_returned_successfully = True


def elastic_operands(api, bound, *, basis, assembly, factor_factory, check):
    """Return exact19 readonly operands using the original native response block.

    factor_factory receives the canonical SCALED CSR D*Kff*D and must return a
    handle with solve(scaled_rhs) and close(). This adapter applies D on both
    sides. The factory owns partial construction cleanup; after acquisition the
    original finally always calls close. The owner must retain stable inputs and
    provide resource/library qualification. Borrowed storage is not copied.
    """
    require(getattr(bound, "token", None) is _RESPONSE_TOKEN and bound.np is api.np
            and bound.sparse is api.sparse and bound.linalg is api.linalg, "authenticated response binding and identical API required")
    require(callable(check) and callable(factor_factory), "owner guard and scaled-system factor factory required")
    check()
    np, p, k = api.np, basis["P"], assembly["kff"]
    require(len(p.shape) == 2 and p.shape[0] > 0 and p.shape[0] % 3 == 0 and p.shape[1] > 0,
            "positive native physical/body dimensions required")
    physical, micro = p.shape
    _immutable_csr(api, p, (physical, micro), "P", check)
    _immutable_csr(api, k, (micro, micro), "kff", check)
    for name, shape in (("kaa", (6, 6)), ("c", (micro, 6)), ("fm", (6,)), ("ff", (micro,))):
        _immutable_array(np, assembly[name], shape, name, check)
    for name, shape in (("coordinates_mm", (physical // 3, 3)), ("macro_fields", (physical, 6))):
        _immutable_array(np, basis[name], shape, name, check)
    diagonal = k.diagonal()
    physical_height = float(np.ptp(basis["coordinates_mm"][:, 1]))
    require(bool(np.isfinite(diagonal).all()) and bool(np.all(diagonal > 0.))
            and math.isfinite(physical_height) and physical_height > 0.,
            "positive finite native energy diagonal and physical height required")
    check()
    lease, residuals = _FactorLease(factor_factory), []
    try:
        response = bound.responses(basis, assembly, lease.acquire, check, residuals.append)
    finally:
        # Cover interruption between acquisition and the extracted inner try.
        # A failed close is never retried automatically.
        if lease.handle is not None and lease.close_calls == 0:
            lease.close()
    require(lease.factory_calls == 1 and lease.solve_calls == 7 and lease.close_calls == 1
            and lease.close_returned_successfully and len(residuals) == 7,
            "complete original seven-solve factor lifecycle required")
    check()
    arrays = {name: assembly[name] for name in ("kaa", "c", "fm", "ff")}
    arrays.update(macro_fields=basis["macro_fields"], **{name: response[name] for name in
        ("schur", "mechanical_q", "mechanical_u", "mechanical_reactions", "thermal_q", "thermal_u")})
    for prefix, matrix in (("P", p), ("kff", k)):
        arrays.update({prefix + "_" + name: getattr(matrix, name) for name in ("data", "indices", "indptr")})
        arrays[prefix + "_shape"] = np.asarray(matrix.shape, dtype=np.int64)
    require(set(arrays) == ELASTIC_KEYS, "exact19 static elastic output schema required")
    # Freeze only newly computed arrays. Borrowed inputs were already immutable.
    for value in arrays.values():
        if value.flags.writeable:
            value.setflags(write=False)
    check()
    return NS(arrays=arrays, report=dict(state="CLOSED_NATIVE_ELASTIC_RESPONSES_UNVALIDATED",
        source_sha256=ELASTIC_REFERENCE_SHA, source_bytes_authenticated=True,
        original_numeric_block_inverse_AST_exact=bound.inverse_AST_exact,
        original_block_AST_sha256=bound.original_block_AST_sha256,
        factor_factory_matrix="D*Kff*D canonical CSR; physical answer D*solve(D*rhs)",
        factory_calls=lease.factory_calls, body_solve_calls=lease.solve_calls, close_calls=lease.close_calls,
        close_returned_successfully=lease.close_returned_successfully,
        external_factor_implementation_verified_here=False, OS_resource_release_verified_here=False,
        relative_body_residuals=residuals,
        coupled_thermal_relative_residual=response["coupled_thermal_relative_residual"],
        original_residual_gate=1e-8, original_residual_denominator_floor=1e-30,
        mechanical_columns_are_unit_amplitude=False, physical_height_mm=physical_height,
        input_buffers_copied=False, numeric_input_custody_verified_here=False, file_identity_verified_here=False,
        library_loaded_here=False, vendor_timing_claim=False, resource_admission=False,
        Cu_updates=0, history_executed=False, bow_accuracy_validated=False, operator_admission=False))
