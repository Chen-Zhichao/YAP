"""Owned-source BI4/Si37 dual-BI50 geometry; no I/O, solver or admission.

Only selected pinned functions execute: legacy module affinity/import side
effects never run. Main owns source custody, seed construction and resources.
"""
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
from types import ModuleType, SimpleNamespace as NS


SOURCE_PINS = {
    'append': 'fd37a15adde2fe56b97e489d8515f3cdc2d0a1c969bc157becdc2fe159f7b5ff',
    'quadrature': '254b8d815953166afb51b49aefb8f5192e9619b810b88251647e2f1f685b2e85',
    'geometry': '4d1165905829da719742666cd78e2ceed2f8592c5ecc786b55b9aa7f6dbaf5d1',
    'extension': '37d7ca5f2317c130865307cc8527476e6f6bdf9127765c2a95d6503d211cd729',
    'periodic': 'de638f0fb73785694324df2155ba35bfd4d2a7090de1a47199291f682d5839c6',
}
_TOKEN = object()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _function(raw, name):
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, 'one original function required: ' + name)
    return nodes[0]


def _adapt(node, pairs):
    """Count exact AST replacements and prove the full function inverse."""
    original = ast.dump(node)
    parsed = [(ast.parse(a, mode='eval').body, ast.parse(b, mode='eval').body) for a, b in pairs]

    class Change(ast.NodeTransformer):
        def __init__(self, rules):
            self.rules, self.counts = rules, [0] * len(rules)

        def visit(self, value):
            for i, (before, after) in enumerate(self.rules):
                if ast.dump(value) == ast.dump(before):
                    self.counts[i] += 1
                    return deepcopy(after)
            return super().visit(value)

    forward = Change(parsed)
    changed = forward.visit(deepcopy(node))
    require(forward.counts == [1] * len(pairs), 'original adaptation anchors differ')
    backward = Change([(b, a) for a, b in parsed])
    restored = backward.visit(deepcopy(changed))
    require(backward.counts == [1] * len(pairs) and ast.dump(restored) == original,
            'whole-function adaptation inverse differs')
    return changed, dict(inverse_AST_exact=True, changed_anchors=len(pairs),
                        original_AST_sha256=hashlib.sha256(original.encode()).hexdigest())


def _compile(node, namespace):
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<pinned-prepared-coupon-function>', 'exec', dont_inherit=True), scope)
    return scope[node.name]


def bind_sources(raw):
    require(type(raw) is dict and set(raw) == set(SOURCE_PINS), 'exact five-source closure required')
    for name, pin in SOURCE_PINS.items():
        value = raw[name]
        require(type(value) is bytes and 0 < len(value) <= 256 * 1024
                and hashlib.sha256(value).hexdigest() == pin, 'pinned source differs: ' + name)
    geometry = ModuleType('_prepared_coupon_native_geometry')
    exec(compile(raw['geometry'], '<pinned-native-geometry>', 'exec', dont_inherit=True), vars(geometry))
    analytic = _compile(_function(raw['extension'], 'analytic_fields'), {})
    periodic = _compile(_function(raw['periodic'], 'periodic_equations'), {})
    append, append_proof = _adapt(_function(raw['append'], 'append_back_bi'), [
        ("(nx,ny,nz) == (45,44,45)", "(nx,ny,nz) == (96,44,96)"),
        ("8*np.count_nonzero(mats == 4) == 185696",
         "8*np.count_nonzero(mats == 4) == 1038976 and np.allclose(np.ptp(xyz,axis=0),[.012,.0485,.012],rtol=0,atol=1e-14)"),
        ("'production native45 full seed only'", "'production BI4 cell12 transition96 Si37 seed only'"),
        ("result['P'].shape == (323748,309822) and offset == 205136 and [b-a for _,a,b in ranges] == [19440,102096,64160,19440]",
         "result['P'].shape == (1439577,1410045) and offset == 1160704 and [b-a for _,a,b in ranges] == [121728,569088,348160,121728]"),
    ])
    production_guards = tuple(deepcopy(n.body[0].value.args[0]) for n in append.body
                              if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                              and n.test.id == 'production')
    require(len(production_guards) == 2, 'both original production guard sites required')
    original_plan = _function(raw['quadrature'], 'tensor_plan')
    plan, plan_proof = _adapt(original_plan, [
        ('len(subdivisions) == 3', 'len(subdivisions) == 4'),
        ("'explicit three-region thickness ledger required'", "'explicit four-region thickness ledger required'"),
        ("zip(('bi','beol','si'),(1,2,3),subdivisions)",
         "zip(('front_bi','beol','si','back_bi'),(1,2,3,1),subdivisions)"),
    ])
    scope = dict(require=require, math=math, SimpleNamespace=NS)
    return NS(token=_TOKEN, geometry=geometry, analytic=analytic, periodic=periodic,
              append=_compile(append, dict(require=require, replace=replace, NS=NS)),
              production_guard_AST=production_guards,
              tensor_plan=_compile(plan, scope), original_tensor_plan=_compile(original_plan, scope),
              proof=dict(source_pins=dict(SOURCE_PINS), append=append_proof, tensor_plan=plan_proof,
                         original_Bbar_assembly_modified=False, legacy_module_imported=False))


def append_back_bi(api, bound, *, seed_deck, seed_basis, check, production=True, divisions=(6, 18, 20)):
    """Original append arithmetic, with BI4 guards replacing BI2 census constants."""
    require(getattr(bound, 'token', None) is _TOKEN and callable(check), 'bound sources and owner guard required')
    check()
    native_api = NS(geometry=bound.geometry, analytic=bound.analytic, periodic=bound.periodic, sparse=api.sparse)
    deck, basis, proof = bound.append(api.np, seed_deck, seed_basis, native_api,
                                      production=production, divisions=divisions)
    check()
    proof.update(source_binding=deepcopy(bound.proof), production_geometry_guards_checked=production,
                 actual_census_independently_qualified=False, resource_admission=False,
                 native_bow_accuracy_validated=False, geometry_only=True)
    return NS(deck=deck, basis=basis, proof=proof, divisions=(*divisions, divisions[0]))


def construct(api, bound, *, construct_bi4, check):
    """construct_bi4() is Main's admitted SeedSources.construct(api, 4) callback."""
    require(callable(construct_bi4) and callable(check), 'owner BI4 constructor and guard required')
    check()
    seed = construct_bi4()
    require(seed.profile['pitch_um'] == 4 and seed.profile['diameter_um'] == 2.
            and seed.profile['source_si_um'] == 37. and seed.profile['cell_um'] == 12
            and seed.profile['mesh_recipe'] == 'BI4_TRANSITION96', 'exact BI4/Si37 seed profile required')
    result = append_back_bi(api, bound, seed_deck=seed.deck, seed_basis=seed.basis, check=check)
    result.proof['seed_proof'] = deepcopy(seed.proof)
    return result


def tensor_plan(api, bound, *, target, bbar, check, max_shapes=4096, chunk=512):
    """Exact-width plan only; a larger shape ceiling does not admit Bbar cache RAM."""
    require(getattr(bound, 'token', None) is _TOKEN and callable(check), 'bound sources and owner guard required')
    require(type(max_shapes) is int and 0 < max_shapes <= 32768
            and type(chunk) is int and 0 < chunk <= 512, 'finite explicit shape/chunk limits required')
    check()
    primitives = NS(np=api.np, sparse=api.sparse, bbar=bbar, analytic_fields=bound.analytic)
    plan = bound.tensor_plan(primitives, target.basis, target.divisions, None, max_shapes, chunk)
    check()
    return plan
