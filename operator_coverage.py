"""Bounded, read-only coverage inventory; never build, import solvers, or scan trees."""

from __future__ import annotations

import argparse
import ast
import csv
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import stat
import time


ROOT = Path('/app/nanocad/projects/personal/zchen/YAP_STACK')
PROGRESS = Path('operator_repair_20261015/progress')
DEVELOPMENT = Path('operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1')
DELIVERY = Path('distribution/yap_prepared_p2_20261006_v1/YAP/runtime')
NATIVE = Path('Ansys_Verify/12_stack_model_validation')
SEED_DECKS = Path('operator_repair_20261015/global_warpage_20261017_v4/native')
OLD_NATIVE = tuple(Path(p) / 'Ansys_Verify/12_stack_model_validation' for p in (
    '/w/ee.00/puneet/zhichao/Desktop/YAP_STACK',
    '/u1/ee/zhichao/Desktop/YAP_STACK', str(ROOT)))


def literal_path(value):
    path = Path(value)
    if not path.is_absolute() or {'..', '#snapshot', '.snapshot'}.intersection(path.parts):
        raise ValueError('absolute nonsnapshot path without parent traversal required')
    return path


@contextmanager
def parent_fd(path):
    """Pin every directory without following symlinks, including ancestor aliases."""
    path = literal_path(path)
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, path.name
    finally:
        os.close(fd)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key: ' + key)
        value[key] = item
    return value


class Inventory:
    def __init__(self, roots, *, seconds=90, max_files=256, max_bytes=12 * 1024**2):
        self.roots = tuple(literal_path(p) for p in roots)
        self.deadline = time.monotonic() + seconds
        self.max_files, self.max_bytes = max_files, max_bytes
        self.bytes_read = 0
        self.files = {}

    def check(self, path):
        path = literal_path(path)
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError('path outside literal inventory roots: ' + str(path))
        if time.monotonic() >= self.deadline:
            raise TimeoutError('inventory wall budget exceeded')
        if str(path) not in self.files and len(self.files) >= self.max_files:
            raise ValueError('inventory file budget exceeded')
        return path

    def metadata(self, path):
        path = self.check(path)
        record = dict(path=str(path), status='MISSING', payload_hashed=False)
        try:
            with parent_fd(path) as (fd, name):
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            record.update(status='REGULAR' if stat.S_ISREG(info.st_mode) else 'BLOCKED_NONREGULAR',
                          bytes=info.st_size, device=info.st_dev, inode=info.st_ino,
                          mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns, mode=info.st_mode)
        except FileNotFoundError:
            pass
        except OSError as exc:
            record.update(status='BLOCKED_NOFOLLOW', errno=exc.errno)
        self.files[str(path)] = record
        return record

    def read(self, path, *, expected=None, limit=1024**2):
        path = self.check(path)
        with parent_fd(path) as (fd, name):
            leaf = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(leaf, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError('bounded regular JSON/source required: ' + str(path))
            if self.bytes_read + before.st_size > self.max_bytes:
                raise ValueError('inventory byte budget exceeded')
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        def identity(info):
            return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                    info.st_ctime_ns, info.st_mode)
        if len(raw) > limit or len(raw) != before.st_size or identity(before) != identity(after):
            raise ValueError('input changed or exceeded bound: ' + str(path))
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('receipt SHA256 mismatch: ' + str(path))
        self.bytes_read += len(raw)
        metadata = self.metadata(path)
        observed = tuple(metadata.get(key) for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns', 'mode'))
        if observed != identity(before):
            raise ValueError('input path identity changed after read: ' + str(path))
        metadata.update(sha256=digest, bounded_content_read=True)
        return raw

    def json(self, path, **kwargs):
        value = json.loads(self.read(path, **kwargs), object_pairs_hook=unique_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(value, dict):
            raise ValueError('JSON object required')
        return value


def registry(source, name):
    """Read literal backend registries without importing runtime initialization."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            class Paths(ast.NodeTransformer):
                def visit_Call(self, call):
                    if (isinstance(call.func, ast.Name) and call.func.id == 'Path'
                            and len(call.args) == 1 and not call.keywords):
                        return call.args[0]
                    raise ValueError('nonliteral registry call')
            return ast.literal_eval(Paths().visit(node.value))
    raise ValueError('missing registry: ' + name)


def resource_path(native, mapping, reference):
    name, sep, suffix = reference.removeprefix('resource:').partition('/')
    if not reference.startswith('resource:') or not sep:
        raise ValueError('resource reference required')
    if name not in mapping:
        return None
    for part in (mapping[name], suffix):
        if Path(part).is_absolute() or {'..', '.', '#snapshot', '.snapshot'}.intersection(part.split('/')):
            raise ValueError('unsafe resource binding')
    return native / mapping[name] / suffix


def payload_path(value, package, native):
    path = Path(value)
    if path.is_absolute():
        for old in OLD_NATIVE:
            if path.is_relative_to(old):
                return literal_path(native / path.relative_to(old))
        raise ValueError('unrecognized absolute payload reference')
    if {'#snapshot', '.snapshot'}.intersection(path.parts):
        raise ValueError('snapshot payload reference')
    path = Path(os.path.normpath(package.parent / path))
    if not path.is_relative_to(native):
        raise ValueError('payload escapes native root')
    return path


def package_record(inv, path, native, pitch, total, *, pin=None):
    record = dict(bonding_pitch_um=pitch, total_thickness_um=total, package=str(path),
                  package_exists=False, payload_exists=False, structural_candidate=False,
                  validated_global_bow=False, D2W_assembly=False, W2W_assembly=False)
    if inv.metadata(path)['status'] != 'REGULAR':
        record['status'] = 'PACKAGE_NOT_ACCESSIBLE_NOFOLLOW'
        return record
    package = inv.json(path, expected=pin)
    if package.get('name') != f'BI{pitch}_{total}':
        raise ValueError('package geometry name mismatch')
    record.update(package_exists=True, declared_state=package.get('state'), payloads={})
    datasets, prepared = package.get('datasets', {}), package.get('prepared', {})
    if not 1 <= len(datasets) <= 12 or set(datasets) != set(prepared):
        raise ValueError('invalid bounded package dataset/prepared ledger')
    for role, table in (('datasets', datasets), ('prepared', prepared),
                        ('basis', {'port_basis': package['port_basis']})):
        for key, value in table.items():
            target = payload_path(value, path, native)
            record['payloads'][role + ':' + key] = inv.metadata(target)
    complete = all(p['status'] == 'REGULAR' and p['bytes'] > 0 for p in record['payloads'].values())
    structural = package.get('state') == 'READY_ENGINEERING_PREVIEW'
    if package.get('schema') == 'yap.prepared_p2_geometry_package.v1':
        audit_path = payload_path(package['audit_receipt'], path, native)
        audit = inv.json(audit_path, expected=package['audit_receipt_sha256'])
        structural = (audit.get('state') == 'PASS_STRUCTURAL_ENGINEERING_CANDIDATE'
                      and audit.get('total_thickness_um') == total
                      and audit.get('bonding_pitch_um') == pitch)
    record.update(payload_exists=complete, structural_candidate=structural and complete,
                  structural_receipt_declared=structural,
                  status='METADATA_PRESENT_NOT_PAYLOAD_REHASHED' if complete else 'INCOMPLETE_PAYLOAD_LEDGER')
    return record


def formal_cases(inv, formal):
    scope = formal['scope']
    if (formal.get('schema') != 'yap.global_bow_strict_or_admission.v1'
            or formal.get('state') != 'BOTH_ORIGINAL1081_GLOBAL_BOW_ACCURACY_PASS'
            or formal.get('global_accuracy_passed') is not True
            or scope.get('BI_pitch_um') != 2 or scope.get('BI_um') != 1.5
            or scope.get('BEOL_um') != 10 or scope.get('states') != 1081
            or scope.get('no_arbitrary_geometry_or_assembly_order_accuracy_claim') is not True):
        raise ValueError('unexpected formal bow acceptance scope')
    rows = []
    for key, si, total in (('h48p5', 48.5, 60), ('h87', 87, 98.5)):
        case = formal['cases'][key]
        if case['Si_TSV_um'] != si or case['total_um'] != total:
            raise ValueError('formal case thickness mismatch')
        execution = inv.json(case['execution']['path'], expected=case['execution']['sha256'])
        if execution.get('case') != key:
            raise ValueError('execution case mismatch')
        artifacts = execution['arrays_read_hash_decode_submeasurement']
        if not 1 <= len(artifacts) <= 12:
            raise ValueError('invalid execution artifact ledger')
        payloads = []
        for artifact in artifacts:
            binding = artifact['binding']
            metadata = inv.metadata(binding['path'])
            payloads.append({**metadata, 'receipt_sha256': binding['sha256'],
                             'recorded_bytes_match': metadata.get('bytes') == binding['bytes']})
        accuracy = all(case[name].get('global_accuracy_passed') is True
                       for name in ('native_fullspace_accuracy', 'reduced_operator_accuracy'))
        rows.append(dict(case=key, bonding_pitch_um=2, total_thickness_um=total, Si_TSV_um=si,
                         BI_layer_count=1, payload_exists=all(p['status'] == 'REGULAR' and p['recorded_bytes_match'] for p in payloads),
                         structural_candidate=True, validated_global_bow=accuracy,
                         D2W_assembly=False, W2W_assembly=False, payloads=payloads,
                         scope=scope, proof=case['execution'],
                         seconds_level_runtime=False, production_integration=False))
    return rows


def coverage_rows(packages, repaired):
    rows = []
    for pitch in range(2, 11):
        for total in range(20, 101, 10):
            legacy = [p for p in packages if p['bonding_pitch_um'] == pitch and p['total_thickness_um'] == total]
            single = [p for p in repaired if p['bonding_pitch_um'] == pitch and p['total_thickness_um'] == total]
            rows.append(dict(bonding_pitch_um=pitch, total_thickness_um=total,
                             common_cell_side_um=math.lcm(pitch, 6),
                             single_BI_si_um=total - 11.5, two_BI_si_um=total - 13,
                             package_exists=any(p['package_exists'] for p in legacy),
                             payload_exists=any(p['payload_exists'] for p in legacy + single),
                             structural_candidate=any(p['structural_candidate'] for p in legacy + single),
                             validated_global_bow=any(p['validated_global_bow'] for p in single),
                             validated_scope='single_BI_macro_homogenized_only' if single else 'NO_EXACT_FORMAL_PROOF',
                             D2W_assembly=False, W2W_assembly=False,
                             inventory_scope='literal registries/packages only; absence is not a whole-tree claim'))
    return rows


def expansion_plan(inv, root, packages):
    source = root / DEVELOPMENT
    files = [source / name for name in ('native_monolithic_reference.py', 'native_periodic_geometry.py',
             'native_monolithic_assembly.py', 'causal_sparse_geometry.py', 'build_shared_component_catalog.py')]
    files += [root / NATIVE / 'prepare_exp12_layer_geometry.py', root / 'YAP/operator_grid.py']
    sources = {str(p): hashlib.sha256(inv.read(p)).hexdigest() for p in files}
    seed_decks = [inv.metadata(root / SEED_DECKS / case / 'prepared_v3/macro_history_stack_rve_plan.dat')
                  for case in ('h48p5', 'h87')]
    canonical_parent = root / NATIVE / 'results/date_ofat_geometry_20260913_v1/packages/BI2_50.json'
    endpoints = {}
    for package in packages:
        for key, item in package.get('payloads', {}).items():
            if key.startswith('datasets:tsv_si_'):
                endpoints.setdefault(key.split(':', 1)[1], []).append(item)
    return dict(
        preferred_new_cases=[dict(Si_um=h, single_BI_total_um=h + 11.5) for h in (38.5, 68.5)],
        next_inner_endpoint_pairs=[dict(total_um=t, two_BI_Si_um=t-13, single_BI_Si_um=t-11.5)
                                   for t in (50, 80)],
        equivalent_full_native_cli_available=False, launch_ready=False, jobs_launched=False,
        existing_full_native_cli=shlex.join(['python', '-B', str(files[0]), '--case', 'h48p5', '--work-root', '<NEW_CANONICAL_REFERENCE_OUTPUT>']),
        existing_cli_limitation='--case choices h48p5/h87 only; command above documents existing syntax, NOT a new-case launch',
        operator_grid_equivalent=False,
        operator_grid_reasons=['validate_batch excludes existing Si37/38.5/67; Si68.5 is accepted',
                              'GeometryCase builds isolated native64 port/layer data, not the full periodic six-macro FE space',
                              'launch requires parent Si37/38.5, donor/template/basis pins, reviewed controller, baseline inventory and fresh host resources'],
        geometry_compiler_cli=shlex.join(['python', '-B', str(root / NATIVE / 'prepare_exp12_layer_geometry.py'),
            '--dataset', 'tsv_si_h38p5=<EXACT_VERIFIED_DATASET_NPZ>', '--rank', '64', '--output-dir', '<NEW_OUTPUT_DIR>']),
        geometry_compiler_limitation='Requires existing LayerROMDataset; does not generate full-native FE or new thicknesses',
        existing_dataset_metadata=endpoints, canonical_parent=str(canonical_parent),
        seed_native_decks_metadata_only=seed_decks,
        seed_deck_payload_read=False,
        proposed_retarget_is_not_executed_or_certified=True,
        next_launch_contract=dict(
            cases=['Si38.5_singleBI_total50', 'Si68.5_singleBI_total80'],
            first_action='Implement and independently review NEW parameterized candidate-FE wrapper in a separately owned file',
            no_existing_exact_new_case_command=True,
            reuse='Seed topology/material labels and periodic physical P graph only after identity checks',
            outputs='New per-case canonical FE-reference output/control, never old raw/prefix or existing results',
            run_after='Fresh host resource checks, all listed geometry/assembly certificates and explicit reviewed launch'),
        required_available=[str(p) for p in files], bounded_source_sha256=sources,
        required_missing_or_not_yet_certified=[
            'New reviewed parameterized candidate-FE entry point and explicit case/input/output/hash bindings for Si38.5 and Si68.5',
            'Literal seed native deck/topology full hash and retarget-use binding; metadata alone is not a payload proof',
            'Retarget physical thickness coordinate y only above 0.0115 mm: y_new=0.0115+(y_old-0.0115)*h_new/h_old; x/z unchanged',
            'Keep BI/BEOL unchanged; preserve ordered node/EN/MAT/GP identities, 6x6 um cell, pitch2, 185696 Cu GPs and 14 state fields',
            'Recompute six analytic macro fields and auxiliary/CE macro coefficients at the new y levels; unchanged physical P graph alone is insufficient',
            'Certify constraints, gauge, rank, dimensions, positive Jacobians, regional thicknesses/volumes and new Cu positions/weights',
            'Fresh Bbar stiffness/thermal assembly and Pardiso factorization; never scale/reuse old K, factors, strain maps, thermal loads or certificates',
            'Fresh mechanical residual/energy, thermal/CTE and map/constraint certificates for each new FE reference',
            'Own geometry-specific first-cycle training and subsequent ROM construction/independent bow qualification remain separate future work',
            'Before launch: executing host actual local mount, free disk/RAM, jobs, dependencies and unique output/control paths'],
        candidate_scope='New elastic full-native FE reference prerequisites only; not completed ROM, thermal-history validation, MAPDL accuracy, D2W or W2W admission')


def collect(root):
    root = literal_path(root)
    inv = Inventory([root / 'YAP', root / DELIVERY, root / NATIVE,
                     root / DEVELOPMENT, root / PROGRESS, root / SEED_DECKS])
    mapping = inv.json(root / DELIVERY / 'RESOURCES.json')
    if mapping.get('schema') != 'yap.native_resources.v1':
        raise ValueError('invalid resource map schema')
    native = root / DELIVERY / 'native'
    catalog = inv.json(root / 'YAP/prepared_operator_catalog.json')
    backend = inv.read(root / 'YAP/history_mechanics_backend.py')
    registries = {key: registry(backend, key) for key in ('PREVIEW_PACKAGES', 'COMPACT_PACKAGES')}
    packages = []
    for total in (30, 80):
        path = resource_path(native, mapping['resources'], f'resource:prepared_p2_native/packages/BI2_{total}.json')
        if path is None:
            raise ValueError('missing prepared_p2_native binding')
        packages.append(package_record(inv, path, native, 2, total, pin=catalog['packages'][str(total)]['package_sha256']))
    for entry in registries['PREVIEW_PACKAGES'].values():
        pitch, _, _, total = entry['geometry']
        path = resource_path(native, mapping['resources'], entry['path'])
        if path is not None:
            packages.append(package_record(inv, path, native, int(pitch), int(total)))
    # Canonical exact p2 parent can contain endpoints omitted from the delivery.
    packages.append(package_record(inv, root / NATIVE / 'results/date_ofat_geometry_20260913_v1/packages/BI2_50.json',
                                   root / NATIVE, 2, 50))
    formal_path = root / PROGRESS / 'FORMAL_BOW_ACCEPTANCE_20261019_v1.json'
    formal = inv.json(formal_path)
    repaired = formal_cases(inv, formal)
    rows = coverage_rows(packages, repaired)
    plan = expansion_plan(inv, root, packages)
    counts = {key: sum(row[key] for row in rows) for key in (
        'package_exists', 'payload_exists', 'structural_candidate', 'validated_global_bow', 'D2W_assembly', 'W2W_assembly')}
    return dict(schema='yap.operator_coverage.bounded.v1', root=str(root),
                evidence=inv.files[str(formal_path)], grid_rows=rows, grid_combinations=len(rows),
                grid_counts=counts, packages=packages, repaired_single_BI=repaired,
                backend_registries=registries, expansion_plan=plan,
                inventory=list(inv.files.values()), bytes_read=inv.bytes_read,
                bounds=dict(max_files=inv.max_files, max_read_bytes=inv.max_bytes, wall_seconds=90),
                payload_hashes_reverified=False, recursive_scans=False, interpolation=False,
                legacy_geometry_scope='Legacy p2 total50 uses both Si37 and Si38.5; total30/80 candidates carry Si17/67 (two-BI ledger), not new single-BI proofs',
                assembly_status_meaning='False means not established by this evidence set, not proven impossible',
                single_BI_vs_assembly='Single-BI total T needs Si T-11.5; two-BI total T needs Si T-13. No transfer of accuracy across these ledgers.')


def write_outputs(output, report):
    output = literal_path(output)
    with parent_fd(output) as (parent, name):
        os.mkdir(name, dir_fd=parent)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        table = io.StringIO(newline='')
        writer = csv.DictWriter(table, fieldnames=list(report['grid_rows'][0]))
        writer.writeheader()
        writer.writerows(report['grid_rows'])
        single = io.StringIO(newline='')
        columns = ['case', 'bonding_pitch_um', 'total_thickness_um', 'Si_TSV_um', 'BI_layer_count',
                   'payload_exists', 'structural_candidate', 'validated_global_bow', 'D2W_assembly', 'W2W_assembly']
        single_writer = csv.DictWriter(single, fieldnames=columns, extrasaction='ignore')
        single_writer.writeheader()
        single_writer.writerows(report['repaired_single_BI'])
        documents = {'COVERAGE.json': json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n',
                     'GRID81.csv': table.getvalue(),
                     'REPAIRED_SINGLE_BI.csv': single.getvalue(),
                     'NEXT_BUILD.json': json.dumps(report['expansion_plan'], indent=2, sort_keys=True, allow_nan=False) + '\n'}
        for name, content in documents.items():
            child = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=fd)
            with os.fdopen(child, 'w') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.root != ROOT or args.output != ROOT / PROGRESS / 'coverage_20261019_v1':
        raise ValueError('this inventory is scoped to the literal canonical root and new coverage_20261019_v1')
    report = collect(args.root)
    write_outputs(args.output, report)
    print(json.dumps(dict(output=str(args.output), grid_counts=report['grid_counts'],
                          files=len(report['inventory']), bytes_read=report['bytes_read'])))


if __name__ == '__main__':
    main()
