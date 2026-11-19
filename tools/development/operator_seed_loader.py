"""Owner-injected Si37 pitch-seed adapter; no direct I/O, FE solve or launcher.

Constructors retain their frozen mesh, phase, material and CE algorithms. Only
private source-reader/path bindings are adapted. The caller must admit resources
BEFORE construct(); these are real geometry builds, not metadata operations.
BI7 remains blocked by the original exact-transition mesh limit.
"""
from __future__ import annotations

import builtins
import hashlib
import math
from pathlib import PurePosixPath
from types import MappingProxyType, ModuleType, SimpleNamespace


SOURCE_RELATIVE = "operator_repair_20261015/development/trial_forcing_closure_v1/continuous_cu_screen_v1"
SEED_RELATIVE = "operator_repair_20261015/global_warpage_20261017_v4/native/h48p5/prepared_v3/macro_history_stack_rve_plan.dat"
SEED_SHA = "ba6fd863c2156fe1ac51ca5f8e7d9ea610f2d738f8cd66aa56cddac995f420c8"
SOURCE_PINS = MappingProxyType({
    "bi_pitch_geometry": "ed49e22f800fa52abce33860455d5551a42c9c869e332d7bad74862621d5ceb1",
    "bi_remeshed_geometry": "de638f0fb73785694324df2155ba35bfd4d2a7090de1a47199291f682d5839c6",
    "bi3_coarse_candidate": "4ecf5b867fa1754b47bcc8f25cf710ef7d927ede1b068874c5aacd7ef163e929",
    "common_cell_geometry": "66791facfd98716388d17e66c53f2d3f3a250a2ff3cb51ba5de04771f9d9b8c9",
    "common_cell_reference": "05b852a54341b927053c09c3a57967f1edc2f3dd4e06a3647fe1a8f608cb159d",
    "common_cell_coverage_extension": "bcfe72e2ee5b218a6d987f00e191451d51d633b10e49bc057dc1a4a456ce5140",
    "prepared_geometry_extension": "37d7ca5f2317c130865307cc8527476e6f6bdf9127765c2a95d6503d211cd729",
    "native_periodic_geometry": "4d1165905829da719742666cd78e2ceed2f8592c5ecc786b55b9aa7f6dbaf5d1",
})
RECIPE_IDS = MappingProxyType({2: "BI2_NATIVE45", 3: "BI3_NATIVE45_VOXEL",
    4: "BI4_TRANSITION96", 5: "BI5_TRANSITION127", 6: "BI6_NATIVE45",
    8: "BI8_TRANSITION108", 9: "BI9_TRANSITION95", 10: "BI10_TRANSITION127"})
EXPECTED_INTERVALS = MappingProxyType({2: 45, 3: 45, 4: 96, 5: 127, 6: 45, 8: 108, 9: 95, 10: 127})


def require(ok, message):
    if not ok:
        raise ValueError(message)


def profile(pitch_um):
    require(type(pitch_um) is int and 2 <= pitch_um <= 10, "integer BI pitch2..10 required")
    return dict(pitch_um=pitch_um, diameter_um=pitch_um / 2, source_si_um=37.,
        cell_um=math.lcm(pitch_um, 6), mesh_recipe=RECIPE_IDS.get(pitch_um),
        implementation="PINNED_SEED_CONSTRUCTOR" if pitch_um != 7 else "BLOCKED_MESH_RECIPE",
        reason=None if pitch_um != 7 else "Exact body-transition minimum141 exceeds frozen maximum135; new mesh recipe required",
        requires_resource_owner=True, operator_admission=False, mesh_accuracy_inferred=False)


class SeedSources:
    """Compile only explicit pinned geometry sources using the owner's reader.

    read_source(binding, cap=..., readonly=False) is provided by the resource/I/O
    owner, which verifies files, mounts, byte/rate budgets and live resources.
    This adapter checks returned bytes too, but is NOT such a resource owner.
    No original module or global import namespace is modified.
    """

    def __init__(self, *, project_root, read_source, check):
        root = PurePosixPath(project_root)
        require(root.is_absolute() and str(root) == str(project_root)
                and not {"..", "#snapshot", ".snapshot"}.intersection(root.parts), "literal project root required")
        require(callable(read_source) and callable(check), "owner source reader and guard required")
        self.root, self.directory = root, root / SOURCE_RELATIVE
        self._read_source, self._check = read_source, check
        self._modules, self._cache, self._proofs = {}, {}, {}
        self._physical_bytes = 0
        check()

    def _read(self, path, expected, limit=None):
        self._check()
        path = PurePosixPath(str(path))
        if limit is None:
            limit = 16 * 1024**2 if path == self.root / SEED_RELATIVE else 256 * 1024
        if path == self.root / SEED_RELATIVE:
            require(expected == SEED_SHA and 0 < limit <= 16 * 1024**2, "exact bounded physical seed required")
        else:
            require(path.parent == self.directory and path.suffix == ".py"
                    and SOURCE_PINS.get(path.stem) == expected and 0 < limit <= 256 * 1024,
                    "source outside explicit geometry closure")
        key = str(path), expected
        if key not in self._cache:
            require(self._physical_bytes + limit <= 32 * 1024**2, "finite source cache/read budget exceeded")
            raw = self._read_source(dict(path=str(path), sha256=expected), cap=limit, readonly=False)
            self._check()
            require(type(raw) is bytes and 0 < len(raw) <= limit
                    and hashlib.sha256(raw).hexdigest() == expected, "owner returned changed/incomplete source")
            self._cache[key] = raw
            self._physical_bytes += len(raw)
            self._proofs[str(path)] = dict(sha256=expected, bytes=len(raw))
        raw = self._cache[key]
        require(len(raw) <= limit, "cached source exceeds the caller's tighter bound")
        self._check()
        return raw

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in SOURCE_PINS:
            return self.load(name)
        require(name not in ("native_monolithic_reference", "native_monolithic_assembly"),
                "FE workers/assemblers are not geometry-seed imports")
        return builtins.__import__(name, globals, locals, fromlist, level)

    def _adapt_paths(self, module, name):
        from pathlib import Path
        values = dict(R=Path(self.root), D=Path(self.directory), RT=Path(self.directory / "runtime"),
            SELF=Path(self.directory / (name + ".py")), SEED=Path(self.root / SEED_RELATIVE),
            OUTPUT=Path(self.root / "operator_night_20261021/coverage"))
        for key, value in values.items():
            if key in vars(module):
                setattr(module, key, value)

    def load(self, name):
        require(name in SOURCE_PINS, "unpinned geometry module")
        if name in self._modules:
            return self._modules[name]
        path = self.directory / (name + ".py")
        raw = self._read(path, SOURCE_PINS[name])
        module = ModuleType("_operator_seed_" + name)
        module.__file__ = str(path)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": self._import}
        self._modules[name] = module
        try:
            exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
            # Operational path bindings only; original numerical functions and
            # their explicit mesh/card/phase/rank gates stay byte-for-byte code.
            self._adapt_paths(module, name)
            if name == "bi_pitch_geometry":
                module.read_source = self._read
                original_load = module.PrivateSources.load
                def owned_private_load(instance, child_name):
                    require(child_name in ("prepared_geometry_extension", "native_periodic_geometry"),
                            "geometry phase cannot load an FE worker")
                    child = original_load(instance, child_name)
                    self._adapt_paths(child, child_name)
                    if child_name == "prepared_geometry_extension":
                        child.bounded_read = self._read
                    return child
                module.PrivateSources.load = owned_private_load
            if name == "prepared_geometry_extension":
                module.bounded_read = self._read
            self._check()
            return module
        except BaseException:
            self._modules.pop(name, None)
            raise

    def construct(self, api, pitch_um):
        """Build one real Si37 geometry only after owner admission; no FE/Cu."""
        info = profile(pitch_um)
        require(info["implementation"] == "PINNED_SEED_CONSTRUCTOR", info["reason"])
        self._check()
        if pitch_um == 2:
            deck, basis, proof = self.load("prepared_geometry_extension").prepare(37.)
        elif pitch_um == 3:
            deck, basis, proof = self.load("bi3_coarse_candidate").construct()
        elif pitch_um == 6:
            deck, basis, proof = self.load("bi_pitch_geometry").construct()
        else:
            geometry = self.load("common_cell_geometry")
            if pitch_um == 4:
                deck, basis, proof = self.load("common_cell_reference").construct(geometry)
            else:
                deck, basis, proof = self.load("common_cell_coverage_extension").construct(geometry, pitch_um, "coarse")
        self._check()
        require(type(basis["coordinates_mm"]) is api.np.ndarray
                and isinstance(basis["P"], api.sparse.csr_matrix), "seed/API numerical library identity differs")
        n = EXPECTED_INTERVALS[pitch_um]
        require(basis["proof"]["Cartesian_intervals"] == [n, 44, n], "frozen seed mesh/profile dimensions differ")
        xyz = basis["coordinates_mm"]
        require(all(abs(float(api.np.ptp(xyz[:, axis])) * 1000 - info["cell_um"]) < 1e-10 for axis in (0, 2))
                and abs(float(api.np.ptp(xyz[:, 1])) * 1000 - 48.5) < 1e-10,
                "actual pitch-seed cell or Si37 height differs")
        return SimpleNamespace(deck=deck, basis=basis, proof=proof, profile=info,
            source_closure=dict(self._proofs), source_cache_bytes=self._physical_bytes,
            FE_assembly_performed=False, Cu_updates=0, operator_admission=False,
            resource_admission_performed_here=False)
