#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 20, 2026

import numpy as np
import re
import hashlib
from scipy.special import ndtr
from debond import debond_dishing_bounds_calculator, debond_dishing_intervals_from_coords


_PG_NET_RE = re.compile(r"(^|_)(vdd|vss|vpp|vddq|vddql|gnd|vcc)($|_)", re.IGNORECASE)
_MECHANICAL_MODEL_CACHE = {}


def _is_non_signal_shared_net(net: str) -> bool:
    lowered = str(net).lower()
    return "dummy" in lowered or _PG_NET_RE.search(str(net)) is not None


def _mechanical_model_cache_key(cfg, pad_bitmap_collection):
    hasher = hashlib.sha1()
    for key in (
        "CRITICAL_PAD_BITMAP",
        "REDUNDANT_PAD_BITMAP",
        "redundant_group_id_per_pad",
        "redundant_tolerated_mechanical_failures",
        "redundant_tolerated_esd_failures",
    ):
        if key not in pad_bitmap_collection:
            continue
        arr = np.asarray(pad_bitmap_collection[key])
        hasher.update(str(arr.shape).encode("utf-8"))
        if arr.dtype == np.bool_:
            hasher.update(np.packbits(arr.reshape(-1)).tobytes())
        else:
            hasher.update(np.ascontiguousarray(arr).tobytes())

    for key in (
        "TOP_DISH_MEAN_nm",
        "TOP_DISH_STD_nm",
        "BOT_DISH_MEAN_nm",
        "BOT_DISH_STD_nm",
        "PITCH_r_um",
        "PITCH_c_um",
        "PAD_ARR_ROW",
        "PAD_ARR_COL",
    ):
        hasher.update(f"{key}={getattr(cfg, key, None)};".encode("utf-8"))

    return hasher.hexdigest()


# =====================================================================
# Gauss-Hermite quadrature nodes/weights mapped to N(0,1)
# =====================================================================
def _gh_nodes_weights(n: int):
    """Return (z, w) for N(0,1) Gauss-Hermite quadrature with *n* points."""
    x, w = np.polynomial.hermite.hermgauss(n)
    z = np.sqrt(2.0) * x
    w_norm = w / np.sqrt(np.pi)
    return z, w_norm


# =====================================================================
# Semi-analytic Cu-recess die yield with 3-level spatial correlation
# (Double Gauss–Hermite, log-domain for numerical stability)
#
#   Y = E_{G_L}[ Π_b  E_{G_T,b}[ Π_{i∈b} p_i(G_L, G_T,b) ] ]   (Eq. 13)
#
# Structure follows modeled_gh() in simulator_main.ipynb:
#   outer loop over G_L  →  inner loop over G_T  →  vectorised over pads
# Optimisations vs. naïve (N, nT) broadcast:
#   1. Precompute standardised z-scores  (avoid repeated (b-mu)/σ_ε)
#   2. Skip trivially-safe pads  (p ≈ 1 for all GH nodes → log p ≈ 0)
#   3. Inner loop over G_T nodes with (n_active,) arrays + reduceat
#      (avoids allocating the huge (N, nT) matrix)
# =====================================================================
def cu_recess_die_yield_spatial(
    mu: float,
    a: np.ndarray,
    b: np.ndarray,
    sigma_L: float,
    sigma_T: float,
    sigma_eps: float,
    block_indices: np.ndarray,
    n_gh_outer: int = 40,
    n_gh_inner: int = 40,
    g: float = None,
) -> float:
    r"""
    Compute Cu-recess die yield under a 3-level spatial-correlation
    model using Gauss–Hermite quadrature.

    When **g is None** (default):
        Full Eq. 13 — double GH over both G_L and G_T.

    When **g is given** (float):
        G_L = g is fixed; only single GH over G_T remains:

        Y(g) = Π_b  E_{G_{T,b}}[ Π_{i∈b} p_i(g, G_{T,b}) ]

        This is ~n_gh_outer× faster.

    Parameters
    ----------
    mu        : mean Cu height (nm)
    a, b      : per-pad lower/upper survival bounds, shape (N,)
    sigma_L   : std-dev of die-level (lot/wafer) random effect
    sigma_T   : std-dev of block-level (tile) random effect
    sigma_eps : std-dev of pad-level (residual) random effect
    block_indices : block assignment per pad, shape (N,), values in [0, B-1]
    n_gh_outer, n_gh_inner : GH quadrature points for G_L / G_T
    g         : if not None, fixed realisation of G_L (skip outer integral)

    Returns
    -------
    Y : float, estimated die yield in [0, 1]
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    block_indices = np.asarray(block_indices, dtype=np.intp)
    N = a.size

    # ── Sort all pads by block so reduceat works ──
    order    = np.argsort(block_indices, kind='mergesort')
    a_s      = a[order]
    b_s      = b[order]
    bi_s     = block_indices[order]
    _, blk_start = np.unique(bi_s, return_index=True)
    B = blk_start.size

    zT, wT = _gh_nodes_weights(n_gh_inner)
    inv_sE = 1.0 / sigma_eps if sigma_eps > 0 else 0.0

    # ── Pre-allocate work buffers ──
    _p = np.empty(N, dtype=np.float64)
    _q = np.empty(N, dtype=np.float64)
    log_blk = np.empty((B, n_gh_inner), dtype=np.float64)

    # ------------------------------------------------------------------
    # Build list of g-values to evaluate
    # ------------------------------------------------------------------
    if g is not None:
        g_vals = [float(g)]
        g_weights = None
    else:
        zG, wG = _gh_nodes_weights(n_gh_outer)
        g_vals = zG
        g_weights = wG

    log_outer = np.empty(len(g_vals), dtype=np.float64)

    for k, g_val in enumerate(g_vals):
        mu_g = mu + sigma_L * g_val

        for j in range(n_gh_inner):
            mean_j = mu_g + sigma_T * zT[j]

            if sigma_eps > 0:
                np.subtract(b_s, mean_j, out=_p)
                _p *= inv_sE
                ndtr(_p, out=_p)
                np.subtract(a_s, mean_j, out=_q)
                _q *= inv_sE
                ndtr(_q, out=_q)
                np.subtract(_p, _q, out=_p)
            else:
                _p[:] = np.where((mean_j >= a_s) & (mean_j <= b_s), 1.0, 0.0)

            np.clip(_p, 1e-300, 1.0, out=_p)
            np.log(_p, out=_p)
            log_blk[:, j] = np.add.reduceat(_p, blk_start)

        # Per-block logsumexp over T nodes
        mx = log_blk.max(axis=1, keepdims=True)
        log_E = mx[:, 0] + np.log(
            np.sum(wT[None, :] * np.exp(log_blk - mx), axis=1)
        )
        log_outer[k] = log_E.sum()

    # ------------------------------------------------------------------
    if g is not None:
        # Fast path: single g
        return float(np.clip(np.exp(log_outer[0]), 0.0, 1.0))

    # Full path: logsumexp over G_L
    mx = log_outer.max()
    Y = float(np.sum(g_weights * np.exp(log_outer - mx)) * np.exp(mx))
    return float(np.clip(Y, 0.0, 1.0))


# =====================================================================
# Helper: assign pads to spatial blocks on a regular grid
# =====================================================================
def assign_pads_to_blocks(
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    block_size_r: int,
    block_size_c: int,
) -> np.ndarray:
    """
    Partition an (PAD_ARR_ROW x PAD_ARR_COL) pad array into rectangular
    tiles of *block_size_r x block_size_c* pads and return a 1-D block-index
    array aligned with the given (pad_rows, pad_cols) coordinates.

    Parameters
    ----------
    pad_rows, pad_cols : 1-D int arrays, shape (N_valid,)
        Row / column indices of the valid (non-dummy) pads.
    PAD_ARR_ROW, PAD_ARR_COL : int
        Full pad-array dimensions.
    block_size_r, block_size_c : int
        Tile side lengths **in pads** (e.g. 10 means 10×10 pad tiles).

    Returns
    -------
    block_indices : 1-D int array, shape (N_valid,)
    """
    pad_rows, pad_cols = np.meshgrid(np.arange(PAD_ARR_ROW), np.arange(PAD_ARR_COL), indexing='ij')
    pad_rows = pad_rows.ravel()
    pad_cols = pad_cols.ravel()
    block_r = pad_rows // block_size_r
    block_c = pad_cols // block_size_c
    return block_r * int(np.ceil(PAD_ARR_COL / block_size_c)) + block_c


def block_indices_from_flat_indices(
    flat_indices: np.ndarray,
    pad_arr_col: int,
    block_size_r: int,
    block_size_c: int,
) -> np.ndarray:
    flat_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    pad_rows = flat_indices // int(pad_arr_col)
    pad_cols = flat_indices % int(pad_arr_col)
    block_rows = pad_rows // int(block_size_r)
    block_cols = pad_cols // int(block_size_c)
    num_block_cols = int(np.ceil(int(pad_arr_col) / int(block_size_c)))
    return (block_rows * num_block_cols + block_cols).astype(np.int64)


def _poisson_binomial_at_most_k_from_pass_probs(pass_probs, tolerated_failures):
    pass_probs = np.asarray(pass_probs, dtype=np.float64).reshape(-1)
    tolerated_failures = int(tolerated_failures)

    if pass_probs.size == 0:
        return 1.0
    if tolerated_failures < 0:
        return 0.0
    if tolerated_failures >= pass_probs.size:
        return 1.0

    fail_probs = np.clip(1.0 - pass_probs, 0.0, 1.0)
    pmf = np.zeros(tolerated_failures + 1, dtype=np.float64)
    pmf[0] = 1.0

    for fail_prob in fail_probs:
        pass_prob = 1.0 - fail_prob
        next_pmf = pmf * pass_prob
        next_pmf[1:] += pmf[:-1] * fail_prob
        pmf = next_pmf

    return float(np.clip(np.sum(pmf), 0.0, 1.0))


def _poisson_binomial_pmf_truncated_from_pass_probs(pass_probs, tolerated_failures):
    pass_probs = np.asarray(pass_probs, dtype=np.float64).reshape(-1)
    tolerated_failures = int(tolerated_failures)
    pmf = np.zeros(tolerated_failures + 1, dtype=np.float64)
    pmf[0] = 1.0

    for pass_prob in np.clip(pass_probs, 0.0, 1.0):
        fail_prob = 1.0 - pass_prob
        next_pmf = pmf * pass_prob
        next_pmf[1:] += pmf[:-1] * fail_prob
        pmf = next_pmf

    return pmf


def cu_recess_redundant_group_yield_spatial(
    mu: float,
    a: np.ndarray,
    b: np.ndarray,
    sigma_L: float,
    sigma_T: float,
    sigma_eps: float,
    block_indices: np.ndarray,
    tolerated_failures: int,
    n_gh_outer: int = 40,
    n_gh_inner: int = 40,
) -> float:
    """
    Probability that a redundant pad group has no more than
    ``tolerated_failures`` failed pads under the same 3-level Cu-recess
    correlation model used by critical pads.

    This integrates the shared die-level and block-level components. It is
    exact for one redundant group under this model.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    block_indices = np.asarray(block_indices, dtype=np.int64).reshape(-1)
    tolerated_failures = int(tolerated_failures)

    if a.size == 0:
        return 1.0
    if tolerated_failures < 0:
        return 0.0
    if tolerated_failures >= a.size:
        return 1.0
    if tolerated_failures == 0:
        return cu_recess_die_yield_spatial(
            mu=mu,
            a=a,
            b=b,
            sigma_L=sigma_L,
            sigma_T=sigma_T,
            sigma_eps=sigma_eps,
            block_indices=block_indices,
            n_gh_outer=n_gh_outer,
            n_gh_inner=n_gh_inner,
        )

    order = np.argsort(block_indices, kind="mergesort")
    a_s = a[order]
    b_s = b[order]
    block_s = block_indices[order]
    _, block_starts = np.unique(block_s, return_index=True)
    block_stops = np.r_[block_starts[1:], a_s.size]

    zG, wG = _gh_nodes_weights(n_gh_outer)
    zT, wT = _gh_nodes_weights(n_gh_inner)

    if tolerated_failures == 1:
        outer_vals = np.zeros(n_gh_outer, dtype=np.float64)
        for g_idx, g_val in enumerate(zG):
            mu_g = mu + sigma_L * g_val
            block_p0 = []
            block_p1 = []

            for start, stop in zip(block_starts, block_stops):
                a_block = a_s[start:stop]
                b_block = b_s[start:stop]
                mean_t = mu_g + sigma_T * zT[:, None]

                if sigma_eps > 0:
                    pass_probs = ndtr((b_block[None, :] - mean_t) / sigma_eps) - ndtr(
                        (a_block[None, :] - mean_t) / sigma_eps
                    )
                else:
                    pass_probs = (
                        (mean_t >= a_block[None, :])
                        & (mean_t <= b_block[None, :])
                    ).astype(np.float64)
                pass_probs = np.clip(pass_probs, 1e-300, 1.0)
                fail_probs = 1.0 - pass_probs
                p0_t = np.prod(pass_probs, axis=1)
                p1_t = p0_t * np.sum(fail_probs / pass_probs, axis=1)
                block_p0.append(float(np.sum(wT * p0_t)))
                block_p1.append(float(np.sum(wT * p1_t)))

            block_p0 = np.clip(np.asarray(block_p0, dtype=np.float64), 1e-300, 1.0)
            block_p1 = np.clip(np.asarray(block_p1, dtype=np.float64), 0.0, 1.0)
            total_p0 = float(np.prod(block_p0))
            outer_vals[g_idx] = total_p0 * (
                1.0 + float(np.sum(block_p1 / block_p0))
            )

        return float(np.clip(np.sum(wG * outer_vals), 0.0, 1.0))

    outer_vals = np.zeros(n_gh_outer, dtype=np.float64)

    for g_idx, g_val in enumerate(zG):
        total_pmf = np.zeros(tolerated_failures + 1, dtype=np.float64)
        total_pmf[0] = 1.0
        mu_g = mu + sigma_L * g_val

        for start, stop in zip(block_starts, block_stops):
            a_block = a_s[start:stop]
            b_block = b_s[start:stop]
            block_pmf = np.zeros(tolerated_failures + 1, dtype=np.float64)

            for t_val, t_weight in zip(zT, wT):
                mean_j = mu_g + sigma_T * t_val
                if sigma_eps > 0:
                    pass_probs = ndtr((b_block - mean_j) / sigma_eps) - ndtr(
                        (a_block - mean_j) / sigma_eps
                    )
                else:
                    pass_probs = ((mean_j >= a_block) & (mean_j <= b_block)).astype(
                        np.float64
                    )
                block_pmf += t_weight * _poisson_binomial_pmf_truncated_from_pass_probs(
                    pass_probs,
                    tolerated_failures,
                )

            total_pmf = np.convolve(total_pmf, block_pmf)[: tolerated_failures + 1]

        outer_vals[g_idx] = np.sum(total_pmf)

    return float(np.clip(np.sum(wG * outer_vals), 0.0, 1.0))


def _redundant_group_yield_spatial(
    *,
    mu,
    lower_limits_by_selected_idx,
    upper_limits_by_selected_idx,
    selected_flat_idx,
    cfg,
    sigma_L,
    sigma_T,
    sigma_eps,
    pad_bitmap_collection,
    block_size_r,
    block_size_c,
):
    redundant_net_to_1d_physical_mask = pad_bitmap_collection.get(
        "redundant_net_to_1d_physical_mask",
        {},
    )
    criticality_info = pad_bitmap_collection.get("criticality_info", {})
    if not redundant_net_to_1d_physical_mask:
        return 1.0

    representative_groups = {}
    for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
        if _is_non_signal_shared_net(redundant_net):
            continue

        physical_mask = np.asarray(physical_mask, dtype=np.int64).reshape(-1)
        physical_mask = physical_mask[physical_mask >= 0]
        if physical_mask.size == 0:
            continue
        if redundant_net not in criticality_info:
            raise KeyError(
                f"Missing criticality info for redundant net '{redundant_net}'."
            )

        selected_pos = np.searchsorted(selected_flat_idx, physical_mask)
        in_range = selected_pos < selected_flat_idx.size
        matches = np.zeros_like(in_range, dtype=bool)
        matches[in_range] = selected_flat_idx[selected_pos[in_range]] == physical_mask[in_range]
        if not np.all(matches):
            missing = physical_mask[~matches][:5]
            raise ValueError(
                f"Redundant net '{redundant_net}' references pads not present in "
                f"the selected Cu-yield pad set, e.g. {missing.tolist()}."
            )

        tolerated_failures = int(
            criticality_info[redundant_net]["tolerated_mechanical_failures"]
        )

        lower = lower_limits_by_selected_idx[selected_pos]
        upper = upper_limits_by_selected_idx[selected_pos]
        block_indices = block_indices_from_flat_indices(
            physical_mask,
            cfg.PAD_ARR_COL,
            block_size_r,
            block_size_c,
        )
        _, compact_blocks = np.unique(block_indices, return_inverse=True)
        order = np.lexsort((upper, lower, compact_blocks))
        key = (
            tolerated_failures,
            tuple(np.round(lower[order], 9)),
            tuple(np.round(upper[order], 9)),
            tuple(compact_blocks[order].astype(np.int16, copy=False)),
        )
        if key in representative_groups:
            representative_groups[key]["count"] += 1
        else:
            representative_groups[key] = {
                "count": 1,
                "lower": lower[order],
                "upper": upper[order],
                "blocks": compact_blocks[order],
                "tolerated_failures": tolerated_failures,
            }

    redundant_yield = 1.0
    for group in representative_groups.values():
        group_yield = cu_recess_redundant_group_yield_spatial(
            mu=mu,
            a=group["lower"],
            b=group["upper"],
            sigma_L=sigma_L,
            sigma_T=sigma_T,
            sigma_eps=sigma_eps,
            block_indices=group["blocks"],
            tolerated_failures=group["tolerated_failures"],
        )
        redundant_yield *= group_yield ** group["count"]
        if redundant_yield <= 0.0:
            return 0.0

    return float(np.clip(redundant_yield, 0.0, 1.0))




# def stack_stress_yield_calculator_0(
#         cfg_dict: dict,
#         waf_stack,
#         pad_bitmap_collection_dict: dict,
#         valid_pad_mask_dict: dict,
# ):
#     for interface_name, cfg in cfg_dict.items():
#         interface = waf_stack.interfaces.interface_dict[interface_name]
#         pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
#         valid_pad_mask = valid_pad_mask_dict[interface_name]

#         # Extract the necessary parameters for Cu expansion yield calculation
#         TOP_DISH_MEAN_nm, TOP_DISH_STD_nm = cfg.TOP_DISH_MEAN_nm, cfg.TOP_DISH_STD_nm
#         BOT_DISH_MEAN_nm, BOT_DISH_STD_nm = cfg.BOT_DISH_MEAN_nm, cfg.BOT_DISH_STD_nm
#         CRITICAL_PAD_MASK = pad_bitmap_collection['CRITICAL_PAD_BITMAP'].flatten()
#         redundant_net_to_1d_physical_mask = pad_bitmap_collection['redundant_net_to_1d_physical_mask']


#         stress_yield_list = []

#         for die_ind, die in enumerate(interface.die_list):
#             die_pad_coords = interface.base_pad_coords + die.die_center
#             valid_die_pad_coords = die_pad_coords[valid_pad_mask.flatten() == 1]
#             start_time = time.time()
#             valid_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
#             # print("Dishing bound calculation time for die {}: {:.2f} seconds".format(die_ind, time.time() - start_time))
#             upper_limits_valid_pads = - valid_dishing_bound_array[:, 0] * 2 # - upper limits of the sum of top and bottom Cu heights
#             lower_limits_valid_pads = - valid_dishing_bound_array[:, 1] * 2 # - lower limits of the sum of top and bottom Cu heights
#             pos_valid_pads = norm.cdf(upper_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
#                     norm.cdf(lower_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
#             # Critical yield is the pos of the critical pads multiplied together
#             stress_yield_critical_pads = np.prod(pos_valid_pads[CRITICAL_PAD_MASK == 1])
#             stress_yield_redundant_nets = 1.0
#             for redundant_net, physical_pad_indices in redundant_net_to_1d_physical_mask.items():
#                 num_replicas = len(physical_pad_indices)
#                 stress_yield_redundant_nets *= 1 - (1 - np.prod(pos_valid_pads[physical_pad_indices])) ** num_replicas
#             stress_yield = stress_yield_critical_pads * stress_yield_redundant_nets
#             stress_yield_list.append(stress_yield)

#             # break
            
#         # Update the die yield list for this interface in the wafer stack
#         waf_stack.die_yield_list_per_interface_dict[interface_name]['mechanical'] = np.array(stress_yield_list)




def _vertex_distance_key(die, radial_lut_array, decimals: int = 1):
    """
    Build a hashable cache key from the dishing-LUT values that a die's
    four vertices map to.

    The radial LUT columns are [r_um, D_sio2_nm, D_cu_nm].
    For each vertex we look up (D_sio2, D_cu) by interpolating into the
    LUT, round to *decimals* decimal places, and pack min/max into a tuple.
    Two dies that produce the same key have identical dishing bounds on
    all their pads (because dishing is radially monotone and the vertex
    radii bracket all pad radii inside the die).
    """
    lut_r    = radial_lut_array[:, 0]
    lut_dishing_ub = radial_lut_array[:, 1]
    lut_dishing_lb   = radial_lut_array[:, 2]

    verts = np.asarray(die.vertices_coords, dtype=np.float64)   # (4, 2)
    v_radii = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)          # (4,)

    d_ub = np.interp(v_radii, lut_r, lut_dishing_ub)  # (4,)
    d_lb   = np.interp(v_radii, lut_r, lut_dishing_lb)  # (4,)

    # Round and build immutable key from the range seen across vertices
    return (
        round(float(d_ub.min()), decimals),
        round(float(d_ub.max()), decimals),
        round(float(d_lb.min()),   decimals),
        round(float(d_lb.max()),   decimals),
    )


def stack_stress_yield_calculator(
        cfg_dict: dict,
        waf_stack,
):
    """
    Cu-expansion (mechanical) yield per die.

    W2W now uses the same die-level Cu-recess model as D2W for this term:
    the dishing survival interval is a function of the local interface/pad
    geometry only, not the die's wafer-center position. Therefore each
    interface has one representative die-level mechanical yield, broadcast
    to every die on the wafer.
    """
    for interface_name, cfg in cfg_dict.items():
        cfg.num_dies_per_wafer = waf_stack.num_dies_per_wafer
        interface = waf_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = waf_stack.interfaces.pad_bitmap_collection_dict[interface_name]
        cache_key = _mechanical_model_cache_key(cfg, pad_bitmap_collection)

        # --- Config ---
        PAD_ARR_ROW, PAD_ARR_COL = cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL
        TOP_DISH_MEAN_nm  = cfg.TOP_DISH_MEAN_nm
        TOP_DISH_STD_nm = cfg.TOP_DISH_STD_nm
        BOT_DISH_MEAN_nm  = cfg.BOT_DISH_MEAN_nm
        BOT_DISH_STD_nm = cfg.BOT_DISH_STD_nm

        block_size_r = max(1, int(PAD_ARR_ROW))
        block_size_c = max(1, int(PAD_ARR_COL))

        mu        = TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm
        sigma_L   = 0.0
        sigma_T   = 0.0
        sigma_eps = np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)

        stress_yield_array = np.full(interface.num_dies, np.nan, dtype=np.float64)
        if cache_key in _MECHANICAL_MODEL_CACHE:
            stress_yield_array[:] = _MECHANICAL_MODEL_CACHE[cache_key]
            waf_stack.die_yield_list_per_interface_dict[interface_name]['mechanical'] = stress_yield_array
            continue

        critical_pad_mask_flat = (
            pad_bitmap_collection["CRITICAL_PAD_BITMAP"].reshape(-1).astype(bool)
        )
        redundant_pad_mask_flat = (
            pad_bitmap_collection["REDUNDANT_PAD_BITMAP"].reshape(-1).astype(bool)
        )
        redundant_signal_mask_flat = np.zeros_like(redundant_pad_mask_flat)
        for redundant_net, physical_mask in pad_bitmap_collection.get(
            "redundant_net_to_1d_physical_mask",
            {},
        ).items():
            if _is_non_signal_shared_net(redundant_net):
                continue
            physical_mask = np.asarray(physical_mask, dtype=np.int64).reshape(-1)
            physical_mask = physical_mask[
                (physical_mask >= 0) & (physical_mask < redundant_signal_mask_flat.size)
            ]
            redundant_signal_mask_flat[physical_mask] = True

        selected_pad_mask_flat = critical_pad_mask_flat | redundant_signal_mask_flat
        selected_flat_idx = np.flatnonzero(selected_pad_mask_flat).astype(np.int64)
        critical_flat_idx = np.flatnonzero(critical_pad_mask_flat).astype(np.int64)
        if selected_flat_idx.size == 0:
            stress_yield_array[:] = 1.0
            waf_stack.die_yield_list_per_interface_dict[interface_name]['mechanical'] = stress_yield_array
            continue
        if interface.base_pad_coords is None:
            raise ValueError(
                f"Interface '{interface_name}' does not have pad coordinates. "
                "Cu mechanical yield modeling needs coordinates for critical "
                "and redundant pads."
            )
        base_pad_xy = np.asarray(interface.base_pad_coords, dtype=np.float64)
        selected_base_pad_xy = base_pad_xy[selected_flat_idx]
        if not np.all(np.isfinite(selected_base_pad_xy)):
            bad = selected_flat_idx[~np.all(np.isfinite(selected_base_pad_xy), axis=1)][:5]
            raise ValueError(
                f"Interface '{interface_name}' has selected critical/redundant "
                f"pads with invalid coordinates, e.g. flat indices {bad.tolist()}."
            )
        critical_pos = np.searchsorted(selected_flat_idx, critical_flat_idx)
        critical_block_idx = np.zeros(critical_flat_idx.size, dtype=np.int32)

        pad_dishing_bound_array = debond_dishing_intervals_from_coords(
            cfg,
            selected_base_pad_xy,
        )
        upper_limits = np.clip(
            -pad_dishing_bound_array[:, 0] * 2,
            a_min=None,
            a_max=0,
        )
        lower_limits = np.clip(
            -pad_dishing_bound_array[:, 1] * 2,
            a_min=None,
            a_max=0,
        )

        if critical_flat_idx.size > 0:
            critical_yield = cu_recess_die_yield_spatial(
                mu=mu,
                a=lower_limits[critical_pos],
                b=upper_limits[critical_pos],
                sigma_L=sigma_L,
                sigma_T=sigma_T,
                sigma_eps=sigma_eps,
                block_indices=critical_block_idx,
            )
        else:
            critical_yield = 1.0

        redundant_yield = _redundant_group_yield_spatial(
            mu=mu,
            lower_limits_by_selected_idx=lower_limits,
            upper_limits_by_selected_idx=upper_limits,
            selected_flat_idx=selected_flat_idx,
            cfg=cfg,
            sigma_L=sigma_L,
            sigma_T=sigma_T,
            sigma_eps=sigma_eps,
            pad_bitmap_collection=pad_bitmap_collection,
            block_size_r=block_size_r,
            block_size_c=block_size_c,
        )
        stress_die_yield = float(critical_yield * redundant_yield)
        _MECHANICAL_MODEL_CACHE[cache_key] = stress_die_yield
        stress_yield_array[:] = stress_die_yield
        waf_stack.die_yield_list_per_interface_dict[interface_name]['mechanical'] = stress_yield_array
