from __future__ import annotations

"""
D2D room-temperature stack-bow model.

This module is separate from the existing D2W warpage model.  It uses the same
multilayer curvature equation, but with no thermal load and no anneal state:
all die layers are flattened/bonded at room temperature, then released at room
temperature.
"""

import pandas as pd

from warpage_yield_calculator import compute_total_stack_warpage


def compute_d2d_roomtemp_stack_bow(
    layer_df: pd.DataFrame,
    *,
    L_m: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return released D2D room-temperature stack bow for a bonded stack."""

    summary, layer_out = compute_total_stack_warpage(layer_df, DeltaT_K=0.0, L_m=L_m)
    out = dict(summary)
    out["roomtemp_stack_bow_um"] = float(summary["signed_W_um"])
    out["roomtemp_abs_stack_bow_um"] = float(abs(summary["signed_W_um"]))
    out["model_stage"] = "roomtemp_released"
    return out, layer_out
