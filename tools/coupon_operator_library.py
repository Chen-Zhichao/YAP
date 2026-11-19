"""Stable import path for the geometry-independent numerical operator catalog.

Existing ``coupon:BI*_Prepared*`` selectors remain data aliases. No historical
Python loader or source archive is imported by an assembly simulation.
"""

from .operators.catalog import load_prepared_coupon_library, prepared_operator_request
from .operators.runtime import (
    CouponOperatorLibrary, configure_coupon_runtime, _compiled_material,
)

__all__ = ("CouponOperatorLibrary", "configure_coupon_runtime",
           "load_prepared_coupon_library", "prepared_operator_request")
