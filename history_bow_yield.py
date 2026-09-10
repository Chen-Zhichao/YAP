"""Pure pre-bond mechanics-to-overlay bridge for the DATE paper convention.

Lengths are micrometers, rotations radians, and magnification strains
dimensionless (not ppm). Inputs must describe the two released bodies just
before attachment, in one aligned coordinate frame. This module does not
infer mechanical history, face levers, material properties, or Gaussian bow.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


def _finite_array(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _pair(value, name):
    array = _finite_array(value, name)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly the x and y components")
    return tuple(float(x) for x in array)


@dataclass(frozen=True)
class PrebondBody:
    """An explicitly supplied, released pre-attachment body state.

    ``bow_um`` is signed center-minus-mid-edge paper bow, so
    ``w(x,y) = w0 - Bx*x**2/Lx**2 - By*y**2/Ly**2``. ``halfspan_um``
    contains HALF dimensions (wafer radius for a circular body).
    ``face_lever_um = z_face - z_stiffness_centroid`` is signed in the
    declared frame; it is never replaced with half the body thickness.
    Supply a scalar or explicit directional (ell_x,ell_y) values. A caller
    reducing coupled ABD bending to effective directional levers must
    establish their validity for this curvature state; a finite diagonal
    lever cannot represent nonzero cross-coupled strain at zero curvature.
    ``chi`` contains the dimensionless directional membrane coefficients.

    The frame must align both bodies' x/y axes and transverse signs; this
    class performs no flip or rotation. State and lever-source identifiers
    record caller provenance, not independently verified mechanical evidence.
    Non-um inputs must be converted explicitly before constructing a body.
    """

    body_id: str
    state_id: str
    frame_id: str
    bow_um: tuple[float, float]
    halfspan_um: tuple[float, float]
    face_lever_um: float | tuple[float, float]
    chi: tuple[float, float]
    face_lever_source: str
    length_unit: str = "um"

    def __post_init__(self):
        for name in ("body_id", "state_id", "frame_id", "face_lever_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty provenance identifier")
        if self.length_unit != "um":
            raise ValueError("length_unit must be 'um'; convert all lengths explicitly")
        for name in ("bow_um", "halfspan_um", "chi"):
            object.__setattr__(self, name, _pair(getattr(self, name), name))
        if min(self.halfspan_um) <= 0:
            raise ValueError("halfspan_um must be strictly positive")
        lever = _finite_array(self.face_lever_um, "face_lever_um")
        if lever.ndim == 0:
            object.__setattr__(self, "face_lever_um", float(lever))
        elif lever.shape == (2,):
            object.__setattr__(self, "face_lever_um", tuple(float(x) for x in lever))
        else:
            raise ValueError("face_lever_um must be a signed scalar or (ell_x,ell_y)")


def paired_magnification(plus: PrebondBody, minus: PrebondBody):
    """Return dimensionless [Ex, Ey] using paper Eq. (7), upper minus lower.

    E = [-2*(ell_plus*B_plus - ell_minus*B_minus)
         + chi_plus*B_plus**2 - chi_minus*B_minus**2] / L_half**2.

    Both bodies must use the same half-spans and aligned frame. Unequal
    domains, unaligned principal axes, and non-affine shape maps require a
    separately qualified transfer; they are not inferred here.
    """
    if not isinstance(plus, PrebondBody) or not isinstance(minus, PrebondBody):
        raise TypeError("plus and minus must be explicit PrebondBody descriptors")
    if plus.frame_id != minus.frame_id:
        raise ValueError("paired bodies must use the same aligned frame_id")
    if not np.allclose(plus.halfspan_um, minus.halfspan_um, rtol=1e-12, atol=0):
        raise ValueError("paper Eq. (7) requires matching halfspan_um")
    bp, bm = np.asarray(plus.bow_um), np.asarray(minus.bow_um)
    halfspan = np.asarray(plus.halfspan_um)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result = (-2 * (np.asarray(plus.face_lever_um) * bp - np.asarray(minus.face_lever_um) * bm)
                      + np.asarray(plus.chi) * bp**2
                      - np.asarray(minus.chi) * bm**2) / halfspan**2
        except FloatingPointError as exc:
            raise ValueError("body lengths exceed the numerical range of Eq. (7)") from exc
    return _finite_array(result, "magnification")


def affine_overlay_um(coordinates_um, magnification, *, translation_um=(0., 0.),
                      rotation_rad=0.):
    """Return (..., 2) directional affine displacement vectors in um.

    dx = Tx - rotation*y + Ex*x; dy = Ty + rotation*x + Ey*y.
    Pair-valued inputs have final axis (x,y); leading axes follow NumPy
    broadcasting. For shared draws over sites/pads, for example, use
    coordinates (site,pad,2), magnification/translation (draw,1,1,2), and
    rotation (draw,1,1). No random draws or ppm conversion occur here.
    """
    xy = _finite_array(coordinates_um, "coordinates_um")
    strain = _finite_array(magnification, "magnification")
    shift = _finite_array(translation_um, "translation_um")
    rotation = _finite_array(rotation_rad, "rotation_rad")
    for name, value in (("coordinates_um", xy), ("magnification", strain),
                        ("translation_um", shift)):
        if value.ndim == 0 or value.shape[-1] != 2:
            raise ValueError(f"{name} must have final axis (x,y)")
    with np.errstate(over="raise", invalid="raise"):
        try:
            x, y, ex, ey, tx, ty, angle = np.broadcast_arrays(
                xy[..., 0], xy[..., 1], strain[..., 0], strain[..., 1],
                shift[..., 0], shift[..., 1], rotation)
            result = np.stack((tx - angle*y + ex*x, ty + angle*x + ey*y), axis=-1)
        except (ValueError, FloatingPointError) as exc:
            raise ValueError("affine inputs must broadcast and remain finite") from exc
    return _finite_array(result, "overlay_um")


def residual_interval_probability(delta_um, threshold_um, sigma_um):
    """P(-threshold <= delta + Z <= threshold), Z ~ N(0, sigma**2).

    This follows the DATE spatial study's signed-scalar residual convention,
    using the same symmetric CDF evaluation. It is NOT a 2D radial Gaussian
    probability. Inputs broadcast; zero sigma uses the closed deterministic
    interval. Threshold and sigma must be nonnegative.
    """
    delta, threshold, sigma = np.broadcast_arrays(
        _finite_array(delta_um, "delta_um"),
        _finite_array(threshold_um, "threshold_um"),
        _finite_array(sigma_um, "sigma_um"))
    if np.any(threshold < 0) or np.any(sigma < 0):
        raise ValueError("threshold_um and sigma_um must be nonnegative")
    delta = np.abs(delta)
    positive = sigma > 0
    # Use the negative-tail representation instead of subtracting near-one CDFs.
    with np.errstate(over="ignore", divide="ignore", invalid="raise"):
        upper = np.divide(threshold - delta, sigma,
                          out=np.zeros_like(delta), where=positive)
        lower = (-np.divide(threshold, sigma, out=np.zeros_like(delta), where=positive)
                 - np.divide(delta, sigma, out=np.zeros_like(delta), where=positive))
    probability = np.clip(ndtr(upper) - ndtr(lower), 0., 1.)
    return np.where(positive, probability, (delta <= threshold).astype(float))


@dataclass(frozen=True)
class WorstPadSurvival:
    probability: np.ndarray
    pad_index: np.ndarray
    misalignment_um: np.ndarray


def worst_critical_pad_survival(overlay_um, *, critical_pad_mask, threshold_um, sigma_um):
    """Evaluate the scalar interval surrogate at the worst critical pad.

    ``overlay_um`` has shape (..., pad, 2). The explicit 1D Boolean mask
    selects critical pads, shared across leading site/draw axes. The worst
    pad maximizes systematic vector magnitude; ties select its first input
    index. Z is a signed scalar along that selected displacement direction
    (any fixed direction when displacement is zero). A common threshold
    and scalar sigma apply to the pads within each leading-axis entry.

    This is NOT a 2D radial-noise probability, a joint all-pad survival
    probability, or a guarantee that every pad survives. Invalid values on
    selected critical pads raise; excluded pads are not silently promoted.
    """
    overlay = np.asarray(overlay_um, dtype=float)
    mask = np.asarray(critical_pad_mask)
    if overlay.ndim < 2 or overlay.shape[-1] != 2:
        raise ValueError("overlay_um must have shape (..., pad, 2)")
    if mask.dtype != np.bool_ or mask.shape != (overlay.shape[-2],) or not mask.any():
        raise ValueError("require a nonempty explicit Boolean critical_pad_mask")
    active = _finite_array(overlay[..., mask, :], "critical-pad overlay_um")
    with np.errstate(over="raise"):
        try:
            magnitude = np.hypot(active[..., 0], active[..., 1])
        except FloatingPointError as exc:
            raise ValueError("critical-pad magnitude exceeds the numerical range") from exc
    local_index = np.argmax(magnitude, axis=-1)
    pad_index = np.flatnonzero(mask)[local_index]
    worst = np.max(magnitude, axis=-1)
    probability = residual_interval_probability(worst, threshold_um, sigma_um)
    return WorstPadSurvival(
        probability=probability,
        pad_index=np.broadcast_to(pad_index, probability.shape).copy(),
        misalignment_um=np.broadcast_to(worst, probability.shape).copy())


@dataclass(frozen=True)
class SpatialYieldSummary:
    registered_site_yield: float
    conditional_product_of_means: float
    product_of_marginal_means: float
    included_sites: int
    interfaces: int
    latent_samples: int


def compound_site_yield(probabilities, site_mask, *, latent_weights=None):
    """Compound conditional probabilities without losing shared latent draws.

    Input is (interface,site), or (latent,interface,site) with each latent
    index denoting ONE JOINT draw shared by all interfaces/sites. Interface
    failures are assumed independent conditional on that draw and site.
    Sites have equal weights. Latent weights, when supplied for 3D inputs,
    are nonnegative quadrature/sample weights normalized here to sum to one.

    Outputs distinguish three expectations:
      registered_site_yield = E_latent E_site prod_interface p
      conditional_product_of_means = E_latent prod_interface E_site p
      product_of_marginal_means = prod_interface E_latent E_site p

    The second is the independent-site/shuffle expectation while retaining
    shared process randomness. The third also removes latent correlation.
    For 2D inputs the second and third coincide. Included sites must contain
    finite probabilities in [0,1] for EVERY interface/draw. Excluded sites
    may contain NaNs; this function never changes the explicit mask.
    """
    p = np.asarray(probabilities, dtype=float)
    mask = np.asarray(site_mask)
    if p.ndim not in (2, 3) or any(size == 0 for size in p.shape):
        raise ValueError("require nonempty (interface,site) or (latent,interface,site) data")
    if mask.dtype != np.bool_ or mask.shape != (p.shape[-1],) or not mask.any():
        raise ValueError("require a nonempty explicit Boolean site_mask")
    if p.ndim == 2:
        if latent_weights is not None:
            raise ValueError("latent_weights require explicit 3D joint-latent probabilities")
        p = p[None, ...]
    values = _finite_array(p[..., mask], "included-site probabilities")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("included-site probabilities must lie in [0,1]")
    if latent_weights is None:
        weights = np.full(p.shape[0], 1. / p.shape[0])
    else:
        weights = _finite_array(latent_weights, "latent_weights")
        if weights.shape != (p.shape[0],) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("require one nonnegative latent weight per joint draw, with positive sum")
        weights = weights / np.max(weights)
        weights = weights / weights.sum()
    site_means = values.mean(axis=2)
    registered = np.prod(values, axis=1).mean(axis=1)
    unregistered = np.prod(site_means, axis=1)
    return SpatialYieldSummary(
        registered_site_yield=float(np.clip(weights @ registered, 0., 1.)),
        conditional_product_of_means=float(np.clip(weights @ unregistered, 0., 1.)),
        product_of_marginal_means=float(np.clip(np.prod(weights @ site_means), 0., 1.)),
        included_sites=int(mask.sum()), interfaces=p.shape[1], latent_samples=p.shape[0])
