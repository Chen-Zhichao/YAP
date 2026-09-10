"""Shared failure-mechanism policy for the W2W yield flow."""

FAILURE_MECHANISMS = ("overlay", "particle", "mechanical", "ESD", "warpage")

# ESD and final-warpage yield are temporarily excluded from assembly yield.
# Warpage may still be sampled internally to generate overlay-error inputs.
DISABLED_YIELD_MECHANISMS = frozenset({"ESD", "warpage"})


def requested_failure_mechanisms(input_args):
    raw = input_args.get("mechanism_filter", "all")
    if raw is None:
        requested = []
    elif isinstance(raw, (set, list, tuple)):
        requested = [str(item).strip() for item in raw]
    else:
        requested = [item.strip() for item in str(raw).split(",")]
    requested = [item for item in requested if item]

    if not requested or any(item.lower() == "all" for item in requested):
        return set(FAILURE_MECHANISMS)

    canonical = {item.lower(): item for item in FAILURE_MECHANISMS}
    result = set()
    for item in requested:
        key = item.lower()
        if key not in canonical:
            raise ValueError(
                f"Unknown mechanism_filter '{item}'. "
                f"Valid mechanisms: all, {', '.join(FAILURE_MECHANISMS)}."
            )
        result.add(canonical[key])
    return result


def active_failure_mechanisms(input_args):
    return requested_failure_mechanisms(input_args) - DISABLED_YIELD_MECHANISMS


def disabled_requested_mechanisms(input_args):
    return requested_failure_mechanisms(input_args) & DISABLED_YIELD_MECHANISMS
