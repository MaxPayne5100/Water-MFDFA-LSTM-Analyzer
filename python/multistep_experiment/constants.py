"""Shared, pre-registered constants for the multi-step experiment."""

BASE_FEATURE_COLUMNS = (
    "H",
    "Delta_H",
    "Delta_alpha",
    "Alpha_0",
    "Delta_f_alpha",
    "B",
    "A",
    "beta",
    "beta2",
    "Mean",
    "Volatility",
    "Skew",
    "Slope",
    "Kurtosis",
    "Entropy",
    "Width",
)

SPECTRUM_TARGET_COLUMNS = (
    "Spectrum_Alpha_Width",
    "Spectrum_Alpha_0",
    "Spectrum_Delta_f_alpha",
    "Spectrum_B",
    "Spectrum_F_Height",
    "Spectrum_A",
    "Spectrum_Beta",
    "Spectrum_Beta2",
    "Spectrum_Eps",
)

SPECTRUM_LOG_COLUMNS = (
    "Spectrum_Alpha_Width",
    "Spectrum_F_Height",
    "Spectrum_A",
    "Spectrum_Beta",
    "Spectrum_Beta2",
)

VALUE_ONLY = "value_only"
VALUE_FRACTIONAL = "value_fractional"
VALUE_FRACTIONAL_SPECTRUM = "value_fractional_spectrum"

MODEL_VARIANTS = (VALUE_ONLY, VALUE_FRACTIONAL, VALUE_FRACTIONAL_SPECTRUM)

# Historical experiment artifacts and older callers used the auxiliary-task
# name as the whole-model name. Keep these aliases readable, but always emit
# the canonical objective-bundle names for new runs.
LEGACY_VARIANT_ALIASES = {
    "value": VALUE_ONLY,
    "fractional": VALUE_FRACTIONAL,
    "spectrum": VALUE_FRACTIONAL_SPECTRUM,
}


def canonical_variant(variant: str) -> str:
    """Map an accepted historical variant name to its canonical bundle name."""
    return LEGACY_VARIANT_ALIASES.get(variant, variant)


def legacy_variant_name(variant: str) -> str | None:
    """Return the historical name corresponding to a canonical variant."""
    return next(
        (
            legacy
            for legacy, canonical in LEGACY_VARIANT_ALIASES.items()
            if canonical == variant
        ),
        None,
    )


SEEDS = (7, 21, 42, 84, 123)
GRADNORM_ALPHA = 1.5
FRACTIONAL_ENERGY_THRESHOLD = 0.995
FRACTIONAL_MIN_TRUNCATION = 1
FRACTIONAL_MAX_TRUNCATION = 100
D_MIN = -0.45
D_MAX = 0.45
