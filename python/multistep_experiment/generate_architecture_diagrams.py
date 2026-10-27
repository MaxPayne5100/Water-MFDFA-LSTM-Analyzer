"""Generate architecture diagrams for the three multi-step experiment models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    from lineage import publish_atomic_output, temporary_output_path
except ImportError:  # pragma: no cover
    from ..lineage import publish_atomic_output, temporary_output_path

from .constants import (
    VALUE_FRACTIONAL,
    VALUE_FRACTIONAL_SPECTRUM,
    VALUE_ONLY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "data_lake" / "config.json"
EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "data_lake"
    / "forecasts"
    / "experiments"
    / "multistep_fractal_ablation_v1"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "data_lake" / "artifacts" / "analysis"

INK = "#17324D"
MUTED = "#5F7182"
FORWARD = "#2F6690"
ENCODER = "#D9EAF4"
DECODER = "#CFE8E2"
VALUE = "#DCEED5"
FRACTIONAL = "#FBE3B3"
SPECTRUM = "#E8DDF4"
GRADNORM = "#F4C7C3"
NEUTRAL = "#EEF2F5"
WHITE = "#FFFFFF"


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str,
    edgecolor: str = INK,
    title_size: float = 10.5,
    body_size: float = 8.4,
    linewidth: float = 1.4,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    axis.text(
        x + width / 2,
        y + height * 0.31,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=MUTED,
        linespacing=1.20,
        zorder=3,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = FORWARD,
    dashed: bool = False,
    label: str | None = None,
    connectionstyle: str = "arc3",
    label_offset: tuple[float, float] = (0.0, 0.16),
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=connectionstyle,
        zorder=1,
    )
    axis.add_patch(arrow)
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            color=color,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.2},
            zorder=4,
        )


def _load_lock(horizon: int) -> dict:
    path = EXPERIMENT_ROOT / f"horizon_{horizon}" / "locked_parameters.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _configuration_text() -> str:
    lock_10 = _load_lock(10)["parameters"]
    lock_50 = _load_lock(50)["parameters"]
    shared = (
        f"lookback={lock_10['lookback']}   •   encoder={lock_10['encoder_units']}   "
        f"•   decoder={lock_10['decoder_units']}   •   dropout={lock_10['dropout']:.1f}   "
        f"•   batch={lock_10['batch_size']}"
    )
    horizons = (
        f"H=10: top-{lock_10['top_k']} features, Adam lr={lock_10['learning_rate']:.3g}   |   "
        f"H=50: top-{lock_50['top_k']} features, Adam lr={lock_50['learning_rate']:.3g}"
    )
    return f"Locked backbone: {shared}\n{horizons}"


def _canvas(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(16, 9), dpi=160)
    figure.patch.set_facecolor(WHITE)
    axis.set_facecolor(WHITE)
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")
    axis.text(
        0.45,
        8.58,
        title,
        fontsize=22,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )
    axis.text(
        0.47,
        8.18,
        subtitle,
        fontsize=10.5,
        color=MUTED,
        ha="left",
        va="center",
    )
    _box(
        axis,
        0.45,
        7.35,
        15.1,
        0.58,
        "EXPERIMENT LOCK",
        _configuration_text(),
        facecolor=NEUTRAL,
        title_size=8.8,
        body_size=7.8,
        linewidth=0.9,
    )
    axis.text(
        0.48,
        7.05,
        "FORWARD ARCHITECTURE",
        fontsize=8.5,
        fontweight="bold",
        color=MUTED,
    )
    axis.text(
        0.48,
        4.66,
        "TRAINING OBJECTIVES",
        fontsize=8.5,
        fontweight="bold",
        color=MUTED,
    )
    axis.plot([0.45, 15.55], [4.53, 4.53], color="#D9E0E5", linewidth=1.0)
    return figure, axis


def _backbone(axis: plt.Axes, *, spectrum_head: bool = False) -> None:
    y = 5.55
    height = 1.18
    nodes = [
        (0.45, 1.95, "Feature sequence", "[batch, 10, K]\ntraining-fold scaled", ENCODER),
        (2.78, 1.85, "LSTM encoder", "64 tanh units\nfinal state vector", ENCODER),
        (5.00, 1.40, "Dropout", "rate = 0.10", NEUTRAL),
        (6.76, 1.65, "RepeatVector", "repeat state H times", NEUTRAL),
        (8.78, 1.95, "Shared decoder", "LSTM 64 tanh\nreturn sequences", DECODER),
        (11.10, 1.40, "Dropout", "rate = 0.10", NEUTRAL),
    ]
    for x, width, title, body, color in nodes:
        _box(axis, x, y, width, height, title, body, facecolor=color)
    for left, right in zip(nodes, nodes[1:]):
        _arrow(
            axis,
            (left[0] + left[1], y + height / 2),
            (right[0], y + height / 2),
        )

    if not spectrum_head:
        _box(
            axis,
            12.88,
            y,
            2.67,
            height,
            "Value head",
            "TimeDistributed Dense(1)\nŷ shape: [batch, H, 1]",
            facecolor=VALUE,
        )
        _arrow(axis, (12.50, y + height / 2), (12.88, y + height / 2))
        return

    _box(
        axis,
        13.02,
        6.05,
        2.53,
        0.90,
        "Value head",
        "TD Dense(1)  •  ŷ",
        facecolor=VALUE,
        title_size=9.8,
        body_size=8.0,
    )
    _box(
        axis,
        13.02,
        4.92,
        2.53,
        0.90,
        "Spectrum head",
        "TD Dense(9)  •  ŝ",
        facecolor=SPECTRUM,
        title_size=9.8,
        body_size=8.0,
    )
    _arrow(
        axis,
        (12.50, y + height / 2),
        (13.02, 6.50),
        connectionstyle="arc3,rad=-0.15",
    )
    _arrow(
        axis,
        (12.50, y + height / 2),
        (13.02, 5.37),
        connectionstyle="arc3,rad=0.15",
    )


def _footer(axis: plt.Axes, *, gradnorm: bool) -> None:
    axis.text(
        0.48,
        0.18,
        "Solid arrows: forward computation   •   Dashed arrows: loss supervision / parameter update"
        + ("   •   GradNorm norms are measured on the shared decoder" if gradnorm else "")
        + "   •   Primary comparison: original-scale value RMSE",
        fontsize=8.0,
        color=MUTED,
        ha="left",
        va="center",
    )


def value_only_diagram() -> plt.Figure:
    figure, axis = _canvas(
        "Model A — value_only",
        "Direct multi-step LSTM baseline • one output head • one optimization objective",
    )
    _backbone(axis)

    _box(
        axis,
        0.65,
        2.52,
        2.55,
        1.02,
        "Observed future",
        "y shape: [batch, H, 1]\noriginal target, scaled for fit",
        facecolor=NEUTRAL,
    )
    _box(
        axis,
        4.25,
        2.52,
        3.00,
        1.02,
        "Global value RMSE",
        "Lvalue = √mean[(ŷ − y)²]\nall samples × all H steps",
        facecolor=VALUE,
    )
    _box(
        axis,
        8.35,
        2.52,
        2.40,
        1.02,
        "Adam update",
        "minimize Lvalue\nall network parameters",
        facecolor=ENCODER,
    )
    _box(
        axis,
        12.10,
        2.52,
        3.05,
        1.02,
        "Primary criterion",
        "validation RMSE for stopping\noriginal-scale RMSE for comparison",
        facecolor=NEUTRAL,
    )
    _arrow(axis, (3.20, 3.03), (4.25, 3.03), dashed=True, label="y")
    _arrow(
        axis,
        (14.20, 5.55),
        (7.25, 3.34),
        color="#4F8A5B",
        dashed=True,
        connectionstyle="arc3,rad=0.18",
    )
    _arrow(axis, (7.25, 3.03), (8.35, 3.03), color="#4F8A5B")
    _arrow(
        axis,
        (9.55, 3.54),
        (9.75, 5.55),
        color="#C44536",
        dashed=True,
        connectionstyle="arc3,rad=-0.22",
    )
    _footer(axis, gradnorm=False)
    return figure


def value_fractional_diagram() -> plt.Figure:
    figure, axis = _canvas(
        "Model B — value_fractional",
        "Same value network as baseline • value and fractional-difference RMSE balanced dynamically",
    )
    _backbone(axis)

    _box(
        axis,
        0.45,
        3.00,
        2.40,
        0.92,
        "Value RMSE",
        "y versus ŷ\nglobal over horizon",
        facecolor=VALUE,
        title_size=9.8,
    )
    _box(
        axis,
        0.45,
        1.45,
        2.40,
        1.12,
        "Fractional RMSE",
        "(1−B)^d y versus (1−B)^d ŷ\nknown boundary context included",
        facecolor=FRACTIONAL,
        title_size=9.8,
        body_size=7.8,
    )
    _box(
        axis,
        3.75,
        1.86,
        3.10,
        1.65,
        "GradNorm controller",
        "initial-loss calibration before fit\npositive weights: wv + wf = 2\nα = 1.5 • shared-decoder norms",
        facecolor=GRADNORM,
        title_size=10.2,
        body_size=8.0,
    )
    _box(
        axis,
        7.85,
        2.12,
        2.80,
        1.15,
        "Weighted objective",
        "wv Lvalue + wf Lfractional\nweights update every batch",
        facecolor=NEUTRAL,
    )
    _box(
        axis,
        11.65,
        2.12,
        2.60,
        1.15,
        "Two Adam updates",
        "network parameters\n+ GradNorm task weights",
        facecolor=ENCODER,
    )
    axis.text(
        0.50,
        0.74,
        "d is estimated from training data only • finite kernel • actual and predicted futures use the same pre-origin context",
        fontsize=8.5,
        color=MUTED,
        ha="left",
    )
    _arrow(axis, (2.85, 3.46), (3.75, 3.20), color="#4F8A5B")
    _arrow(axis, (2.85, 2.01), (3.75, 2.16), color="#D18A1D")
    _arrow(axis, (6.85, 2.68), (7.85, 2.68), color="#C44536")
    _arrow(axis, (10.65, 2.68), (11.65, 2.68), color="#C44536")
    _arrow(
        axis,
        (14.20, 5.55),
        (2.45, 3.92),
        color="#4F8A5B",
        dashed=True,
        connectionstyle="arc3,rad=0.13",
    )
    _arrow(
        axis,
        (14.05, 5.55),
        (2.45, 2.57),
        color="#D18A1D",
        dashed=True,
        connectionstyle="arc3,rad=0.22",
    )
    _arrow(
        axis,
        (12.95, 3.27),
        (9.75, 5.55),
        color="#C44536",
        dashed=True,
        connectionstyle="arc3,rad=-0.18",
    )
    _arrow(
        axis,
        (9.75, 5.55),
        (5.30, 3.51),
        color="#A45A52",
        dashed=True,
        connectionstyle="arc3,rad=0.22",
    )
    _footer(axis, gradnorm=True)
    return figure


def value_fractional_spectrum_diagram() -> plt.Figure:
    figure, axis = _canvas(
        "Model C — value_fractional_spectrum",
        "Shared encoder–decoder • value and nine-parameter spectrum heads • three dynamically balanced RMSE tasks",
    )
    _backbone(axis, spectrum_head=True)

    losses = [
        (0.45, 3.38, "Value RMSE", "y versus ŷ", VALUE, "#4F8A5B"),
        (
            0.45,
            2.12,
            "Fractional RMSE",
            "(1−B)^d y versus (1−B)^d ŷ\nwith boundary context",
            FRACTIONAL,
            "#D18A1D",
        ),
        (
            0.45,
            0.86,
            "Masked spectrum RMSE",
            "nine standardized labels\ninvalid fits excluded by mask",
            SPECTRUM,
            "#7B5EA7",
        ),
    ]
    for x, y, title, body, color, _ in losses:
        _box(
            axis,
            x,
            y,
            2.85,
            0.94,
            title,
            body,
            facecolor=color,
            title_size=9.5,
            body_size=7.6,
        )

    _box(
        axis,
        4.15,
        1.47,
        3.05,
        2.06,
        "GradNorm controller",
        "initial-loss calibration\nwv + wf + ws = 3\nα = 1.5\nshared-decoder gradient norms",
        facecolor=GRADNORM,
        title_size=10.2,
        body_size=7.9,
    )
    _box(
        axis,
        8.10,
        1.80,
        3.10,
        1.38,
        "Weighted objective",
        "wv Lvalue + wf Lfractional\n+ ws Lspectrum",
        facecolor=NEUTRAL,
    )
    _box(
        axis,
        12.15,
        1.80,
        2.85,
        1.38,
        "Two Adam updates",
        "shared network + both heads\nGradNorm task weights",
        facecolor=ENCODER,
    )

    for _, y, _, _, _, color in losses:
        _arrow(axis, (3.30, y + 0.47), (4.15, 2.50), color=color)
    _arrow(axis, (7.20, 2.50), (8.10, 2.50), color="#C44536")
    _arrow(axis, (11.20, 2.50), (12.15, 2.50), color="#C44536")
    _arrow(
        axis,
        (14.20, 6.05),
        (2.95, 4.32),
        color="#4F8A5B",
        dashed=True,
        connectionstyle="arc3,rad=0.12",
    )
    _arrow(
        axis,
        (14.05, 6.05),
        (2.95, 3.06),
        color="#D18A1D",
        dashed=True,
        connectionstyle="arc3,rad=0.20",
    )
    _arrow(
        axis,
        (14.20, 4.92),
        (2.95, 1.80),
        color="#7B5EA7",
        dashed=True,
        connectionstyle="arc3,rad=0.18",
    )
    _arrow(
        axis,
        (13.58, 3.18),
        (9.75, 5.55),
        color="#C44536",
        dashed=True,
        connectionstyle="arc3,rad=-0.16",
    )
    _arrow(
        axis,
        (9.75, 5.55),
        (5.70, 3.53),
        color="#A45A52",
        dashed=True,
        connectionstyle="arc3,rad=0.19",
    )
    _footer(axis, gradnorm=True)
    return figure


def _validate_temporary(path: Path, file_format: str) -> None:
    if path.stat().st_size < 1_000:
        raise ValueError(f"Generated diagram is unexpectedly small: {path}")
    if file_format == "svg":
        prefix = path.read_text(encoding="utf-8")[:500]
        if "<svg" not in prefix:
            raise ValueError(f"Generated file is not SVG: {path}")
        return
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Generated file is not PNG: {path}")


def generate_architecture_diagrams() -> dict[str, dict[str, Path]]:
    """Generate all diagrams atomically and return their published paths."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    generators = {
        VALUE_ONLY: value_only_diagram,
        VALUE_FRACTIONAL: value_fractional_diagram,
        VALUE_FRACTIONAL_SPECTRUM: value_fractional_spectrum_diagram,
    }
    inputs = [
        Path(__file__),
        Path(__file__).with_name("models.py"),
        Path(__file__).with_name("gradnorm.py"),
        Path(__file__).with_name("losses.py"),
        Path(__file__).with_name("constants.py"),
        EXPERIMENT_ROOT / "horizon_10" / "locked_parameters.json",
        EXPERIMENT_ROOT / "horizon_50" / "locked_parameters.json",
    ]
    published: dict[str, dict[str, Path]] = {}
    for variant, generator in generators.items():
        figure = generator()
        published[variant] = {}
        try:
            for file_format in ("png", "svg"):
                output = OUTPUT_DIRECTORY / f"multistep_{variant}_architecture.{file_format}"
                temporary = temporary_output_path(output)
                try:
                    figure.savefig(
                        temporary,
                        format=file_format,
                        dpi=180 if file_format == "png" else None,
                        facecolor=WHITE,
                        bbox_inches="tight",
                        pad_inches=0.12,
                    )
                    _validate_temporary(temporary, file_format)
                    temporary.chmod(0o644)
                    sidecar = publish_atomic_output(
                        temporary,
                        output,
                        CONFIG_PATH,
                        "multistep_architecture_diagram",
                        input_files=inputs,
                        extra={
                            "artifact_type": "model_architecture_diagram",
                            "experiment_id": "multistep_fractal_ablation_v1",
                            "variant": variant,
                            "format": file_format,
                        },
                    )
                    published[variant][file_format] = output
                    published[variant][f"{file_format}_lineage"] = sidecar
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            plt.close(figure)
    return published


if __name__ == "__main__":
    for model_variant, paths in generate_architecture_diagrams().items():
        print(model_variant)
        for name, path in paths.items():
            print(f"  {name}: {path}")
