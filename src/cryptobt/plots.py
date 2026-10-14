"""Static figures for the README (matplotlib, light background)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # fixed order, validated palette


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w=9, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _title(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def equity_chart(equities: dict[str, pd.Series], path: Path, title: str, subtitle: str = "") -> Path:
    fig, ax = _fig()
    for i, (name, eq) in enumerate(equities.items()):
        ax.plot(eq.index, eq.values, color=SERIES[i], linewidth=1.8, label=name)
        ax.annotate(
            f"{eq.iloc[-1]:.1f}x",
            (eq.index[-1], eq.iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            va="center",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    _title(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def drawdown_chart(drawdowns: dict[str, pd.Series], path: Path, title: str, subtitle: str = "") -> Path:
    fig, ax = _fig(h=3.6)
    for i, (name, dd) in enumerate(drawdowns.items()):
        ax.plot(dd.index, dd.values * 100, color=SERIES[i], linewidth=1.4, label=name)
    ax.set_ylabel("Drawdown from peak (%)", color=INK_2, fontsize=9)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK, ncol=len(drawdowns))
    _title(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def sharpe_ci_chart(ci: pd.DataFrame, path: Path, title: str, subtitle: str = "") -> Path:
    """Dot-and-interval chart. ``ci`` rows = labels, columns estimate/ci_low/ci_high/is_benchmark."""
    fig, ax = _fig(h=0.42 * len(ci) + 1.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ys = range(len(ci))[::-1]
    for y, (label, row) in zip(ys, ci.iterrows()):
        color = SERIES[1] if row.get("is_benchmark", False) else SERIES[0]
        ax.plot([row.ci_low, row.ci_high], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(row.estimate, y, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{row.estimate:.2f}", (row.ci_high, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=INK_2, va="center")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(ci.index, fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Annualised Sharpe ratio (point estimate and 95% bootstrap interval)", color=INK_2, fontsize=9)
    _title(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
