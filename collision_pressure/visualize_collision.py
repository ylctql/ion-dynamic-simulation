"""Collision simulation visualization: trajectory snapshots, before/after, batch statistics."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from collision_pressure._mpl_backend import pyplot

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_trajectory_snapshots(
    trajectory: np.ndarray,
    time_us: np.ndarray,
    r0_um: np.ndarray,
    hit_ion: int,
    n_snapshots: int = 8,
    output: str | Path | None = None,
    v0: float | None = None,
    b: float | None = None,
    theta: float | None = None,
    dv: np.ndarray | None = None,
    reconfigured: bool | None = None,
) -> Figure:
    """Static multi-snapshot figure of a collision trajectory.

    Parameters
    ----------
    trajectory : (6N, n_steps) array — [r; v] in um, um/us
    time_us : (n_steps,) time array in us
    r0_um : (N, 3) equilibrium positions in um
    hit_ion : index of struck ion
    n_snapshots : number of time points to sample
    output : save path (png)
    v0 : H2 speed (m/s)
    b : impact parameter (m)
    theta : scattering angle (rad)
    dv : (3,) kick velocity (m/s)
    reconfigured : whether reconfiguration was detected
    """
    plt = pyplot()
    N = r0_um.shape[0]
    indices = np.linspace(0, trajectory.shape[1] - 1, n_snapshots, dtype=int)

    # Compute shared axis limits from all frames + equilibrium (same as before-after)
    r_all = trajectory[:3*N, :].reshape(N, 3, -1)
    z_all = np.concatenate([r_all[:, 2, :].ravel(), r0_um[:, 2]])
    x_all = np.concatenate([r_all[:, 0, :].ravel(), r0_um[:, 0]])
    y_all = np.concatenate([r_all[:, 1, :].ravel(), r0_um[:, 1]])
    z_lo, z_hi = z_all.min(), z_all.max()
    x_lo, x_hi = x_all.min(), x_all.max()
    y_lo, y_hi = y_all.min(), y_all.max()
    margin = max((z_hi - z_lo) * 0.08, (x_hi - x_lo) * 0.08, (y_hi - y_lo) * 0.08, 0.5)
    zlim = (z_lo - margin, z_hi + margin)
    xlim = (x_lo - margin, x_hi + margin)
    ylim = (y_lo - margin, y_hi + margin)

    fig, axes = plt.subplots(n_snapshots, 2, figsize=(10, 4 * n_snapshots))

    for row, idx in enumerate(indices):
        r = trajectory[:3*N, idx].reshape(N, 3)
        t = time_us[idx]

        colors = np.full(N, "steelblue")
        colors[hit_ion] = "red"

        # Left column: zox
        ax = axes[row, 0]
        ax.scatter(r[:, 2], r[:, 0], c=colors.tolist(), s=10, zorder=3)
        ax.scatter(r0_um[:, 2], r0_um[:, 0], c="gray", s=4, alpha=0.3, zorder=2)
        ax.set_xlim(zlim)
        ax.set_ylim(xlim)
        ax.set_aspect("equal")
        ax.set_ylabel(f"t={t:.1f} us  x (um)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Right column: zoy
        ax = axes[row, 1]
        ax.scatter(r[:, 2], r[:, 1], c=colors.tolist(), s=10, zorder=3)
        ax.scatter(r0_um[:, 2], r0_um[:, 1], c="gray", s=4, alpha=0.3, zorder=2)
        ax.set_xlim(zlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # Column titles and bottom x-labels
    axes[0, 0].set_title("zox", fontsize=9)
    axes[0, 1].set_title("zoy", fontsize=9)
    axes[-1, 0].set_xlabel("z (um)", fontsize=8)
    axes[-1, 1].set_xlabel("z (um)", fontsize=8)

    # Build title with collision parameters
    title = f"hit ion #{hit_ion}"
    parts = [title]
    if v0 is not None:
        parts.append(f"v0={v0:.1f} m/s")
    if b is not None:
        parts.append(f"b={b*1e9:.2f} nm")
    if theta is not None:
        parts.append(f"theta={np.degrees(theta):.1f} deg")
    if dv is not None:
        parts.append(f"|dv|={np.linalg.norm(dv):.2f} m/s")
    if reconfigured is not None:
        status = "RECONFIGURED" if reconfigured else "unchanged"
        parts.append(status)
    fig.suptitle("  |  ".join(parts), fontsize=9)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig


def plot_before_after(
    r0_um: np.ndarray,
    r_final_um: np.ndarray,
    hit_ion: int,
    flip_axis: int = 1,
    reconfigured: bool = False,
    output: str | Path | None = None,
    v0: float | None = None,
    b: float | None = None,
    theta: float | None = None,
    dv: np.ndarray | None = None,
) -> Figure:
    """Before/after comparison, stacked vertically.

    Top: equilibrium, Bottom: post-collision.
    Color by sign along flip_axis (blue=positive, orange/negative).
    """
    plt = pyplot()
    N = r0_um.shape[0]
    sign0 = np.sign(r0_um[:, flip_axis])
    sign_f = np.sign(r_final_um[:, flip_axis])

    def _colors(signs):
        c = []
        for s in signs:
            if s > 0:
                c.append("steelblue")
            elif s < 0:
                c.append("orange")
            else:
                c.append("gray")
        return c

    # Compute shared axis limits from both configurations
    z_all = np.concatenate([r0_um[:, 2], r_final_um[:, 2]])
    f_all = np.concatenate([r0_um[:, flip_axis], r_final_um[:, flip_axis]])
    z_lo, z_hi = z_all.min(), z_all.max()
    f_lo, f_hi = f_all.min(), f_all.max()
    margin = max((z_hi - z_lo) * 0.08, (f_hi - f_lo) * 0.08, 0.5)
    zlim = (z_lo - margin, z_hi + margin)
    flip_lim = (f_lo - margin, f_hi + margin)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

    for ax, r, signs, title in [
        (ax0, r0_um, sign0, "Before (equilibrium)"),
        (ax1, r_final_um, sign_f, "After (post-collision)"),
    ]:
        ax.scatter(r[:, 2], r[:, flip_axis], c=_colors(signs), s=15, zorder=3)
        ax.scatter(r[hit_ion, 2], r[hit_ion, flip_axis],
                   facecolors="none", edgecolors="red", s=80, linewidths=2, zorder=4)
        ylabel = ["x", "y", "z"][flip_axis]
        ax.set_ylabel(f"{ylabel} (um)")
        ax.set_xlim(zlim)
        ax.set_ylim(flip_lim)
        ax.set_aspect("equal")
        ax.set_title(title)

    ax1.set_xlabel("z (um)")

    status = "RECONFIGURED" if reconfigured else "unchanged"
    parts = [f"hit ion #{hit_ion}", status]
    if v0 is not None:
        parts.append(f"v0={v0:.1f} m/s")
    if theta is not None:
        parts.append(f"theta={np.degrees(theta):.1f} deg")
    if dv is not None:
        parts.append(f"|dv|={np.linalg.norm(dv):.2f} m/s")
    fig.suptitle("  |  ".join(parts), fontsize=10)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig


def plot_batch_statistics(
    collisions: list,
    N_ions: int,
    output: str | Path | None = None,
) -> Figure:
    """4-panel summary of batch collision statistics.

    Top-left: scattering angle histogram (reconfigured vs not)
    Top-right: H2 velocity histogram (reconfigured vs not)
    Bottom-left: reconfigured fraction per hit-ion index
    Bottom-right: kick magnitude histogram (reconfigured vs not)
    """
    plt = pyplot()
    reconfig = np.array([c.reconfigured for c in collisions])
    thetas = np.array([c.theta for c in collisions])
    v0s = np.array([c.v0 for c in collisions])
    hit_ions = np.array([c.hit_ion for c in collisions])
    dv_mags = np.array([np.linalg.norm(c.dv) for c in collisions])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    mask_t = reconfig
    mask_f = ~reconfig

    # Top-left: scattering angles
    ax = axes[0, 0]
    bins = np.linspace(0, thetas.max() + 0.1, 25)
    ax.hist(thetas[mask_f], bins=bins, alpha=0.6, label="unchanged", color="steelblue")
    ax.hist(thetas[mask_t], bins=bins, alpha=0.6, label="reconfigured", color="orange")
    ax.set_xlabel("Scattering angle (rad)")
    ax.set_ylabel("Count")
    ax.set_title("Scattering angles")
    ax.legend(fontsize=8)

    # Top-right: H2 velocities
    ax = axes[0, 1]
    bins = np.linspace(0, v0s.max() * 1.1, 25)
    ax.hist(v0s[mask_f], bins=bins, alpha=0.6, label="unchanged", color="steelblue")
    ax.hist(v0s[mask_t], bins=bins, alpha=0.6, label="reconfigured", color="orange")
    ax.set_xlabel("H2 speed (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("H2 velocities")
    ax.legend(fontsize=8)

    # Bottom-left: per-ion reconfiguration fraction
    ax = axes[1, 0]
    ion_range = np.arange(N_ions)
    n_per_ion = np.array([np.sum(hit_ions == i) for i in ion_range])
    n_reconfig_per = np.array([np.sum((hit_ions == i) & reconfig) for i in ion_range])
    with np.errstate(invalid="ignore"):
        frac = np.where(n_per_ion > 0, n_reconfig_per / n_per_ion, 0.0)
    ax.bar(ion_range, frac, color="steelblue", edgecolor="white")
    ax.set_xlabel("Ion index")
    ax.set_ylabel("Reconfig fraction")
    ax.set_title("Reconfiguration per ion")
    ax.set_ylim(0, 1.05)

    # Bottom-right: kick magnitudes
    ax = axes[1, 1]
    bins = np.linspace(0, dv_mags.max() * 1.1, 25)
    ax.hist(dv_mags[mask_f], bins=bins, alpha=0.6, label="unchanged", color="steelblue")
    ax.hist(dv_mags[mask_t], bins=bins, alpha=0.6, label="reconfigured", color="orange")
    ax.set_xlabel("Kick magnitude (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Kick magnitudes")
    ax.legend(fontsize=8)

    fig.suptitle(f"Batch statistics ({len(collisions)} collisions, "
                 f"{reconfig.sum()} reconfigured)", fontsize=11)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig
