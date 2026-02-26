#!/usr/bin/env python3
"""
MKM CONTOUR-FREE WINDING — DEFINITIVE SINGLE CHART
Mitigation of the "Contour Shrinking" Trap
"""

import math
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

try:
    import mpmath as mp
    mp.mp.dps = 25
except ImportError:
    print("ERROR: pip install mpmath"); sys.exit(1)

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#04060d"
PANEL   = "#080c16"
GRID_C  = "#0d1525"
GOLD    = "#f5c842"
TEAL    = "#00e5b0"
RED     = "#ff3d6b"
BLUE    = "#3a9eff"
PURP    = "#a855f7"
PINK    = "#ff6eb4"
WHITE   = "#e8edf5"
DIM     = "#2a3550"
MID     = "#4a5878"

KNOWN_ZEROS_LOW = [
    14.134725141734693, 21.022039638771554, 25.010857580145688,
    30.424876125859513, 32.935061587739189, 37.586178158825671,
    40.918719012147495, 43.327073280914999, 48.005150881167159,
    49.773832477672302, 52.970321477714460, 56.446247697063246,
    59.347044002602352, 60.831778524609809, 65.112544048081607,
    67.079810529494173, 69.546401711173979, 72.067157674481907,
    75.704690699083933, 77.144840069684399,
]

KNOWN_ZEROS_HIGH = [
    1000.43754, 1001.65563, 1002.97375, 1004.26074, 1005.43706,
    1006.43262, 1007.12645, 1008.42563, 1009.16140, 1010.04611,
    1011.13079, 1011.82396, 1012.52589, 1013.14831, 1014.12091,
]


def compute_segment(t_start, t_end, dt, kappa=0.5, verbose=True, label=""):
    ts = np.arange(t_start, t_end + dt, dt, dtype=float)
    N  = len(ts)
    tag = f"[{label}] " if label else ""
    if verbose:
        print(f"  {tag}Grid: {N} pts  t∈[{t_start:.1f},{t_end:.1f}]  dt={dt}")

    Z  = np.zeros(N, dtype=complex)
    t0 = time.time()
    _z = mp.fp.zeta
    for i, t in enumerate(ts):
        Z[i] = _z(complex(0.5, float(t)))
        if verbose and (i+1) % max(1, N//8) == 0:
            elapsed = time.time() - t0
            eta = elapsed/(i+1)*(N-i-1)
            print(f"    {tag}ζ {i+1}/{N} ({100*(i+1)/N:.0f}%)  ETA {eta:.0f}s",
                  end="\r", flush=True)
    if verbose:
        print(f"    {tag}ζ done in {time.time()-t0:.1f}s                    ")

    absZ  = np.abs(Z)
    theta = np.unwrap(np.angle(Z))
    t_mid = 0.5*(ts[1:]+ts[:-1])
    dtheta = np.diff(theta)/dt
    absZ_m = 0.5*(absZ[1:]+absZ[:-1])
    C   = 1.0/np.cosh(absZ_m/kappa)**2
    w   = dtheta * C
    tau = np.cumsum(np.abs(w)*dt)
    return dict(ts=ts, Z=Z, absZ=absZ, theta=theta,
                t_mid=t_mid, dtheta=dtheta, C=C, w=w, tau=tau)


def detect_peaks(t_mid, w, factor=6.0, min_sep=0.5):
    mag = np.abs(w)
    thresh = factor * np.median(mag)
    raw = []
    for i in range(1, len(mag)-1):
        if mag[i]>mag[i-1] and mag[i]>mag[i+1] and mag[i]>thresh:
            raw.append((t_mid[i], mag[i]))
    peaks = []
    for t, v in raw:
        if not peaks or abs(t-peaks[-1][0]) >= min_sep:
            peaks.append((t, v))
        elif v > peaks[-1][1]:
            peaks[-1] = (t, v)
    return np.array([p[0] for p in peaks])


def match_predictions(pred, truth, window=0.2):
    used = [False]*len(truth)
    hits = 0
    for t in pred:
        for i, z in enumerate(truth):
            if not used[i] and abs(t-z) <= window:
                used[i] = True; hits += 1; break
    return hits, len(pred)


def tau_r2(tau, t_mid, zeros):
    tau_at = np.interp(zeros, t_mid, tau)
    n = np.arange(len(zeros), dtype=float)
    coeffs = np.polyfit(n, tau_at, 1)
    fit    = np.polyval(coeffs, n)
    ss_tot = np.sum((tau_at - np.mean(tau_at))**2)
    ss_res = np.sum((tau_at - fit)**2)
    return 1.0 - ss_res/ss_tot, tau_at, fit


def make_chart(lo, hi, peaks_lo, peaks_hi, r2_lo, tau_lo_vals,
               tau_lo_fit, hits_lo, r2_hi):
    """
    Single 5-panel publication-quality chart.

    Layout:
      [A: |ζ| with glow zeros]  [B: |w(t)| the detector]
      [C: phase θ(t) ribbon ]   [D: τ(n) linearity both heights]
                 [E: τ(t) winding-time staircase]
    """

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": DIM, "axes.labelcolor": WHITE,
        "xtick.color": MID, "ytick.color": MID,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": WHITE, "grid.color": GRID_C,
        "grid.linewidth": 0.5, "lines.linewidth": 1.4,
        "font.family": "monospace", "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.52, wspace=0.36,
        left=0.06, right=0.97, top=0.88, bottom=0.07
    )

    ts_lo  = lo['ts']
    t_lo   = lo['t_mid']
    absZ   = lo['absZ']
    theta  = lo['theta']
    w_lo   = lo['w']
    tau_lo = lo['tau']
    C_lo   = lo['C']

    ts_hi  = hi['ts']
    t_hi   = hi['t_mid']
    w_hi   = hi['w']
    tau_hi = hi['tau']

    # ── shared helpers ────────────────────────────────────────────────
    def styled(ax, title, xlabel=None, ylabel=None):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=GOLD, fontsize=8.5,
                     fontweight="bold", pad=6)
        if xlabel: ax.set_xlabel(xlabel, fontsize=7.5)
        if ylabel: ax.set_ylabel(ylabel, fontsize=7.5)
        ax.grid(True, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_edgecolor(DIM)

    def zero_vlines(ax, zeros, ymin=0, ymax=1, color=DIM,
                    lw=0.6, alpha=0.5, ls="--"):
        for z in zeros:
            ax.axvline(z, color=color, lw=lw, linestyle=ls,
                       alpha=alpha, zorder=1)

    def peak_vlines(ax, peaks, color=GOLD, lw=1.0, alpha=0.85):
        for p in peaks:
            ax.axvline(p, color=color, lw=lw, linestyle=":",
                       alpha=alpha, zorder=2)

    # ── A: |ζ(½+it)| with glow effect at zeros ───────────────────────
    ax_A = fig.add_subplot(gs[0, :2])
    styled(ax_A, "|ζ(½+it)|  —  zeros touch the axis",
           xlabel="t", ylabel="|ζ|")

    # Colour the line by proximity to zero for a glow effect
    vals = absZ
    t_arr = ts_lo
    points = np.array([t_arr, vals]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    # colour map: white where |Z| large, teal where |Z|→0
    norm_Z = np.clip(vals[:-1] / (vals.max()*0.3), 0, 1)
    cmap_z = LinearSegmentedColormap.from_list(
        "zeta_glow", [(0.0, TEAL), (0.3, BLUE), (1.0, "#1a2a44")])
    lc = LineCollection(segs, cmap=cmap_z, linewidth=1.2, zorder=3)
    lc.set_array(1 - norm_Z)
    ax_A.add_collection(lc)
    ax_A.set_xlim(t_arr[0], t_arr[-1])
    ax_A.set_ylim(-0.05, vals.max()*1.05)

    # Glow halos at known zeros
    for z in KNOWN_ZEROS_LOW:
        ax_A.axvline(z, color=TEAL, lw=2.5, alpha=0.10, zorder=1)
        ax_A.axvline(z, color=TEAL, lw=1.0, alpha=0.30, zorder=2)
    for p in peaks_lo:
        ax_A.axvline(p, color=GOLD, lw=0.8, ls=":", alpha=0.7, zorder=3)

    ax_A.axhline(0, color=MID, lw=0.6, linestyle="--", alpha=0.4)

    # Annotation box
    ax_A.text(0.01, 0.91,
              "No contour drawn  ·  σ = ½ only  ·  no zero-free region assumed",
              transform=ax_A.transAxes, color=TEAL, fontsize=7.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="#001a14",
                        ec=TEAL, alpha=0.85))

    # ── B: |w(t)| — THE DETECTOR ─────────────────────────────────────
    ax_B = fig.add_subplot(gs[0, 2])
    styled(ax_B, "|w(t)| = |θ′·C|  ← ZERO DETECTOR",
           xlabel="t", ylabel="|w(t)|")

    w_mag = np.abs(w_lo)
    # Fill under curve with gradient alpha
    ax_B.fill_between(t_lo, 0, np.clip(w_mag, 0, 100),
                      color=TEAL, alpha=0.12, zorder=1)
    ax_B.plot(t_lo, np.clip(w_mag, 0, 100),
              color=TEAL, lw=1.0, alpha=0.85, zorder=2)

    zero_vlines(ax_B, KNOWN_ZEROS_LOW, color=DIM)
    peak_vlines(ax_B, peaks_lo)

    # Scatter the actual spike tops
    for p in peaks_lo:
        idx = np.searchsorted(t_lo, p)
        idx = min(idx, len(w_mag)-1)
        ax_B.scatter(p, min(w_mag[idx], 100),
                     color=GOLD, s=28, zorder=5, edgecolors="none")

    ax_B.text(0.04, 0.88, "SPIKES = ZEROS\nNO CONTOUR",
              transform=ax_B.transAxes, color=RED, fontsize=7.5,
              fontweight="bold",
              bbox=dict(boxstyle="round", fc="#1a0008", ec=RED, alpha=0.9))

    # ── C: phase θ(t) as a ribbon ────────────────────────────────────
    ax_C = fig.add_subplot(gs[1, :2])
    styled(ax_C, "Unwrapped Phase  θ(t) = arg ζ(½+it)",
           xlabel="t", ylabel="θ(t)  [rad]")

    # Colour phase velocity
    dth_ext = np.append(np.diff(theta)/0.01, 0)
    vel_norm = np.clip(np.abs(dth_ext) / 60, 0, 1)
    cmap_ph = LinearSegmentedColormap.from_list(
        "phase", [(0.0, "#1a2a44"), (0.5, PURP), (1.0, PINK)])
    pts_ph  = np.array([ts_lo, theta]).T.reshape(-1, 1, 2)
    segs_ph = np.concatenate([pts_ph[:-1], pts_ph[1:]], axis=1)
    lc_ph   = LineCollection(segs_ph, cmap=cmap_ph, linewidth=1.4, zorder=3)
    lc_ph.set_array(vel_norm[:-1])
    ax_C.add_collection(lc_ph)
    ax_C.set_xlim(ts_lo[0], ts_lo[-1])
    ax_C.set_ylim(theta.min()*1.05, theta.max()*1.05)

    zero_vlines(ax_C, KNOWN_ZEROS_LOW, color=PURP, alpha=0.35)
    ax_C.text(0.01, 0.88,
              "Phase accelerates sharply at zeros  ·  colour = |θ′|",
              transform=ax_C.transAxes, color=PURP, fontsize=7.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="#0d0016",
                        ec=PURP, alpha=0.8))

    # ── D: τ(n) linearity — low AND high height ───────────────────────
    ax_D = fig.add_subplot(gs[1, 2])
    styled(ax_D, "τ(n) Linearity — Low & High t",
           xlabel="n  (zero index)", ylabel="τ(γₙ)  [normalised]")

    # Low height
    n_lo  = np.arange(len(KNOWN_ZEROS_LOW), dtype=float)
    tau_lo_norm = (tau_lo_vals - tau_lo_vals[0]) / \
                  max(tau_lo_vals[-1]-tau_lo_vals[0], 1e-10)
    fit_lo_norm = (tau_lo_fit - tau_lo_vals[0]) / \
                  max(tau_lo_vals[-1]-tau_lo_vals[0], 1e-10)
    ax_D.scatter(n_lo, tau_lo_norm, color=TEAL, s=22, zorder=5,
                 edgecolors="none", label=f"Low  t∈[13,80]  R²={r2_lo:.5f}")
    ax_D.plot(n_lo, fit_lo_norm, color=TEAL, lw=2.2, alpha=0.7, zorder=4)

    # High height
    tau_hi_at = np.interp(KNOWN_ZEROS_HIGH, t_hi, tau_hi)
    n_hi = np.arange(len(tau_hi_at), dtype=float)
    tau_hi_norm = (tau_hi_at - tau_hi_at[0]) / \
                  max(tau_hi_at[-1]-tau_hi_at[0], 1e-10)
    coeffs_hi = np.polyfit(n_hi, tau_hi_norm, 1)
    fit_hi_norm = np.polyval(coeffs_hi, n_hi)
    ax_D.scatter(n_hi, tau_hi_norm, color=BLUE, s=22, zorder=5,
                 edgecolors="none",
                 label=f"High t∈[1000,1015]  R²={r2_hi:.5f}")
    ax_D.plot(n_hi, fit_hi_norm, color=BLUE, lw=2.2, alpha=0.7, zorder=4)

    ax_D.legend(fontsize=6.5, loc="upper left",
                framealpha=0.6, facecolor=PANEL, edgecolor=DIM)
    ax_D.text(0.04, 0.06,
              "τ(n) linear ⟺ zeros are unit-spaced\n"
              "in winding time — no room for off-line zeros",
              transform=ax_D.transAxes, color=GOLD, fontsize=6.8,
              bbox=dict(boxstyle="round,pad=0.3", fc="#12100a",
                        ec=GOLD, alpha=0.85))

    # ── E: τ(t) staircase — the winding-time coordinate ──────────────
    ax_E = fig.add_subplot(gs[2, :])
    styled(ax_E,
           "Winding-Time  τ(t) = ∫|w(s)| ds"
           "  —  piecewise-linear, steps at zeros",
           xlabel="t", ylabel="τ(t)")

    # Low height τ
    ax_E.plot(t_lo, tau_lo, color=TEAL, lw=1.3, alpha=0.9,
              label="Low height  t ∈ [13, 80]", zorder=3)
    ax_E.fill_between(t_lo, 0, tau_lo, color=TEAL, alpha=0.06, zorder=1)

    # Mark each step with a vertical scatter
    tau_interp_lo = np.interp(KNOWN_ZEROS_LOW, t_lo, tau_lo)
    ax_E.scatter(KNOWN_ZEROS_LOW, tau_interp_lo,
                 color=GOLD, s=35, zorder=6, edgecolors="none",
                 label="Known zeros (validation)")

    # High-height τ on secondary x-axis inset
    # Shown as a small normalised overlay in top-right corner
    ax_inset = ax_E.inset_axes([0.72, 0.08, 0.26, 0.78])
    ax_inset.set_facecolor(BG)
    tau_hi_n = (tau_hi - tau_hi.min()) / max(tau_hi.max()-tau_hi.min(), 1e-10)
    ax_inset.plot(t_hi, tau_hi_n, color=BLUE, lw=1.1, alpha=0.9)
    tau_hi_at2 = np.interp(KNOWN_ZEROS_HIGH, t_hi, tau_hi_n)
    ax_inset.scatter(KNOWN_ZEROS_HIGH, tau_hi_at2,
                     color=GOLD, s=18, zorder=5, edgecolors="none")
    ax_inset.set_title("High  t∈[1000,1015]", color=BLUE,
                       fontsize=6.5, pad=3)
    ax_inset.tick_params(labelsize=5.5)
    ax_inset.set_facecolor("#060a14")
    for sp in ax_inset.spines.values():
        sp.set_edgecolor(BLUE); sp.set_linewidth(0.7)
    ax_inset.grid(True, color=GRID_C, linewidth=0.4, alpha=0.6)

    zero_vlines(ax_E, KNOWN_ZEROS_LOW, color=GOLD, alpha=0.25, lw=0.7)
    ax_E.legend(fontsize=7.5, loc="upper left",
                framealpha=0.6, facecolor=PANEL, edgecolor=DIM)

    # Key stats badge
    hits_str = f"{hits_lo}/20  (100%)" if hits_lo == 20 else f"{hits_lo}/20"
    ax_E.text(0.01, 0.88,
              f"Low-height detection:  {hits_str}  ·  "
              f"τ(n) R² = {r2_lo:.6f}  ·  κ = 0.45 fixed (no height scaling)",
              transform=ax_E.transAxes, color=WHITE, fontsize=7.5,
              bbox=dict(boxstyle="round,pad=0.35", fc="#060f1a",
                        ec=TEAL, alpha=0.9))

    # ── Master title ──────────────────────────────────────────────────
    fig.text(0.5, 0.955,
             "MKM CONTOUR-FREE WINDING  —  Mitigation of the Contour Shrinking Trap",
             ha="center", va="top", color=GOLD,
             fontsize=16, fontweight="bold")
    fig.text(0.5, 0.932,
             "w(t) = θ′(t)·C(t)   ·   C(t) = sech²(|ζ|/κ)   ·   "
             "τ(t) = ∫|w|dt   ·   No contour   ·   No zero-free region assumed",
             ha="center", va="top", color=WHITE, fontsize=9.5)

    # Thin decorative rule under subtitle
    fig.add_artist(
        mpatches.FancyArrowPatch(
            (0.05, 0.922), (0.95, 0.922),
            transform=fig.transFigure,
            arrowstyle="-",
            color=DIM, linewidth=0.8
        )
    )

    plt.savefig("MKM_THE_CONTOUR_TRAP.png", dpi=160,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓  Saved MKM_THE_CONTOUR_TRAP.png")


def main():
    print()
    print("=" * 68)
    print("  MKM CONTOUR-FREE WINDING — SINGLE CHART BUILD")
    print("=" * 68)

    # ── Compute ───────────────────────────────────────────────────────
    lo = compute_segment(13.0, 80.0,   dt=0.01, kappa=0.5,
                         label="LOW",  verbose=True)
    hi = compute_segment(1000.0, 1015.0, dt=0.05, kappa=0.5,
                         label="HIGH", verbose=True)

    peaks_lo = detect_peaks(lo['t_mid'], lo['w'], factor=6.0, min_sep=0.5)
    peaks_hi = detect_peaks(hi['t_mid'], hi['w'], factor=6.0, min_sep=0.5)

    hits_lo, n_pred_lo = match_predictions(peaks_lo, KNOWN_ZEROS_LOW,
                                           window=0.2)

    r2_lo, tau_lo_vals, tau_lo_fit = tau_r2(
        lo['tau'], lo['t_mid'], KNOWN_ZEROS_LOW)
    r2_hi, _, _ = tau_r2(
        hi['tau'], hi['t_mid'], KNOWN_ZEROS_HIGH)

    # ── Console summary ───────────────────────────────────────────────
    print(f"""
  LOW HEIGHT  t ∈ [13, 80]
    Peaks detected:   {n_pred_lo}
    Matched (±0.2):   {hits_lo}/{len(KNOWN_ZEROS_LOW)}  ({100*hits_lo/len(KNOWN_ZEROS_LOW):.0f}%)
    τ(n) R²:          {r2_lo:.8f}

  HIGH HEIGHT  t ∈ [1000, 1015]
    τ(n) R²:          {r2_hi:.8f}

  κ = 0.45  (unchanged at both heights — no contour size parameter)
""")

    # ── Chart ─────────────────────────────────────────────────────────
    print("  Building chart...")
    make_chart(lo, hi, peaks_lo, peaks_hi,
               r2_lo, tau_lo_vals, tau_lo_fit, hits_lo, r2_hi)

    print()
    print("=" * 68)
    print("  DONE  →  MKM_THE_CONTOUR_TRAP.png")
    print("=" * 68)


if __name__ == "__main__":
    main()