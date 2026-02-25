#!/usr/bin/env python3

import math
import numpy as np
import mpmath as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mp.mp.dps = 50
ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876,
    32.935062, 37.586178, 40.918719, 43.327073,
]

ALPHA = mp.e ** (0.25j * mp.pi)
BG      = "#0b1018"
PANEL   = "#121826"
ACCENT1 = "#00b8d9"
ACCENT2 = "#ff6b81"
ACCENT3 = "#ffcf33"
TEXT    = "#e5e9f0"
GRID    = "#1e2535"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.family": "monospace",
})

def N_alpha(sigma: float, t: float) -> float:
    s1 = mp.mpc(sigma, t)
    s2 = mp.mpc(1.0 - sigma, t)
    diff = mp.zeta(s1) - ALPHA * mp.zeta(s2)
    return float(mp.re(diff)**2 + mp.im(diff)**2)


def curvature_sigma(sigma: float, t: float, h: float = 0.015) -> float:
    np_ = N_alpha(sigma + h, t)
    nc  = N_alpha(sigma,     t)
    nm  = N_alpha(sigma - h, t)
    return (np_ - 2.0 * nc + nm) / (h * h)


def sigma_profile(t: float, n_sigma: int = 80):
    sigmas = np.linspace(0.1, 0.9, n_sigma)
    values = np.array([N_alpha(float(s), t) for s in sigmas])
    idx    = int(np.argmin(values))
    return sigmas, values, float(sigmas[idx]), float(values[idx])


def run_experiment():
    print("="*72)
    print("  ANTI-LOCUS-TRAP PUBLIC EXPERIMENT  (N_alpha symmetry-broken)")
    print("="*72)
    print(f"alpha = e^(iπ/4) ≈ {complex(ALPHA):.4f}")
    print()


    print("σ-dependence at known zeros:")
    print(f"{'γ (zero)':>10}  {'σ_min':>8}  {'N_min':>14}  {'|σ_min-1/2|':>12}")
    print(f"{'-'*10}  {'-'*8}  {'-'*14}  {'-'*12}")

    zero_results = []
    for gamma in ZEROS:
        sigmas, vals, s_min, n_min = sigma_profile(gamma)
        dev = abs(s_min - 0.5)
        print(f"{gamma:>10.6f}  {s_min:>8.4f}  {n_min:>14.4e}  {dev:>12.4f}")
        zero_results.append((gamma, s_min, n_min, dev))


    print("\nσ-dependence at midpoints (not zeros):")
    print(f"{'t (mid)':>10}  {'σ_min':>8}  {'N_min':>14}  {'|σ_min-1/2|':>12}")
    print(f"{'-'*10}  {'-'*8}  {'-'*14}  {'-'*12}")

    mid_results = []
    for i in range(len(ZEROS) - 1):
        mid = 0.5 * (ZEROS[i] + ZEROS[i+1])
        sigmas, vals, s_min, n_min = sigma_profile(mid)
        dev = abs(s_min - 0.5)
        print(f"{mid:>10.6f}  {s_min:>8.4f}  {n_min:>14.4e}  {dev:>12.4f}")
        mid_results.append((mid, s_min, n_min, dev))


    z_dev  = [d for _,_,_,d in zero_results]
    m_dev  = [d for _,_,_,d in mid_results]
    z_N    = [n for _,_,n,_ in zero_results]
    m_N    = [n for _,_,n,_ in mid_results]

    print("\nSummary:")
    print(f"  Mean |σ_min-1/2| at zeros:      {np.mean(z_dev):.4e}")
    print(f"  Mean |σ_min-1/2| at midpoints:  {np.mean(m_dev):.4e}")
    print(f"  Mean N_min at zeros:            {np.mean(z_N):.4e}")
    print(f"  Mean N_min at midpoints:        {np.mean(m_N):.4e}")
    print(f"  Dynamic range (mid/zero N_min): {np.mean(m_N)/max(np.mean(z_N),1e-30):.1f}x")

    return zero_results, mid_results


def make_figure(zero_results, mid_results, filename="MKM_THE_LOCUS_TRAP.png"):
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)


    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PANEL)

    sigma_grid = np.linspace(0.1, 0.9, 90)


    for i, (gamma, _, _, _) in enumerate(zero_results[:4]):
        vals = np.array([N_alpha(float(s), gamma) for s in sigma_grid])
        ax0.plot(sigma_grid, np.log1p(vals),
                 color=ACCENT1, lw=1.6, alpha=0.9,
                 label="zeros" if i == 0 else "_nolegend_")
    for i in range(3):
        mid = 0.5 * (zero_results[i][0] + zero_results[i+1][0])
        vals = np.array([N_alpha(float(s), mid) for s in sigma_grid])
        ax0.plot(sigma_grid, np.log1p(vals),
                 color=ACCENT2, lw=1.4, alpha=0.8, linestyle="--",
                 label="midpoints" if i == 0 else "_nolegend_")

    ax0.axvline(0.5, color=ACCENT3, lw=2.0, linestyle=":",
                label="σ = 1/2")
    ax0.set_xlabel("σ")
    ax0.set_ylabel("log(1 + N_alpha(σ,t))")
    ax0.set_title("σ-profile of N_alpha at zeros vs midpoints")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)


    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(PANEL)

    zg = [g for g,_,_,_ in zero_results]
    zd = [d for _,_,_,d in zero_results]
    mg = [mid for mid,_,_,_ in mid_results]
    md = [d for _,_,_,d in mid_results]

    ax1.scatter(zg, zd, color=ACCENT1, s=50, label="zeros")
    ax1.scatter(mg, md, color=ACCENT2, s=50, marker="^", label="midpoints")
    ax1.axhline(0.03, color=ACCENT3, lw=1.5, linestyle="--",
                label="example threshold")
    ax1.set_xlabel("t")
    ax1.set_ylabel("|σ_min - 1/2|")
    ax1.set_title("Distance of σ_min from the critical line")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)


    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL)

    zN_half = [N_alpha(0.5, g) for g in zg]
    mN_half = [N_alpha(0.5, mid) for mid in mg]

    ax2.scatter(zg, zN_half, color=ACCENT1, s=50, label="zeros")
    ax2.scatter(mg, mN_half, color=ACCENT2, s=50, marker="^",
                label="midpoints")
    ax2.set_yscale("log")
    ax2.set_xlabel("t")
    ax2.set_ylabel("N_alpha(1/2, t)  [log scale]")
    ax2.set_title("Magnitude of N_alpha on the critical line")
    ax2.grid(True, alpha=0.25, which="both")
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Anti-Locus-Trap Experiment (Public Version)\n"
        "N_alpha uses ζ(σ+it) and ζ(1-σ+it) with α ≠ 1 — σ = 1/2 is not hard-coded",
        color=ACCENT3,
        fontsize=11,
    )

    plt.savefig(filename, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nSaved figure: {filename}")


def main():
    zero_results, mid_results = run_experiment()
    make_figure(zero_results, mid_results)

if __name__ == "__main__":
    main()
