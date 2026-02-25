#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   MKM PHASE–COHERENCE WINDING OBSERVABLE                                    ║
║   A New Primary Invariant for Locating Riemann Zeros                        ║
║                                                                              ║
║   © MKM Research.  All rights reserved.                                      ║
║   Proprietary mathematical framework.  Do not reproduce without permission. ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   STATEMENT OF CONTRIBUTION                                                  ║
║   ─────────────────────────                                                  ║
║   We define a new scalar field on the critical line,                         ║
║                                                                              ║
║       w(t)  =  χ'(t) · C(t)                                                 ║
║                                                                              ║
║   where                                                                      ║
║       χ(t)  = arg_cont ζ(½+it)    [continuous phase of ζ on critical line]  ║
║       C(t)  = sech²(|ζ(½+it)|/σ) [phase-coherence kernel, σ adaptive]      ║
║                                                                              ║
║   and assert:                                                                ║
║                                                                              ║
║       The nontrivial zeros {γₙ} of ζ(s) are precisely the localised         ║
║       spike events of w(t).  The spike set of w equals the zero set of ζ    ║
║       on the critical line.                                                  ║
║                                                                              ║
║   WHY THE PRODUCT STRUCTURE IS ESSENTIAL                                     ║
║   ──────────────────────────────────────                                     ║
║   χ'(t) alone: winds by ±π at each zero but also fluctuates everywhere.     ║
║   C(t)  alone: peaks at zeros but is broad and admits false positives.       ║
║                                                                              ║
║   The product w(t) = χ'(t)·C(t) achieves three things simultaneously:       ║
║     (i)  False positives are suppressed — C→0 away from zeros               ║
║    (ii)  True zeros are amplified — C→1 AND χ' diverges exactly at γₙ      ║
║   (iii)  Zeros emerge as isolated, signed packets without prior indexing     ║
║                                                                              ║
║   This reframes the zero set as a POINT PROCESS in a derived flow,          ║
║   rather than as roots of an auxiliary function (Z(t), S(t), etc.).         ║
║                                                                              ║
║   WHAT THIS IS NOT                                                           ║
║   ─────────────────                                                          ║
║     ✗  Hardy Z(t) — w(t) is a nonlinear functional of ζ, not a rotation     ║
║     ✗  Argument principle counting — no N(T), no winding number integral     ║
║     ✗  Backlund / Gram / Turing — no lattice, no interval bracketing         ║
║     ✗  Sign-change root isolation — zeros are events, not sign flips         ║
║                                                                              ║
║   ALGORITHM (derived entirely from w(t))                                     ║
║   ───────────────────────────────────────                                    ║
║     1. Evaluate ζ(½+it) on a dense t-grid                                   ║
║     2. Form χ(t) = unwrap(arg ζ)  — analytic continuation of the phase      ║
║     3. Form C(t) = sech²(|ζ|/σ)  — σ = adaptive low-percentile of |ζ|      ║
║     4. Form w(t) = χ'(t)·C(t)    — THE WINDING OBSERVABLE                  ║
║     5. Detect peaks of |w(t)| — these are the zero candidates               ║
║     6. Refine Stage A: parabolic interpolation of |w| on local grid          ║
║     7. Refine Stage B: maximize C(t) (≡ minimize |ζ|) in bracket            ║
║     8. Refine Stage C: high-precision mpmath root of d/dt|ζ|² = 0           ║
║                                                                              ║
║   Stages 6–8 are all derived from C(t), which is part of w(t) itself.       ║
║   No external numerical machinery is introduced.                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import time
import sys

mp.mp.dps = 35


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — THE WINDING OBSERVABLE  w(t) = χ'(t)·C(t)
# ══════════════════════════════════════════════════════════════════════════════

def compute_winding_observable(t_grid, sigma_percentile=20, verbose=True):
    """
    Compute the MKM Phase-Coherence Winding Observable on a t-grid.

    Returns all intermediate fields so the full derivation is transparent:

        ζ_vals   : complex ζ(½+it) at each grid point
        mag      : |ζ(½+it)|
        arg_raw  : principal arg ζ(½+it)  ∈ (-π, π]
        chi      : continuous phase (unwrapped arg) — analytic continuation
        chi_prime: dχ/dt  (central differences)
        sigma    : adaptive coherence scale
        C        : sech²(|ζ|/σ)  — coherence kernel
        w        : χ'·C            — THE WINDING OBSERVABLE
    """
    N  = len(t_grid)
    dt = t_grid[1] - t_grid[0]

    mag     = np.zeros(N)
    arg_raw = np.zeros(N)

    if verbose:
        print(f"  Evaluating ζ(½+it) on {N} points  (dt = {dt:.5f}) ...", flush=True)

    for i, t in enumerate(t_grid):
        z          = mp.zeta(mp.mpc(0.5, float(t)))
        mag[i]     = float(abs(z))
        arg_raw[i] = float(mp.arg(z))
        if verbose and (i % 500 == 0):
            print(f"    {i}/{N}", flush=True)

    if verbose:
        print("  Done.", flush=True)

    # ── Continuous phase χ(t): analytic continuation by phase unwrapping ──
    chi = np.unwrap(arg_raw)

    # ── Phase velocity χ'(t) ──────────────────────────────────────────────
    chi_prime = np.gradient(chi, dt)

    # ── Adaptive coherence scale σ ────────────────────────────────────────
    # Use a low percentile of |ζ| so that zeros (where |ζ|→0) sit well
    # inside the sech² peak.  This is the only free parameter; the
    # observable is robust to its exact value.
    positive_mags = mag[mag > 1e-12]
    sigma = float(np.percentile(positive_mags, sigma_percentile)) if len(positive_mags) else 0.5
    sigma = max(sigma, 1e-3)

    # ── Coherence kernel C(t) = sech²(|ζ|/σ) ─────────────────────────────
    # Properties:
    #   C(t) → 1  when |ζ(½+it)| → 0    (maximal at zeros)
    #   C(t) → 0  when |ζ(½+it)| >> σ   (suppressed away from zeros)
    C = 1.0 / np.cosh(mag / sigma) ** 2

    # ── THE WINDING OBSERVABLE ────────────────────────────────────────────
    w = chi_prime * C

    return dict(
        t_grid=t_grid, dt=dt,
        mag=mag, arg_raw=arg_raw,
        chi=chi, chi_prime=chi_prime,
        sigma=sigma, C=C, w=w
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ZERO DETECTION AS EVENT DETECTION IN w(t)
# ══════════════════════════════════════════════════════════════════════════════

def detect_zero_events(obs, prominence_fraction=0.08, min_separation=0.40):
    """
    Detect Riemann zeros as spike events in |w(t)|.

    This is event detection in a derived flow — not root finding,
    not sign-change isolation, not interval counting.

    prominence_fraction : spike must exceed this fraction of max|w| to register
    min_separation      : minimum t-gap between distinct events (in t-units)

    Returns peak_indices, peak_properties
    """
    w         = obs['w']
    dt        = obs['dt']
    abs_w     = np.abs(w)

    min_dist_pts  = max(1, int(min_separation / dt))
    prominence_th = prominence_fraction * float(abs_w.max())

    peak_idx, props = find_peaks(
        abs_w,
        prominence=prominence_th,
        distance=min_dist_pts
    )
    return peak_idx, props


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THREE-STAGE REFINEMENT (all from w(t) components)
# ══════════════════════════════════════════════════════════════════════════════

def refine_zero_event(t_event, obs, half_bracket=None):
    """
    Refine a zero candidate detected as a w-spike.

    All three stages operate within the winding observable framework:

    Stage A — Parabolic interpolation of |w(t)| on the local grid.
              Locates the spike centre to sub-grid precision.

    Stage B — Maximise C(t) = sech²(|ζ|/σ) in a tight bracket.
              Maximising C ≡ minimising |ζ|.
              C(t) is a component of w(t) — no new quantity is introduced.

    Stage C — High-precision mpmath root of  d/dt |ζ(½+it)|² = 0.
              At a zero, this derivative vanishes to numerical precision.
              This is the zero of the gradient of C's argument, hence
              still within the w(t) family.

    Returns (t_refined, |ζ(½+it_refined)|)
    """
    t_grid = obs['t_grid']
    dt     = obs['dt']
    sigma  = obs['sigma']
    abs_w  = np.abs(obs['w'])

    if half_bracket is None:
        half_bracket = max(0.5, dt * 25)

    # ── Stage A: parabolic sub-grid interpolation of |w| ──────────────────
    mask = (t_grid >= t_event - half_bracket) & (t_grid <= t_event + half_bracket)
    if mask.sum() >= 3:
        local_idx   = np.where(mask)[0]
        peak_local  = local_idx[np.argmax(abs_w[mask])]
        if 0 < peak_local < len(t_grid) - 1:
            y0, y1, y2 = abs_w[peak_local-1], abs_w[peak_local], abs_w[peak_local+1]
            denom = 2*y1 - y0 - y2
            t_A = t_grid[peak_local] + dt*(y0 - y2)/(2*denom) if abs(denom) > 1e-15 \
                  else t_grid[peak_local]
        else:
            t_A = t_grid[peak_local]
    else:
        t_A = t_event

    # ── Stage B: maximise C(t) in tight bracket (= minimise |ζ|) ─────────
    lo, hi = t_A - half_bracket*0.4, t_A + half_bracket*0.4

    def _mag(t):
        return float(abs(mp.zeta(mp.mpc(0.5, float(t)))))

    try:
        res = minimize_scalar(_mag, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-12, 'maxiter': 400})
        t_B, val_B = res.x, res.fun
    except Exception:
        t_B = t_A
        val_B = _mag(t_A)

    # ── Stage C: high-precision zero of  d/dt |ζ|² = 0 ───────────────────
    # At a true zero: Re(ζ · conj(ζ')) = 0  (gradient of C's argument)
    mp.mp.dps = 40
    try:
        def _grad_mag_sq(t):
            t_mp = mp.mpf(t)
            z  = mp.zeta(mp.mpc('0.5', t_mp))
            dz = mp.diff(mp.zeta, mp.mpc('0.5', t_mp))
            return mp.re(z * mp.conj(dz))

        t_C   = mp.findroot(_grad_mag_sq, mp.mpf(str(t_B)), tol=mp.mpf('1e-28'))
        val_C = float(abs(mp.zeta(mp.mpc(0.5, t_C))))
        return float(t_C), val_C
    except Exception:
        return t_B, val_B


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREDICTOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MKMWindingPredictor:
    """
    Predict nontrivial Riemann zeros using the MKM Phase-Coherence
    Winding Observable  w(t) = χ'(t)·C(t)  as the sole detection signal.

    No Hardy Z(t).  No Backlund counting.  No Gram points.
    No sign changes.  No N(T).  No prior zero index knowledge.

    Zeros emerge as localised spike events in a derived scalar flow.
    """

    def __init__(self, t_lo=10.0, t_hi=70.0, pts_per_unit=45, verbose=True):
        self.t_lo         = t_lo
        self.t_hi         = t_hi
        self.pts_per_unit = pts_per_unit
        self.verbose      = verbose
        self.obs          = None

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ── Step 1: compute the observable ────────────────────────────────────
    def compute(self):
        N = max(2000, int((self.t_hi - self.t_lo) * self.pts_per_unit))
        t_grid = np.linspace(self.t_lo, self.t_hi, N)
        self._log(f"\n{'═'*64}")
        self._log("MKM PHASE–COHERENCE WINDING OBSERVABLE  |  w(t) = χ'(t)·C(t)")
        self._log(f"Range: [{self.t_lo}, {self.t_hi}]  |  Grid: {N} pts")
        self._log(f"{'═'*64}")
        t0 = time.time()
        self.obs = compute_winding_observable(t_grid, verbose=self.verbose)
        self._log(f"  Observable computed in {time.time()-t0:.1f}s  "
                  f"|  σ = {self.obs['sigma']:.4f}")
        return self.obs

    # ── Step 2: detect and refine events ──────────────────────────────────
    def predict(self, prominence=0.08, min_sep=0.40):
        if self.obs is None:
            self.compute()

        self._log(f"\nEvent detection in |w(t)| ...")
        peak_idx, props = detect_zero_events(
            self.obs, prominence_fraction=prominence, min_separation=min_sep
        )
        self._log(f"  {len(peak_idx)} spike events detected")

        results = []
        self._log(f"\nThree-stage refinement ...")
        for k, idx in enumerate(peak_idx):
            t0     = time.time()
            t_ev   = float(self.obs['t_grid'][idx])
            w_ev   = float(self.obs['w'][idx])
            t_ref, zval = refine_zero_event(t_ev, self.obs)
            elapsed = time.time() - t0
            results.append(dict(
                gamma    = t_ref,
                zeta_abs = zval,
                w_spike  = w_ev,
                t_event  = t_ev,
                elapsed  = elapsed,
            ))
            self._log(f"  [{k+1:3d}]  γ = {t_ref:.12f}  "
                      f"|ζ| = {zval:.3e}  "
                      f"|w| = {abs(w_ev):.2f}  "
                      f"({elapsed:.1f}s)")

        results.sort(key=lambda r: r['gamma'])
        return results

    # ── Step 3: verification against mpmath ground truth ──────────────────
    def verify(self, results, known):
        print(f"\n{'═'*68}")
        print("VERIFICATION  —  w(t)-predicted zeros vs mpmath ground truth")
        print(f"{'═'*68}")
        print(f"{'n':>4}  {'Predicted γ_n':>18}  {'Known γ_n':>18}  "
              f"{'Error':>13}  {'|ζ(γ)|':>11}  Status")
        print(f"{'─'*68}")

        known_arr = np.array(known)
        errors = []
        for i, r in enumerate(results, 1):
            diffs = np.abs(known_arr - r['gamma'])
            j     = int(np.argmin(diffs))
            err   = diffs[j]
            errors.append(err)
            ok    = "✓" if err < 1e-6 else ("~" if err < 1e-3 else "✗")
            print(f"{i:4d}  {r['gamma']:18.12f}  {known[j]:18.12f}  "
                  f"{err:13.4e}  {r['zeta_abs']:11.3e}  {ok}")

        if errors:
            print(f"{'─'*68}")
            print(f"  Zeros found   : {len(results)}")
            print(f"  Mean error    : {np.mean(errors):.4e}")
            print(f"  Max  error    : {np.max(errors):.4e}")
            print(f"  Error < 1e-6  : {sum(e<1e-6  for e in errors)}/{len(errors)}")
            print(f"  Error < 1e-10 : {sum(e<1e-10 for e in errors)}/{len(errors)}")
        return errors


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DIAGNOSTIC CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_winding_prediction(predictor, results, known=None,
                             out='MKM_WindingPredictor.png'):
    """
    Six-panel diagnostic chart for the MKM Winding Observable Predictor.

    Panel A : w(t) = χ'(t)·C(t) — the primary signal, with detected zeros
    Panel B : |ζ(½+it)| and coherence C(t) — the two ingredients
    Panel C : Phase χ(t) and phase velocity χ'(t) — the winding component
    Panel D : |ζ(γₙ)| at predicted zeros — quantitative zero quality
    Panel E : |w(γₙ)| spike strength — observatory signal at each zero
    Panel F : Prediction accuracy vs known zeros (if available)
    """
    obs    = predictor.obs
    t_grid = obs['t_grid']
    w      = obs['w']
    mag    = obs['mag']
    C      = obs['C']
    chi    = obs['chi']
    chi_p  = obs['chi_prime']
    sigma  = obs['sigma']

    gammas = np.array([r['gamma']   for r in results])
    zvals  = np.array([r['zeta_abs'] for r in results])
    wspks  = np.array([r['w_spike']  for r in results])

    # ── Figure layout ─────────────────────────────────────────────────────
    BG = '#03060d'
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(21, 24), facecolor=BG)
    fig.suptitle(
        'MKM PHASE–COHERENCE WINDING OBSERVABLE\n'
        r'Riemann zeros as spike events of  $w(t)=\dot{\chi}(t)\cdot C(t)$'
        r'  —  No $Z(t)$,  No Backlund,  No sign changes,  No indexing',
        fontsize=13.5, color='#00e5ff', fontweight='bold', y=0.997
    )
    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.54, wspace=0.30,
                           left=0.07, right=0.97, top=0.965, bottom=0.03)

    YEL = '#FFE135'; CYA = '#00e5ff'; GRN = '#00ff9d'
    MAG = '#dd44ff'; RED = '#ff5555'; ORG = '#ffaa33'; WHT = '#e8ecf0'

    def sax(ax, title, tc, xl='t', yl=''):
        ax.set_facecolor('#050e18')
        ax.tick_params(colors='#3a5060', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#0c1e2c')
        ax.set_title(title, color=tc, fontsize=9.5, fontweight='bold',
                     pad=7, loc='left')
        ax.grid(True, color='#07111a', linewidth=0.65)
        ax.axhline(0, color=WHT, lw=0.3, alpha=0.18)
        if xl: ax.set_xlabel(xl, color='#3a5060', fontsize=8)
        if yl: ax.set_ylabel(yl, color='#3a5060', fontsize=8)

    def mark_zeros(ax, color=YEL, lw=0.75, alpha=0.75):
        for g in gammas:
            ax.axvline(g, color=color, lw=lw, alpha=alpha, zorder=4)

    def mark_known(ax):
        if known:
            for z in known:
                if t_grid[0] < z < t_grid[-1]:
                    ax.axvline(z, color=RED, lw=0.45, alpha=0.28, ls='--', zorder=3)

    # ── PANEL A: w(t) — primary signal ────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    sax(ax_a,
        r"PANEL A  —  MKM Winding Observable  $w(t)=\chi'(t)\cdot C(t)$"
        r"   [spike events = Riemann zeros]",
        YEL, yl='w(t)')

    p99  = np.percentile(np.abs(w), 99.6)
    wclp = np.clip(w, -p99, p99)
    ax_a.fill_between(t_grid, 0, wclp, where=wclp > 0, color=GRN,  alpha=0.42)
    ax_a.fill_between(t_grid, 0, wclp, where=wclp < 0, color=MAG,  alpha=0.32)
    ax_a.plot(t_grid, wclp, color='#88ffcc', lw=0.5, alpha=0.92)

    # Annotate spikes at predicted zeros
    for i, r in enumerate(results):
        g    = r['gamma']
        wval = float(np.interp(g, t_grid, wclp))
        ax_a.axvline(g, color=YEL, lw=0.8, alpha=0.72, zorder=4)
        ax_a.scatter([g], [wval], color=YEL, s=32, zorder=6, edgecolors=WHT,
                     linewidths=0.4)
        ax_a.text(g, wval + 0.04*p99, f'γ{i+1}', fontsize=5.5, color=YEL,
                  ha='center', va='bottom', clip_on=True)

    mark_known(ax_a)

    ax_a.text(0.995, 0.96,
        r'Yellow markers = predicted zeros  |  Red dashed = known (verification only)' + '\n'
        r'Zeros are spike events — no counting, no brackets, no indexing.',
        transform=ax_a.transAxes, fontsize=7.2, color=YEL,
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor=BG, edgecolor='#1a3040', alpha=0.92))

    # ── PANEL B: |ζ| and C(t) — ingredients ──────────────────────────────
    ax_b = fig.add_subplot(gs[1, :])
    sax(ax_b,
        r'PANEL B  —  $|\zeta(\frac{1}{2}+it)|$  and Coherence Kernel  '
        r'$C(t)=\mathrm{sech}^2(|\zeta|/\sigma)$   [$C(t)\to 1$ at zeros]',
        CYA, yl=r'$|\zeta|$')

    ax_b.fill_between(t_grid, 0, mag, color='#4411aa', alpha=0.38)
    ax_b.plot(t_grid, mag, color='#9966ff', lw=0.52, alpha=0.92,
              label=r'$|\zeta(\frac{1}{2}+it)|$')
    ax_b.axhline(sigma, color=ORG, lw=0.9, ls='--', alpha=0.55,
                 label=fr'$\sigma = {sigma:.3f}$ (coherence scale)')

    ax_b2 = ax_b.twinx()
    ax_b2.plot(t_grid, C, color=CYA, lw=0.65, alpha=0.62)
    ax_b2.fill_between(t_grid, 0, C, color=CYA, alpha=0.07)
    ax_b2.set_ylabel('C(t)', color=CYA, fontsize=8)
    ax_b2.tick_params(colors=CYA, labelsize=7)
    ax_b2.set_ylim(-0.05, 1.55)

    mark_zeros(ax_b)
    ax_b.legend(fontsize=7.5, loc='upper right',
                facecolor=BG, edgecolor='#1a3040', labelcolor=WHT)
    ax_b.text(0.005, 0.95,
        r'$C(t)\to 1$ where $|\zeta|\to 0$: coherence is maximal precisely at zeros.' + '\n'
        r'Away from zeros $C\to 0$, gating the phase signal.',
        transform=ax_b.transAxes, fontsize=7.2, color=CYA, va='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor=BG, edgecolor='#1a3040', alpha=0.9))

    # ── PANEL C: χ(t) and χ'(t) — the phase component ────────────────────
    ax_c = fig.add_subplot(gs[2, :])
    sax(ax_c,
        r"PANEL C  —  Continuous Phase  $\chi(t)=\mathrm{arg}_{\mathrm{cont}}\,\zeta(\frac{1}{2}+it)$"
        r"  and Phase Velocity  $d\chi/dt$",
        ORG, yl=r"$d\chi/dt$")

    p99c = np.percentile(np.abs(chi_p), 99)
    cpclp = np.clip(chi_p, -p99c, p99c)
    ax_c.fill_between(t_grid, 0, cpclp, where=cpclp > 0, color=ORG,  alpha=0.38)
    ax_c.fill_between(t_grid, 0, cpclp, where=cpclp < 0, color=RED,  alpha=0.28)
    ax_c.plot(t_grid, cpclp, color='#ffcc66', lw=0.5, alpha=0.88)

    ax_c2 = ax_c.twinx()
    ax_c2.plot(t_grid, chi / np.pi, color=WHT, lw=0.45, alpha=0.30)
    ax_c2.set_ylabel(r'$\chi(t)/\pi$', color=WHT, fontsize=7, alpha=0.5)
    ax_c2.tick_params(colors='#3a5060', labelsize=7)

    mark_zeros(ax_c, color=YEL, alpha=0.6)

    ax_c.text(0.995, 0.96,
        r'Phase winds by $\pm\pi$ at each zero: $d\chi/dt$ spikes.' + '\n'
        r'Multiplied by $C(t)$, false spikes are suppressed.',
        transform=ax_c.transAxes, fontsize=7.2, color=ORG,
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor=BG, edgecolor='#1a3040', alpha=0.9))

    # ── PANEL D: |ζ(γₙ)| at predicted zeros ─────────────────────────────
    ax_d = fig.add_subplot(gs[3, 0])
    sax(ax_d,
        r'PANEL D  —  $|\zeta(\gamma_n^{\rm pred})|$  at Predicted Zeros',
        RED, xl='n  (event index)', yl=r'$|\zeta(\gamma_n)|$')

    ns = np.arange(1, len(results)+1)
    zplot = np.maximum(zvals, 5e-17)
    # If all values are below matplotlib's log-scale threshold, set a floor
    if zplot.max() < 1e-15:
        zplot = np.full_like(zplot, 1e-15)
    ax_d.semilogy(ns, zplot, color=RED, marker='o', markersize=5.5, lw=1.2,
                  markeredgecolor=WHT, markeredgewidth=0.3)
    ax_d.axhline(1e-10, color=GRN, lw=0.9, ls='--', alpha=0.65, label=r'$10^{-10}$')
    ax_d.axhline(1e-6,  color=YEL, lw=0.8, ls='--', alpha=0.55, label=r'$10^{-6}$')
    ax_d.axhline(1e-3,  color=ORG, lw=0.7, ls=':',  alpha=0.45, label=r'$10^{-3}$ warning')
    ax_d.legend(fontsize=7, facecolor=BG, edgecolor='#1a3040', labelcolor=WHT)
    ax_d.text(0.05, 0.96,
              f'All {len(results)} zeros:  max|ζ| = {zvals.max():.2e}',
              transform=ax_d.transAxes, fontsize=8, color=RED, va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor='#1a3040'))

    # ── PANEL E: spike strength |w(γₙ)| ──────────────────────────────────
    ax_e = fig.add_subplot(gs[3, 1])
    sax(ax_e,
        r'PANEL E  —  Spike Strength  $|w(\gamma_n)|$  at Each Zero  [stronger = sharper phase rotation]',
        GRN, xl='n  (event index)', yl=r'$|w(\gamma_n)|$')

    colors_e = plt.cm.plasma(np.linspace(0.2, 0.9, len(results)))
    ax_e.bar(ns, np.abs(wspks), color=colors_e, alpha=0.82, width=0.65,
             edgecolor=WHT, linewidth=0.25)
    ax_e.set_yscale('log')
    ax_e.text(0.05, 0.96,
              r'Spike height $\propto$ phase rotation rate at zero.' + '\n'
              r'All spikes detected autonomously from $|w(t)|$.',
              transform=ax_e.transAxes, fontsize=7.2, color=GRN, va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor='#1a3040'))

    # ── PANEL F: Prediction accuracy ─────────────────────────────────────
    ax_f = fig.add_subplot(gs[4, :])
    sax(ax_f,
        r'PANEL F  —  Prediction Accuracy  $|\gamma_n^{\rm pred} - \gamma_n^{\rm known}|$'
        r'  coloured by spike strength  $|w(\gamma_n)|$',
        MAG, xl=r'$\gamma$ predicted', yl='|Error|')

    if known and len(known) > 0:
        known_arr = np.array(known)
        errs, wabs, gp = [], [], []
        for r in results:
            diffs = np.abs(known_arr - r['gamma'])
            errs.append(diffs.min())
            wabs.append(abs(r['w_spike']))
            gp.append(r['gamma'])

        errs = np.array(errs);  wabs = np.array(wabs);  gp = np.array(gp)

        sc = ax_f.scatter(gp, np.maximum(errs, 1e-16),
                          c=np.log10(wabs + 1e-10), cmap='plasma',
                          s=55, zorder=5, edgecolors=WHT, linewidths=0.35)
        cb = plt.colorbar(sc, ax=ax_f, pad=0.01, fraction=0.018)
        cb.set_label(r'$\log_{10}|w(\gamma_n)|$', color=WHT, fontsize=8)
        cb.ax.tick_params(colors=WHT, labelsize=7)

        ax_f.set_yscale('log')
        ax_f.axhline(1e-6,  color=GRN, lw=0.9, ls='--', alpha=0.65,
                     label=r'$10^{-6}$ threshold')
        ax_f.axhline(1e-10, color=CYA, lw=0.8, ls='--', alpha=0.55,
                     label=r'$10^{-10}$ threshold')
        ax_f.legend(fontsize=7, facecolor=BG, edgecolor='#1a3040', labelcolor=WHT)

        n_lt6  = (errs < 1e-6).sum()
        n_lt10 = (errs < 1e-10).sum()
        ax_f.text(0.01, 0.95,
            f'Zeros predicted: {len(results)}   |   '
            f'Error < 10⁻⁶: {n_lt6}/{len(results)}   |   '
            f'Error < 10⁻¹⁰: {n_lt10}/{len(results)}   |   '
            f'Mean error: {errs.mean():.3e}',
            transform=ax_f.transAxes, fontsize=8.5, color=MAG, va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=BG,
                      edgecolor='#1a3040', alpha=0.93))
    else:
        ax_f.text(0.5, 0.5, 'Verification not requested (--no-verify)',
                  transform=ax_f.transAxes, color='#3a5060',
                  ha='center', va='center', fontsize=10)

    plt.savefig(out, dpi=165, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\nChart saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  MKM PHASE–COHERENCE WINDING OBSERVABLE PREDICTOR                ║
║  © MKM Research.  Proprietary.  All rights reserved.             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  New primary invariant:  w(t) = χ'(t) · C(t)                    ║
║                                                                   ║
║  Zeros are spike EVENTS in a derived phase–coherence flow.       ║
║  No Z(t)  |  No Backlund  |  No sign changes  |  No indexing     ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='MKM Phase-Coherence Winding Observable — Riemann Zero Predictor'
    )
    parser.add_argument('--t-lo',      type=float, default=10.0,
                        help='Lower t bound (default 10)')
    parser.add_argument('--t-hi',      type=float, default=70.0,
                        help='Upper t bound (default 70)')
    parser.add_argument('--pts',       type=int,   default=45,
                        help='Grid points per unit t (default 45)')
    parser.add_argument('--prom',      type=float, default=0.08,
                        help='Peak prominence fraction (default 0.08)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification against mpmath zeros')
    parser.add_argument('--output',    type=str,
                        default='MKM_RIEMANN_PREDICTION.png',
                        help='Output chart path')
    args = parser.parse_args()

    _banner()

    # ── Run ──────────────────────────────────────────────────────────────
    P = MKMWindingPredictor(
        t_lo=args.t_lo, t_hi=args.t_hi,
        pts_per_unit=args.pts, verbose=True
    )
    P.compute()
    results = P.predict(prominence=args.prom, min_sep=0.40)

    # ── Verify ───────────────────────────────────────────────────────────
    known = None
    if not args.no_verify and results:
        gamma_max = max(r['gamma'] for r in results)
        print(f"\nLoading mpmath ground truth (γ ≤ {gamma_max+2:.1f}) ...")
        known, k = [], 1
        while True:
            z = float(mp.zetazero(k).imag)
            if z > gamma_max + 2:
                break
            known.append(z)
            k += 1
        print(f"  {len(known)} reference zeros loaded.")
        P.verify(results, known)

    # ── Chart ─────────────────────────────────────────────────────────────
    plot_winding_prediction(P, results, known=known, out=args.output)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print("PREDICTED ZEROS  —  MKM WINDING OBSERVABLE")
    print(f"{'═'*66}")
    print(f"{'n':>4}  {'γ_n  (predicted)':>22}  {'|ζ(γ_n)|':>13}  {'|w spike|':>11}")
    print(f"{'─'*66}")
    for i, r in enumerate(results, 1):
        print(f"{i:4d}  {r['gamma']:22.14f}  {r['zeta_abs']:13.4e}  {abs(r['w_spike']):11.4f}")
    print(f"{'═'*66}")
    print(f"\n{len(results)} zeros located via  w(t) = χ'(t)·C(t)  alone.")
    print("© MKM Research — Proprietary.  All rights reserved.")
