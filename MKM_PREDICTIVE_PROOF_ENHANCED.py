#!/usr/bin/env python3

import sys
import logging
import numpy as np
import mpmath as mp
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

mp.mp.dps = 22

DEFAULT_HEIGHT = 15.0
N_ZEROS = 15

# ============================================================================
# MKM HOOK (IP-SAFE)
# ============================================================================

def mkm_score(gamma: float) -> float:
    """
    MKM bridge distance H(gamma). By default, returns a neutral constant.
    In your private environment, replace this with:
        from mkm_bridge_core import mkm_score as _impl
        return _impl(gamma)
    """
    return 1.0  # neutral


def combined_peak_score(gamma: float, w_val: float) -> float:
    """Combined salience for MKM-enhanced refinement."""
    H = mkm_score(gamma)
    return abs(w_val) / (1.0 + H)


# ============================================================================
# GRAM POINT CENTRING
# ============================================================================

def gram_point_near(T: float) -> float:
    """Snap T to nearest Gram point via Siegel theta θ(t) = nπ."""
    old = mp.mp.dps
    mp.mp.dps = 40
    try:
        t0 = mp.mpf(T)
        theta0 = mp.siegeltheta(t0)
        n = int(theta0 / mp.pi)
        t_star = mp.findroot(lambda t: mp.siegeltheta(t) - n * mp.pi, t0)
        return float(t_star)
    finally:
        mp.mp.dps = old


# ============================================================================
# ORIGINAL LOW-HEIGHT TRUTH (FIRST 15 ZEROS)
# ============================================================================

KNOWN_FIRST15 = np.array([
    14.13472514173469, 21.02203963877155, 25.01085758014569,
    30.42487612585952, 32.93506158773919, 37.58617815882567,
    40.91871901214750, 43.32707328091499, 48.00515088116716,
    49.77383247767230, 52.97032147771446, 56.44624769706340,
    59.34704400260235, 60.83177852460981, 65.11254404808162
])


# ============================================================================
# ζ CORE (WITH ENHANCEMENTS)
# ============================================================================

def compute_zeta_chunk(chunk):
    mp.mp.dps = 22
    zvals = [mp.zeta(mp.mpc(0.5, float(t))) for t in chunk]
    re = np.array([float(mp.re(z)) for z in zvals])
    im = np.array([float(mp.im(z)) for z in zvals])
    return re, im


def zeta_batch_parallel(t_grid, chunks=None):
    if chunks is None:
        chunks = max(1, min(cpu_count() - 1, 8))
    log.info("Parallel ζ on %d cores (%d points)", chunks, len(t_grid))
    splits = np.array_split(t_grid, chunks)
    try:
        with Pool(processes=chunks) as pool:
            results = pool.map(compute_zeta_chunk, splits)
        re_all = np.concatenate([r[0] for r in results])
        im_all = np.concatenate([r[1] for r in results])
        mags = np.sqrt(re_all**2 + im_all**2)
        args = np.unwrap(np.arctan2(im_all, re_all))
        log.info("Parallel batch complete")
        return mags, args
    except Exception as exc:
        log.warning("Multiprocessing failed (%s), falling back to serial", exc)
        return zeta_batch_serial(t_grid)


def zeta_batch_serial(t_grid):
    log.info("Serial ζ computation (%d points)", len(t_grid))
    zvals = [mp.zeta(mp.mpc(0.5, float(t))) for t in t_grid]
    re = np.array([float(mp.re(z)) for z in zvals])
    im = np.array([float(mp.im(z)) for z in zvals])
    mags = np.sqrt(re**2 + im**2)
    args = np.unwrap(np.arctan2(im, re))
    log.info("Serial batch complete")
    return mags, args


def make_adaptive_grid(t_start, t_end, points_per_gap=80):
    t = [t_start]
    while t[-1] < t_end:
        gap = np.log(t[-1] + 10) / (2 * np.pi)
        step = gap / points_per_gap
        t.append(t[-1] + max(step, 0.001))
    return np.array(t)


def winding_observable(t_grid, sigma=None):
    log.info("Evaluating winding observable on %d points", len(t_grid))
    mags, chi = zeta_batch_parallel(t_grid)

    mask = mags < 3.0
    if mask.sum() < len(mags) * 0.8:
        t_grid = t_grid[mask]
        mags = mags[mask]
        chi = chi[mask]
        log.info("Pre-filter: kept %d points (%.1f%%)", len(t_grid), 100 * mask.sum() / len(mask))

    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.1
    chi_prime = np.empty_like(chi)
    chi_prime[1:-1] = (chi[2:] - chi[:-2]) / (2 * dt)
    chi_prime[0] = chi_prime[1]
    chi_prime[-1] = chi_prime[-2]

    if sigma is None:
        sigma = np.percentile(mags[mags > 1e-8], 20)

    # Gaussian-like magnitude suppression
    C = np.exp(-(mags / sigma) ** 2)
    w = chi_prime * C

    log.info("Winding observable computed")
    return w, C, dt, t_grid


def winding_time(t_grid, w, dt):
    return np.cumsum(np.abs(w)) * dt


def pure_local_refine(t_grid, w, C, peak_idx, half_width=None, w_interp=None):
    if half_width is None:
        half_width = 10 if peak_idx > 100 else 15

    start = max(0, peak_idx - half_width)
    end = min(len(t_grid), peak_idx + half_width + 1)
    local_t = t_grid[start:end]
    local_C = C[start:end]
    local_w = np.abs(w[start:end])

    # C-based centroid (bias toward winding spikes)
    i_max = np.argmax(local_C)
    t0 = local_t[i_max]
    tanh2 = np.tanh(local_t - t0) ** 2
    weights = local_C * (1 - tanh2) * (local_w / (local_w.max() + 1e-12))
    if weights.sum() > 0:
        t_corr = np.average(local_t, weights=weights)
        t_final = 0.65 * t0 + 0.35 * t_corr
    else:
        t_final = t0

    # Parabolic interpolation around |w| maximum
    if 0 < i_max < len(local_w) - 1:
        y0, y1, y2 = local_w[i_max - 1:i_max + 2]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            dt_loc = local_t[1] - local_t[0]
            t_final = local_t[i_max] + delta * dt_loc

    # Optional MKM-guided local snap (IP-safe)
    try:
        window = 0.3
        mask = (t_grid >= t_final - window) & (t_grid <= t_final + window)
        local_ts = t_grid[mask]
        if len(local_ts) >= 5:
            if w_interp is None:
                w_func = lambda x: np.interp(x, t_grid, w)
            else:
                w_func = lambda x: float(w_interp(x))
            costs = [abs(w_func(t)) + mkm_score(t) for t in local_ts]
            t_star = local_ts[int(np.argmin(costs))]

            confidence = abs(w_func(t_final))
            alpha = min(0.6, 1.0 / (1.0 + confidence))
            t_final = (1.0 - alpha) * t_final + alpha * t_star
    except Exception:
        pass

    return t_final


def filter_peaks_one_per_gap(peaks, dt):
    """Enforce ≈one peak per expected gap."""
    if len(peaks) <= 1:
        return peaks
    gaps = np.diff(peaks) * dt
    expected_gap = float(np.mean(gaps))
    if expected_gap <= 0:
        return peaks
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if (p - filtered[-1]) * dt > 0.7 * expected_gap:
            filtered.append(p)
    return np.array(filtered, dtype=int)


def find_and_refine_zeros(t_grid, w, C, dt, prominence=0.05, min_dist=0.1):
    # ζ zeros correspond to negative-to-positive phase crossings => lock polarity
    signed_w = w
    abs_w = np.where(signed_w > 0, signed_w, 0.0)
    # precompute interpolant to avoid repeated np.interp calls
    try:
        w_interp = interp1d(t_grid, w, kind="linear", assume_sorted=True)
    except Exception:
        w_interp = None
    peaks, _ = find_peaks(
        abs_w,
        prominence=prominence * abs_w.max(),
        distance=int(min_dist / dt)
    )

    peaks = filter_peaks_one_per_gap(peaks, dt)
    log.info("Refining %d candidate zeros with MKM dual-kernel", len(peaks))

    refined = []
    refined_mkm = []

    for i, p in enumerate(peaks):
        zero_estimate = pure_local_refine(t_grid, w, C, p, half_width=15, w_interp=w_interp)
        refined.append(zero_estimate)

        # MKM-enhanced refinement (IP-safe)
        try:
            window = 0.3
            mask = (t_grid >= zero_estimate - window) & (t_grid <= zero_estimate + window)
            local_ts = t_grid[mask]
            if len(local_ts) >= 5:
                if w_interp is None:
                    w_func = lambda x: np.interp(x, t_grid, w)
                else:
                    w_func = lambda x: float(w_interp(x))
                scores = [combined_peak_score(t, w_func(t)) for t in local_ts]
                best_t = local_ts[int(np.argmax(scores))]
                confidence = abs(w_func(zero_estimate))
                alpha = min(0.6, 1.0 / (1.0 + confidence))
                enhanced = (1.0 - alpha) * zero_estimate + alpha * best_t
            else:
                enhanced = zero_estimate
        except Exception:
            enhanced = zero_estimate

        refined_mkm.append(enhanced)
        log.debug("Zero #%d: γ_raw = %.12f, γ_mkm = %.12f", i + 1, zero_estimate, enhanced)

    return np.array(refined), np.array(refined_mkm)


def predict_zeros_near_height(T_center, n_zeros=N_ZEROS, points_per_gap=120):
    # Gram-align center for better global centring
    T_center = float(gram_point_near(T_center))

    avg_gap = 2 * np.pi / np.log(max(T_center / (2 * np.pi), 1.01))
    half_window = avg_gap * (n_zeros + 5)
    t_start = max(10.0, T_center - half_window)
    t_end = T_center + half_window

    log.info("Jumping to height T = %.6e (Gram-aligned)", T_center)
    log.info("Average zero spacing at this height: %.8f", avg_gap)
    log.info("Search window: [%.4f, %.4f]", t_start, t_end)

    n_points = int((t_end - t_start) / avg_gap * points_per_gap)
    t_grid = np.linspace(t_start, t_end, n_points)
    log.info("Grid: %d points", n_points)

    w, C, dt, t_filtered = winding_observable(t_grid)
    zeros, zeros_mkm = find_and_refine_zeros(t_filtered, w, C, dt)
    return zeros, zeros_mkm, t_filtered, w, C, dt


def run_default(n_zeros=N_ZEROS):
    log.info("No height supplied — running default scan t=[10, 100]")

    coarse_grid = np.linspace(10, 100, 3000)
    w_coarse, _, _, coarse_filtered = winding_observable(coarse_grid)
    peaks_coarse, _ = find_peaks(np.abs(w_coarse), prominence=0.08)
    log.info("Coarse pass: %d candidate regions found", len(peaks_coarse))

    fine_regions = []
    for p in peaks_coarse:
        if p < len(coarse_filtered):
            t_c = coarse_filtered[p]
            fine_regions.append(np.linspace(t_c - 0.8, t_c + 0.8, 600))

    if fine_regions:
        t_grid = np.unique(np.concatenate(fine_regions))
        t_grid.sort()
        log.info("Fine grid: %d targeted points", len(t_grid))
    else:
        log.warning("No coarse candidates — falling back to adaptive grid")
        t_grid = make_adaptive_grid(10, 100, points_per_gap=90)
        log.info("Adaptive grid: %d points", len(t_grid))

    w, C, dt, t_filtered = winding_observable(t_grid)
    zeros, zeros_mkm = find_and_refine_zeros(t_filtered, w, C, dt)
    return zeros, zeros_mkm, t_filtered, w, C, dt


def report_zeros(zeros, zeros_mkm, t_filtered, w, dt):
    log.info("=" * 60)
    log.info("ZERO ESTIMATES (ζ-only vs MKM-enhanced)")
    log.info("=" * 60)
    log.info("Idx  γ_zeta           γ_mkm            |err_zeta|      |err_mkm|   "
             "improv×   dp_zeta  dp_mkm  dp_improv")

    errors_z = []
    errors_m = []

    for i in range(min(N_ZEROS, len(zeros))):
        z = zeros[i]
        zm = zeros_mkm[i] if i < len(zeros_mkm) else z

        if i < len(KNOWN_FIRST15):
            true = KNOWN_FIRST15[i]
            err_z = abs(z - true)
            err_m = abs(zm - true)
            errors_z.append(err_z)
            errors_m.append(err_m)

            improv = (err_z / err_m) if err_m > 0 else np.inf
            dp_z = -np.log10(err_z) if err_z > 0 else 15.0
            dp_m = -np.log10(err_m) if err_m > 0 else 15.0
            dp_improv = dp_m - dp_z

            log.info(
                " #%2d  %12.8f  %12.8f   %11.4e  %11.4e   %7.2f   %7.2f  %7.2f  %8.2f",
                i + 1, z, zm, err_z, err_m, improv, dp_z, dp_m, dp_improv
            )
        else:
            log.info(" #%2d  %12.8f  %12.8f", i + 1, z, zm)

    tau = winding_time(t_filtered, w, dt)
    n = np.arange(1, len(zeros) + 1)
    zero_indices = np.clip(np.searchsorted(t_filtered, zeros), 0, len(tau) - 1)

    plt.figure(figsize=(10, 6))
    plt.plot(n, tau[zero_indices], "o-", ms=3)
    plt.xlabel("Zero index n")
    plt.ylabel("Winding time τ(n)")
    plt.title("Riemann zeros — uniform spacing in winding-time τ\nMKM dual-kernel: Gaussian-weighted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = "MKM_PREDICTIVE_PROOF_ENHANCED.png"
    try:
        tmp_path = out_path + ".tmp.png"
        plt.savefig(tmp_path, dpi=300, bbox_inches="tight")
        import os
        os.replace(tmp_path, out_path)
        log.info("Chart saved to %s", out_path)
    except KeyboardInterrupt:
        log.warning("Plot save interrupted — skipping image write")
    except Exception as exc:
        log.warning("Plot save failed: %s", exc)
    finally:
        plt.close()


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("fork", force=True)
    except Exception:
        pass

    log.info("Winding-Time Riemann Zero Locator — MKM Dual-Kernel (Enhanced, IP-safe)")
    log.info("CPUs available: %d", cpu_count())

    if len(sys.argv) > 1:
        try:
            height = float(sys.argv[1])
        except ValueError:
            log.error("Argument must be a number, e.g.:  python riemann_zero_locator.py 1000")
            sys.exit(1)
        log.info("Height argument received: %.6e", height)
        zeros, zeros_mkm, t_filtered, w, C, dt = predict_zeros_near_height(
            height, n_zeros=N_ZEROS
        )
    else:
        log.info("No height argument — using default scan")
        zeros, zeros_mkm, t_filtered, w, C, dt = run_default(n_zeros=N_ZEROS)

    report_zeros(zeros, zeros_mkm, t_filtered, w, dt)
