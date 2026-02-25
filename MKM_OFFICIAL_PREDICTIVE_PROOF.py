#!/usr/bin/env python3

import sys
import logging
import numpy as np
import mpmath as mp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

mp.mp.dps = 30

DEFAULT_HEIGHT = 15.0
N_ZEROS = 15


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
    original_dps = mp.mp.dps
    mp.mp.dps = 22
    log.info("Evaluating winding observable on %d points", len(t_grid))
    mags, chi = zeta_batch_parallel(t_grid)
    mask = mags < 3.0
    if mask.sum() < len(mags) * 0.8:
        t_grid = t_grid[mask]
        mags = mags[mask]
        chi = chi[mask]
        log.info("Pre-filter: kept %d points (%.1f%%)", len(t_grid), 100 * mask.sum() / len(mask))
    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.1
    chi_prime = np.gradient(chi, dt)
    if sigma is None:
        sigma = np.percentile(mags[mags > 1e-8], 20)
    C = np.exp(-2 * np.log(np.cosh(mags / sigma)))
    w = chi_prime * C
    mp.mp.dps = original_dps
    log.info("Winding observable computed")
    return w, C, dt, t_grid


def winding_time(t_grid, w, dt):
    return np.cumsum(np.abs(w)) * dt


def pure_local_refine(t_grid, w, C, peak_idx, half_width=None):
    original_dps = mp.mp.dps
    mp.mp.dps = 30
    if half_width is None:
        half_width = 10 if peak_idx > 100 else 15
    start = max(0, peak_idx - half_width)
    end = min(len(t_grid), peak_idx + half_width + 1)
    local_t = t_grid[start:end]
    local_C = C[start:end]
    local_w = np.abs(w[start:end])
    i_max = np.argmax(local_C)
    t0 = local_t[i_max]
    tanh2 = np.tanh(local_t - t0)**2
    weights = local_C * (1 - tanh2)
    if weights.sum() > 0:
        t_corr = np.average(local_t, weights=weights)
        t_final = 0.65 * t0 + 0.35 * t_corr
    else:
        t_final = t0
    if 3 < i_max < len(local_w) - 4:
        x = local_t - t_final
        y = local_w
        spike_sharpness = local_w.max() / np.mean(local_w) if np.mean(local_w) > 0 else 1
        if spike_sharpness > 8:
            coeffs = np.polyfit(x, y, 3)
            a, b, c, d = coeffs
            disc = b**2 - 3 * a * c
            if disc > 0:
                dt1 = (-b + np.sqrt(disc)) / (3 * a)
                dt2 = (-b - np.sqrt(disc)) / (3 * a)
                dt_corr = dt1 if abs(dt1) < abs(dt2) else dt2
                t_final += 0.8 * dt_corr
        else:
            try:
                coeffs = np.polyfit(x, y, 5)
                deriv_coeffs = np.polyder(coeffs)
                roots = np.roots(deriv_coeffs)
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
                if real_roots:
                    dt_corr = min(real_roots, key=abs)
                    if abs(dt_corr) < 0.5 * (local_t[1] - local_t[0]):
                        t_final += 0.9 * dt_corr
            except Exception:
                coeffs = np.polyfit(x, y, 3)
                a, b, c, d = coeffs
                disc = b**2 - 3 * a * c
                if disc > 0:
                    dt1 = (-b + np.sqrt(disc)) / (3 * a)
                    dt2 = (-b - np.sqrt(disc)) / (3 * a)
                    dt_corr = dt1 if abs(dt1) < abs(dt2) else dt2
                    t_final += 0.8 * dt_corr
    mp.mp.dps = original_dps
    return t_final


def find_and_refine_zeros(t_grid, w, C, dt, prominence=0.05, min_dist=0.1):
    abs_w = np.abs(w)
    peaks, _ = find_peaks(abs_w, prominence=prominence * abs_w.max(), distance=int(min_dist / dt))
    log.info("Refining %d candidate zeros with MKM dual-kernel", len(peaks))
    refined = []
    for i, p in enumerate(peaks):
        zero_estimate = pure_local_refine(t_grid, w, C, p, half_width=15)
        refined.append(zero_estimate)
        log.debug("Zero #%d: γ = %.12f", i + 1, zero_estimate)
    return np.array(refined)


def predict_zeros_near_height(T_center, n_zeros=N_ZEROS, points_per_gap=120):
    avg_gap = 2 * np.pi / np.log(max(T_center / (2 * np.pi), 1.01))
    half_window = avg_gap * (n_zeros + 5)
    t_start = max(10.0, T_center - half_window)
    t_end = T_center + half_window

    log.info("Jumping to height T = %.6e", T_center)
    log.info("Average zero spacing at this height: %.8f", avg_gap)
    log.info("Search window: [%.4f, %.4f]", t_start, t_end)

    n_points = int((t_end - t_start) / avg_gap * points_per_gap)
    t_grid = np.linspace(t_start, t_end, n_points)
    log.info("Grid: %d points", n_points)

    w, C, dt, t_filtered = winding_observable(t_grid)
    zeros = find_and_refine_zeros(t_filtered, w, C, dt)
    return zeros, t_filtered, w, C, dt


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
    zeros = find_and_refine_zeros(t_filtered, w, C, dt)
    return zeros, t_filtered, w, C, dt


def report_zeros(zeros, t_filtered, w, dt):
    known_first15 = np.array([
        14.13472514173469, 21.02203963877155, 25.01085758014569,
        30.42487612585952, 32.93506158773919, 37.58617815882567,
        40.91871901214750, 43.32707328091499, 48.00515088116716,
        49.77383247767230, 52.97032147771446, 56.44624769706340,
        59.34704400260235, 60.83177852460981, 65.11254404808162
    ])
   # https://www.lmfdb.org/zeros/zeta/?t=30610045974&count=15 Validate your generated numbers - officially.
    log.info("=" * 60)
    log.info("ZERO ESTIMATES")
    log.info("=" * 60)

    # Check if we're searching at a high height where error comparison doesn't make sense
    max_known_zero = known_first15.max()  # ~65.11
    disable_error_calc = len(zeros) > 0 and zeros[0] > 3 * max_known_zero

    errors = []
    for i, z in enumerate(zeros[:N_ZEROS], 1):
        if not disable_error_calc and i <= len(known_first15):
            error = abs(z - known_first15[i - 1])
            errors.append(error)
            pct = error / abs(known_first15[i - 1]) * 100
            dp = -np.log10(error) if error > 0 else 15
            log.info("Zero #%2d:  γ = %.12f  (%.4f%% error, %.1f dp)", i, z, pct, dp)
        else:
            log.info("Zero #%2d:  γ = %.12f", i, z)

    # if errors:
    #     avg_pct = np.mean([(e / abs(known_first15[i])) * 100 for i, e in enumerate(errors)])
    #     max_dp = -np.log10(max(errors))
    #     log.info("Average error: %.4f%%  →  %.1f decimal places", avg_pct, max_dp)

    tau = winding_time(t_filtered, w, dt)
    n = np.arange(1, len(zeros) + 1)
    zero_indices = np.clip(np.searchsorted(t_filtered, zeros), 0, len(tau) - 1)

    plt.figure(figsize=(10, 6))
    plt.plot(n, tau[zero_indices], "o-", ms=3)
    plt.xlabel("Zero index n")
    plt.ylabel("Winding time τ(n)")
    plt.title("Riemann zeros — uniform spacing in winding-time τ\nMKM dual-kernel: sech² + tanh²")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = "MKM_OFFICIAL_PREDICTIVE_PROOF.png"
    plt.savefig(out_path, dpi=600)
    log.info("Chart saved to %s", out_path)
    plt.show()


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("fork", force=True)
    except Exception:
        pass

    log.info("Winding-Time Riemann Zero Locator — MKM Dual-Kernel")
    log.info("CPUs available: %d", cpu_count())

    if len(sys.argv) > 1:
        try:
            height = float(sys.argv[1])
        except ValueError:
            log.error("Argument must be a number, e.g.:  python riemann_zero_locator.py 1000")
            sys.exit(1)
        log.info("Height argument received: %.6e", height)
        zeros, t_filtered, w, C, dt = predict_zeros_near_height(height, n_zeros=N_ZEROS)
    else:
        log.info("No height argument — using default scan")
        zeros, t_filtered, w, C, dt = run_default(n_zeros=N_ZEROS)

    report_zeros(zeros, t_filtered, w, dt)