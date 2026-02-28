#!/usr/bin/env python3
"""
MKM Universe, Beta Precision Proprietary Code — Not for commercial use.

This file is proprietary and may not be used for commercial purposes.
No license is granted.
"""

import csv
from decimal import Decimal, getcontext

import plotly.graph_objects as go

getcontext().prec = 100

ONE = Decimal(1)
TWO = Decimal(2)
THREE = Decimal(3)
FIVE = Decimal(5)

PHI = (ONE + FIVE.sqrt()) / TWO

TWO_THIRDS = TWO / THREE
FOUR_THIRDS = Decimal(4) / THREE

DELTA = ONE / (Decimal(64) * (THREE ** TWO_THIRDS) * (PHI ** FOUR_THIRDS))
DELTA_SQ = DELTA ** 2

DEBT_LEADING = DELTA_SQ / PHI


def unperturbed_term(n: int) -> Decimal:
    """Baseline golden series term: 2 / φ^n."""
    return TWO / (PHI ** Decimal(n))


def perturbed_term(n: int) -> Decimal:
    denom = (PHI ** (TWO * Decimal(n))) - DELTA_SQ
    return (TWO * (PHI ** Decimal(n))) / denom


def partial_sums(N_max: int):
    S0 = Decimal(0)
    SΔ = Decimal(0)
    rows = []

    for n in range(1, N_max + 1):
        S0 += unperturbed_term(n)
        SΔ += perturbed_term(n)
        debt = SΔ - S0
        rows.append({
            "n": n,
            "S_unpert": float(S0),
            "S_pert": float(SΔ),
            "Debt": float(debt),
        })
    return rows


def analytic_checks():
    print("=" * 72)
    print("MKM INFINITE SERIES VALIDATION")
    print("=" * 72)

    print("\n[1] Flexibility Envelope Δ and Δ²")
    print(f"φ              = {PHI}")
    print(f"Δ (analytic)   = {DELTA}")
    print(f"Δ²             = {DELTA_SQ}")

    FOUR_THIRDS = Decimal(4) / THREE
    EIGHT_THIRDS = Decimal(8) / THREE
    analytic_sq = ONE / (
        Decimal(4096) * (THREE ** FOUR_THIRDS) * (PHI ** EIGHT_THIRDS)
    )
    diff_sq = abs(DELTA_SQ - analytic_sq)

    print("\n[2] Squared envelope identity:")
    print(f"Δ² (direct)    = {DELTA_SQ}")
    print(f"Δ² (expanded)  = {analytic_sq}")
    print(f"|Δ² - expanded|= {diff_sq}")
    if diff_sq < Decimal("1e-70"):
        print("Verdict: squared envelope matches to 70+ digits.")
    else:
        print("Warning: discrepancy in squared envelope expression.")


def write_csv(rows, filename="MKM_INFINITE_SERIES_VALIDATOR.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "S_unpert", "S_pert", "Debt"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[3] Wrote partial sums and debts to {filename}")


def plot_partial_sums(rows, filename="MKM_INFINITE_SERIES_VALIDATOR.png"):
    x = [r["n"] for r in rows]
    y0 = [r["S_unpert"] for r in rows]
    yΔ = [r["S_pert"] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y0, mode="lines", name="S0(N)", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=x, y=yΔ, mode="lines", name="SΔ(N)"))

    fig.update_layout(
        title={"text": "Partial sums of golden series (N=1 to 10000)"},
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="N")
    fig.update_yaxes(title_text="Partial sum")

    fig.write_image(filename)
    with open(filename + ".meta.json", "w") as f:
        import json

        json.dump(
            {
                "caption": "Partial sums of unperturbed and perturbed golden series",
                "description": (
                    "Line chart showing S0(N) and SΔ(N) converging to nearby limits "
                    "as N increases to 10000."
                ),
            },
            f,
        )
    print(f"[4] Wrote chart: {filename}")


def plot_debt(rows, filename="MKM_INFINITE_SERIES_VALIDATOR_DEBT.png"):
    x = [r["n"] for r in rows]
    yD = [r["Debt"] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=yD, mode="lines", name="Debt(N)", fill="tozeroy"))

    fig.update_layout(
        title={"text": "Accumulated geometric debt D(N) (N=1 to 10000)"},
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="N")
    fig.update_yaxes(title_text="Debt")

    fig.write_image(filename)
    with open(filename + ".meta.json", "w") as f:
        import json

        json.dump(
            {
                "caption": "Accumulated geometric debt as N increases",
                "description": (
                    "Line chart showing D(N) = SΔ(N) - S0(N) growing monotonically "
                    "toward its limiting value as N increases to 10000."
                ),
            },
            f,
        )
    print(f"[5] Wrote chart: {filename}")


def main():
    analytic_checks()

    N_MAX = 10000
    print(f"\n[3] Computing partial sums up to N={N_MAX} ...")
    rows = partial_sums(N_MAX)

    final = rows[-1]
    S0N = Decimal(str(final["S_unpert"]))
    SΔN = Decimal(str(final["S_pert"]))
    debtN = Decimal(str(final["Debt"]))

    exact_unpert = TWO * PHI
    print("\n[4] Summary at N=10000")
    print(f"S0(10000)   ≈ {S0N}")
    print(f"2φ (exact) = {exact_unpert}")
    print(f"|S0 - 2φ|  ≈ {abs(S0N - exact_unpert)}")
    print(f"SΔ(10000)   ≈ {SΔN}")
    print(f"D(10000)    ≈ {debtN}")
    print(f"D_lead     ≈ Δ²/φ = {DEBT_LEADING}")
    print(f"|D(10000) - Δ²/φ| ≈ {abs(debtN - DEBT_LEADING)}")

    write_csv(rows)
    plot_partial_sums(rows)
    plot_debt(rows)

    print("\n[Done] Infinite series validation complete.")

if __name__ == "__main__":
    main()
