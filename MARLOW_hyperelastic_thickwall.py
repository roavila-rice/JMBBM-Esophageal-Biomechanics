"""
Compare three hyperelastic models for uniform external pressure.

Thick-walled incompressible cylinder, axisymmetric. External pressure
magnitude Pext > 0 is applied at r=b:
    sigma_r(b) = -Pext, sigma_r(a) = 0

Models:
    1. Mooney-Rivlin
    2. Ogden N=3, Abaqus mu_i and alpha_i form
    3. Marlow W(I1), built from uniaxial nominal stress-stretch data
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with the first value set to zero."""
    dx = np.diff(x)
    area = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate(([0.0], np.cumsum(area)))


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral without relying on NumPy's trapz/trapezoid names."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def pressure_from_a_MR(a: float, A: float, B: float, mu: float) -> float:
    """Mooney-Rivlin predicted pressure magnitude for a given a."""
    if a <= 0 or a >= A:
        return np.nan

    k = A**2 - a**2
    b = np.sqrt(B**2 - A**2 + a**2)

    def F(r: float | np.ndarray) -> float | np.ndarray:
        return np.log(r) - 0.5 * np.log(r**2 + k) - k / (2 * r**2)

    P = mu * (F(b) - F(a))
    return float(P)


def stresses_MR(
    r: np.ndarray,
    a: float,
    b: float,
    Pext: float,
    A: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mooney-Rivlin sigma_r and sigma_t."""
    k = A**2 - a**2

    def F(rr: float | np.ndarray) -> float | np.ndarray:
        return np.log(rr) - 0.5 * np.log(rr**2 + k) - k / (2 * rr**2)

    sigma_r = -Pext + mu * (F(b) - F(r))
    delta = -mu * k * (1 / (r**2 + k) + 1 / (r**2))
    sigma_t = sigma_r + delta
    return sigma_r, sigma_t


def delta_ogden_ABAQUS(
    r: np.ndarray,
    A: float,
    a: float,
    mu: np.ndarray,
    alp: np.ndarray,
) -> np.ndarray:
    """Delta = sigma_theta - sigma_r for the incompressible Abaqus Ogden form."""
    k = A**2 - a**2
    R = np.sqrt(r**2 + k)
    lam_t = r / R
    lam_r = R / r

    Delta = np.zeros_like(r)
    for mui, alpi in zip(mu, alp):
        Delta = Delta + (2 * mui / alpi) * (lam_t**alpi - lam_r**alpi)
    return Delta


def Pmodel_ogden(
    a: float,
    A: float,
    B: float,
    mu: np.ndarray,
    alp: np.ndarray,
    Nq: int,
) -> float:
    """Ogden predicted pressure magnitude for a given a."""
    if a <= 0 or a >= A:
        return np.nan

    b = np.sqrt(B**2 - A**2 + a**2)
    r = np.linspace(a, b, Nq)
    Delta = delta_ogden_ABAQUS(r, A, a, mu, alp)
    P = -trapz(Delta / r, r)
    return float(P)


def stresses_OGDEN(
    r: np.ndarray,
    a: float,
    b: float,
    Pext: float,
    A: float,
    mu: np.ndarray,
    alp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ogden sigma_r and sigma_t, enforcing sigma_r(b) = -Pext."""
    Delta = delta_ogden_ABAQUS(r, A, a, mu, alp)
    f = Delta / r
    I_r_to_b = cumtrapz(f[::-1], r[::-1])[::-1]
    sigma_r = -Pext + I_r_to_b
    sigma_t = sigma_r + Delta
    return sigma_r, sigma_t


def build_W1_from_uniaxial_CLAMPED(
    lambda_uni: np.ndarray,
    Pnom_uni: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build Marlow W1(I1) from uniaxial nominal stress-stretch data.

    I1 is clamped to the experimental range before interpolation. This matches
    the MATLAB code's stabilization for thick-wall states beyond the measured
    uniaxial stretch range.
    """
    lambda_uni = np.asarray(lambda_uni, dtype=float).reshape(-1)
    Pnom_uni = np.asarray(Pnom_uni, dtype=float).reshape(-1)

    denom = 2 * (lambda_uni - lambda_uni**-2)
    mask = np.abs(denom) > 1e-10

    lam = lambda_uni[mask]
    Pn = Pnom_uni[mask]

    I1 = lam**2 + 2 / lam
    W1 = Pn / (2 * (lam - lam**-2))

    idx = np.argsort(I1)
    I1s = I1[idx]
    W1s = W1[idx]

    I1min = I1s[0]
    I1max = I1s[-1]
    interpolator = PchipInterpolator(I1s, W1s, extrapolate=False)

    def W1fun(I1q: np.ndarray) -> np.ndarray:
        I1q_clamped = np.clip(I1q, I1min, I1max)
        return interpolator(I1q_clamped)

    return W1fun


def delta_marlow(
    r: np.ndarray,
    A: float,
    a: float,
    W1fun: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Delta = sigma_theta - sigma_r for incompressible Marlow W(I1)."""
    k = A**2 - a**2
    R = np.sqrt(r**2 + k)
    lam_t = r / R
    lam_r = R / r

    I1 = lam_r**2 + lam_t**2 + 1
    W1 = W1fun(I1)

    Delta = 2 * W1 * (lam_t**2 - lam_r**2)
    return Delta


def Pmodel_marlow(
    a: float,
    A: float,
    B: float,
    W1fun: Callable[[np.ndarray], np.ndarray],
    Nq: int,
) -> float:
    """Marlow predicted pressure magnitude for a given a."""
    if a <= 0 or a >= A:
        return np.nan

    b = np.sqrt(B**2 - A**2 + a**2)
    r = np.linspace(a, b, Nq)
    Delta = delta_marlow(r, A, a, W1fun)
    P = -trapz(Delta / r, r)
    return float(P)


def stresses_MARLOW(
    r: np.ndarray,
    a: float,
    b: float,
    Pext: float,
    A: float,
    W1fun: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Marlow sigma_r and sigma_t, enforcing sigma_r(b) = -Pext."""
    Delta = delta_marlow(r, A, a, W1fun)
    f = Delta / r
    I_r_to_b = cumtrapz(f[::-1], r[::-1])[::-1]
    sigma_r = -Pext + I_r_to_b
    sigma_t = sigma_r + Delta
    return sigma_r, sigma_t


def bracket_root_scan(g: Callable[[float], float], A: float) -> tuple[float, float]:
    """Robust bracket by scanning a in (0, A) on a log scale."""
    Nscan = 240
    a_hi = A * (1 - 1e-10)
    a_lo = max(1e-12 * A, np.finfo(float).tiny)

    a_grid = np.logspace(np.log10(a_hi), np.log10(a_lo), Nscan)
    g_grid = np.full(a_grid.shape, np.nan, dtype=float)

    for k, a_value in enumerate(a_grid):
        gk = g(float(a_value))
        if np.isfinite(gk):
            g_grid[k] = gk

    found = False
    aL = np.nan
    aU = np.nan
    for k in range(len(a_grid) - 1):
        if not np.isfinite(g_grid[k]) or not np.isfinite(g_grid[k + 1]):
            continue
        if g_grid[k] == 0:
            aL = a_grid[k]
            aU = a_grid[k]
            found = True
            break
        if np.sign(g_grid[k]) * np.sign(g_grid[k + 1]) < 0:
            aL = a_grid[k + 1]
            aU = a_grid[k]
            found = True
            break

    if not found:
        gf = g_grid[np.isfinite(g_grid)]
        raise RuntimeError(f"Failed to bracket root. g(a) ranged [{np.min(gf):.3e}, {np.max(gf):.3e}].")

    if aL == aU:
        aL = max(0.9 * aL, 1e-12 * A)
        aU = min(1.1 * aU, A * (1 - 1e-10))

    return float(aL), float(aU)


def solve_a_MR(
    Pext: float,
    A: float,
    B: float,
    mu: float,
    tol_f: float,
    tol_a: float,
    maxIter: int,
) -> float:
    """Solve the Mooney-Rivlin deformed inner radius."""

    def g(aa: float) -> float:
        return pressure_from_a_MR(aa, A, B, mu) - Pext

    aL, aU = bracket_root_scan(g, A)

    for _ in range(maxIter):
        aM = 0.5 * (aL + aU)
        gL = g(aL)
        gM = g(aM)
        if abs(gM) < tol_f or (aU - aL) < tol_a:
            return float(aM)
        if gL * gM > 0:
            aL = aM
        else:
            aU = aM

    raise RuntimeError("MR: Bisection did not converge.")


def solve_a_ogden(
    Pext: float,
    A: float,
    B: float,
    mu: np.ndarray,
    alp: np.ndarray,
    Nq: int,
    tol_f: float,
    tol_a: float,
    maxIter: int,
) -> float:
    """Solve the Ogden deformed inner radius."""

    def g(aa: float) -> float:
        return Pmodel_ogden(aa, A, B, mu, alp, Nq) - Pext

    aL, aU = bracket_root_scan(g, A)

    for _ in range(maxIter):
        aM = 0.5 * (aL + aU)
        gL = g(aL)
        gM = g(aM)
        if abs(gM) < tol_f or (aU - aL) < tol_a:
            return float(aM)
        if gL * gM > 0:
            aL = aM
        else:
            aU = aM

    raise RuntimeError("Ogden: Bisection did not converge.")


def solve_a_marlow(
    Pext: float,
    A: float,
    B: float,
    W1fun: Callable[[np.ndarray], np.ndarray],
    Nq: int,
    tol_f: float,
    tol_a: float,
    maxIter: int,
) -> float:
    """Solve the Marlow deformed inner radius."""

    def g(aa: float) -> float:
        return Pmodel_marlow(aa, A, B, W1fun, Nq) - Pext

    aL, aU = bracket_root_scan(g, A)

    for _ in range(maxIter):
        aM = 0.5 * (aL + aU)
        gL = g(aL)
        gM = g(aM)
        if abs(gM) < tol_f or (aU - aL) < tol_a:
            return float(aM)
        if gL * gM > 0:
            aL = aM
        else:
            aU = aM

    raise RuntimeError("Marlow: Bisection did not converge.")


def run_comparison(
    A: float,
    B: float,
    Pext_list: np.ndarray,
    Nr: int,
    Nq: int,
    tol_f: float,
    tol_a: float,
    maxIter: int,
    C1_MR: float,
    C2_MR: float,
    mu_OG: np.ndarray,
    alp_OG: np.ndarray,
    lambda_uni: np.ndarray,
    Pnom_uni: np.ndarray,
) -> dict[str, np.ndarray | Callable[[np.ndarray], np.ndarray]]:
    """Solve a(P) for all models."""
    mu_MR = 2 * (C1_MR + C2_MR)
    W1fun = build_W1_from_uniaxial_CLAMPED(lambda_uni, Pnom_uni)

    a_MR = np.full(Pext_list.shape, np.nan, dtype=float)
    a_OG = np.full(Pext_list.shape, np.nan, dtype=float)
    a_MA = np.full(Pext_list.shape, np.nan, dtype=float)

    for i, Pext in enumerate(Pext_list):
        a_MR[i] = solve_a_MR(Pext, A, B, mu_MR, tol_f, tol_a, maxIter)
        a_OG[i] = solve_a_ogden(Pext, A, B, mu_OG, alp_OG, Nq, tol_f, tol_a, maxIter)
        a_MA[i] = solve_a_marlow(Pext, A, B, W1fun, Nq, tol_f, tol_a, maxIter)

    return {
        "mu_MR": np.array(mu_MR),
        "W1fun": W1fun,
        "a_MR": a_MR,
        "a_OG": a_OG,
        "a_MA": a_MA,
        "Nr": np.array(Nr),
    }


def save_comparison_table(
    output_dir: Path,
    Pext_list: np.ndarray,
    A: float,
    a_MR: np.ndarray,
    a_OG: np.ndarray,
    a_MA: np.ndarray,
) -> None:
    """Save the pressure sweep response for all models."""
    header = "Pext_Pa,a_MR_m,a_OG_m,a_MA_m,a_MR_over_A,a_OG_over_A,a_MA_over_A"
    table = np.column_stack((Pext_list, a_MR, a_OG, a_MA, a_MR / A, a_OG / A, a_MA / A))
    np.savetxt(
        output_dir / "COMPARISON_hyperelastic_thickwall_summary.csv",
        table,
        delimiter=",",
        header=header,
        comments="",
    )


def plot_marlow_input(
    output_dir: Path,
    lambda_uni: np.ndarray,
    Pnom_uni: np.ndarray,
) -> None:
    """Plot Marlow input nominal stress-stretch data."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(lambda_uni, Pnom_uni / 1e3, "o-", linewidth=1.5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$P_{nom}$ (kPa)")
    ax.set_title("Experimental uniaxial data used for Marlow")
    ax.grid(True)

    fig.savefig(output_dir / "COMPARISON_marlow_input.png", dpi=300)
    plt.close(fig)


def plot_global_response(
    output_dir: Path,
    Pext_list: np.ndarray,
    A: float,
    a_MR: np.ndarray,
    a_OG: np.ndarray,
    a_MA: np.ndarray,
) -> None:
    """Plot a/A versus Pext for all models."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(Pext_list / 1e3, a_MR / A, "o-", linewidth=1.8, markersize=5, label="Mooney-Rivlin")
    ax.plot(
        Pext_list / 1e3,
        a_OG / A,
        "s-",
        linewidth=1.8,
        markersize=5,
        label="Ogden N=3 (Abaqus form)",
    )
    ax.plot(
        Pext_list / 1e3,
        a_MA / A,
        "d-",
        linewidth=1.8,
        markersize=5,
        label="Marlow (exp, clamped)",
    )
    ax.set_xlabel(r"$P_{ext}$ (kPa)")
    ax.set_ylabel("a/A")
    ax.set_title("Deformed inner radius vs external pressure")
    ax.grid(True)
    ax.legend(loc="best")

    fig.savefig(output_dir / "COMPARISON_global_response.png", dpi=300)
    plt.close(fig)


def plot_stress_comparison(
    output_dir: Path,
    Pext_list: np.ndarray,
    A: float,
    B: float,
    Nr: int,
    Psel: float,
    a_MR: np.ndarray,
    a_OG: np.ndarray,
    a_MA: np.ndarray,
    mu_MR: float,
    mu_OG: np.ndarray,
    alp_OG: np.ndarray,
    W1fun: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Plot stress profiles at the selected pressure."""
    aMR = float(PchipInterpolator(Pext_list, a_MR)(Psel))
    aOG = float(PchipInterpolator(Pext_list, a_OG)(Psel))
    aMA = float(PchipInterpolator(Pext_list, a_MA)(Psel))

    bMR = np.sqrt(B**2 - A**2 + aMR**2)
    bOG = np.sqrt(B**2 - A**2 + aOG**2)
    bMA = np.sqrt(B**2 - A**2 + aMA**2)

    rMR = np.linspace(aMR, bMR, Nr)
    rOG = np.linspace(aOG, bOG, Nr)
    rMA = np.linspace(aMA, bMA, Nr)

    rhoMR = (rMR - aMR) / (bMR - aMR)
    rhoOG = (rOG - aOG) / (bOG - aOG)
    rhoMA = (rMA - aMA) / (bMA - aMA)

    sigR_MR, sigT_MR = stresses_MR(rMR, aMR, bMR, Psel, A, mu_MR)
    sigR_OG, sigT_OG = stresses_OGDEN(rOG, aOG, bOG, Psel, A, mu_OG, alp_OG)
    sigR_MA, sigT_MA = stresses_MARLOW(rMA, aMA, bMA, Psel, A, W1fun)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].plot(rhoMR, sigR_MR, linewidth=1.8, label="Mooney-Rivlin")
    axes[0].plot(rhoOG, sigR_OG, linewidth=1.8, label="Ogden N=3")
    axes[0].plot(rhoMA, sigR_MA, linewidth=1.8, label="Marlow")
    axes[0].set_xlabel("rho=(r-a)/(b-a)")
    axes[0].set_ylabel(r"$\sigma_r$ (Pa)")
    axes[0].set_title(rf"$\sigma_r(r)$ at P={Psel / 1e3:.2f} kPa")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].plot(rhoMR, sigT_MR, linewidth=1.8, label="Mooney-Rivlin")
    axes[1].plot(rhoOG, sigT_OG, linewidth=1.8, label="Ogden N=3")
    axes[1].plot(rhoMA, sigT_MA, linewidth=1.8, label="Marlow")
    axes[1].set_xlabel("rho=(r-a)/(b-a)")
    axes[1].set_ylabel(r"$\sigma_\theta$ (Pa)")
    axes[1].set_title(rf"$\sigma_\theta(r)$ at P={Psel / 1e3:.2f} kPa")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    fig.savefig(output_dir / "COMPARISON_stress_profiles_selected_pressure.png", dpi=300)
    plt.close(fig)

    print(f"Selected pressure: Psel = {Psel / 1e3:.3f} kPa")
    print("\nBoundary checks (target: sigma_r(a)=0, sigma_r(b)=-P):")
    print(f"MR: sigma_r(a)={sigR_MR[0]:.3e}, sigma_r(b)={sigR_MR[-1]:.3e} (target {-Psel:.3e})")
    print(f"OG: sigma_r(a)={sigR_OG[0]:.3e}, sigma_r(b)={sigR_OG[-1]:.3e} (target {-Psel:.3e})")
    print(f"MA: sigma_r(a)={sigR_MA[0]:.3e}, sigma_r(b)={sigR_MA[-1]:.3e} (target {-Psel:.3e})")


def main() -> None:
    """Run the default model comparison."""
    output_dir = Path(__file__).resolve().parent

    # -------------------- GEOMETRY -------------------------------------
    A = 5e-3
    B = 8.5e-3

    # -------------------- PRESSURE SWEEP -------------------------------
    Pext_list = np.linspace(0.1e3, 60e3, 100)
    Nr = 450
    Nq = 1200

    tol_f = 1e-9
    tol_a = 1e-12
    maxIter = 250

    # -------------------- MATERIAL PARAMETERS --------------------------
    C1_MR = 0.0316e6
    C2_MR = 0e6
    mu_MR = 2 * (C1_MR + C2_MR)

    mu_OG = np.array([-0.0974e6, 0.1827e6, -0.0104e6])
    alp_OG = np.array([1.729, 1.1714, -6.5838])

    lambda_uni = np.array(
        [
            1,
            1.019069767,
            1.039069767,
            1.066511628,
            1.090232558,
            1.114418605,
            1.130697674,
            1.142325581,
            1.150697674,
            1.157209302,
            1.16372093,
            1.170697674,
            1.173953488,
            1.177674419,
            1.182325581,
            1.184186047,
            1.188837209,
            1.193953488,
            1.19627907,
            1.199534884,
        ],
        dtype=float,
    )

    stress_kPa = np.array(
        [
            0,
            0.71942446,
            0.863309353,
            2.158273381,
            4.460431655,
            9.352517986,
            14.24460432,
            19.28057554,
            24.89208633,
            30.79136691,
            36.25899281,
            41.43884892,
            47.3381295,
            52.08633094,
            57.55395683,
            62.58992806,
            67.76978417,
            73.38129496,
            79.28057554,
            86.47482,
        ],
        dtype=float,
    )

    Pnom_uni = 1e3 * stress_kPa
    idx = np.argsort(lambda_uni)
    lambda_uni = lambda_uni[idx]
    Pnom_uni = Pnom_uni[idx]

    results = run_comparison(
        A,
        B,
        Pext_list,
        Nr,
        Nq,
        tol_f,
        tol_a,
        maxIter,
        C1_MR,
        C2_MR,
        mu_OG,
        alp_OG,
        lambda_uni,
        Pnom_uni,
    )

    a_MR = results["a_MR"]
    a_OG = results["a_OG"]
    a_MA = results["a_MA"]
    W1fun = results["W1fun"]

    plot_marlow_input(output_dir, lambda_uni, Pnom_uni)
    plot_global_response(output_dir, Pext_list, A, a_MR, a_OG, a_MA)

    Psel = Pext_list[-1]
    plot_stress_comparison(
        output_dir,
        Pext_list,
        A,
        B,
        Nr,
        Psel,
        a_MR,
        a_OG,
        a_MA,
        mu_MR,
        mu_OG,
        alp_OG,
        W1fun,
    )
    save_comparison_table(output_dir, Pext_list, A, a_MR, a_OG, a_MA)

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
