"""
Single-layer thick-walled tube under uniform external pressure.

Incompressible Marlow-type hyperelastic model W = W(I1), plane strain
(lambda_z = 1). The model uses uniaxial nominal stress-stretch data to
construct W1(I1) = dW/dI1, then solves for the deformed inner radius a(Pext)
and computes sigma_r(r) and sigma_theta(r).
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


def build_W1_from_uniaxial(
    lambda_uni: np.ndarray,
    Pnom_uni: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construct W1(I1) = dW/dI1 from uniaxial nominal stress data.

    For incompressible uniaxial tension:
        I1(lambda) = lambda^2 + 2/lambda
        Pnom(lambda) = 2*W1(I1)*(lambda - lambda^-2)

    Therefore:
        W1(I1(lambda)) = Pnom(lambda) / (2*(lambda - lambda^-2))
    """
    lambda_uni = np.asarray(lambda_uni, dtype=float).reshape(-1)
    Pnom_uni = np.asarray(Pnom_uni, dtype=float).reshape(-1)

    denom = 2 * (lambda_uni - lambda_uni**-2)
    mask = np.abs(denom) > 1e-12

    lam = lambda_uni[mask]
    Pn = Pnom_uni[mask]

    I1 = lam**2 + 2 / lam
    W1 = Pn / (2 * (lam - lam**-2))

    idx = np.argsort(I1)
    I1s = I1[idx]
    W1s = W1[idx]

    interpolator = PchipInterpolator(I1s, W1s, extrapolate=True)

    def W1fun(I1q: np.ndarray) -> np.ndarray:
        return interpolator(I1q)

    return W1fun


def delta_marlow(
    r: np.ndarray,
    A: float,
    a: float,
    W1fun: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Delta(r) = sigma_theta - sigma_r for incompressible Marlow W(I1).
    """
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
    """Predicted external pressure for a given deformed inner radius a."""
    if a <= 0 or a > A:
        return np.nan

    b = np.sqrt(B**2 - A**2 + a**2)
    r = np.linspace(a, b, Nq)

    Delta = delta_marlow(r, A, a, W1fun)
    integrand = Delta / r

    P = -trapz(integrand, r)
    return float(P)


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
    """Solve g(a) = Pmodel_marlow(a) - Pext = 0 by bisection."""

    def g(aa: float) -> float:
        return Pmodel_marlow(aa, A, B, W1fun, Nq) - Pext

    aU = A
    aL = 0.1 * A
    aMin = 1e-12 * A

    gU = g(aU)
    if gU > 0:
        print("Warning: g(A) > 0. Check sign conventions or parameter ranges.")

    gL = g(aL)
    itB = 0
    while (gL <= 0) and (aL > aMin) and (itB < 120):
        aL = 0.5 * aL
        gL = g(aL)
        itB += 1

    if gL <= 0:
        raise RuntimeError(
            "Failed to bracket root for a. Reduce Pext or check W1(I1) "
            "extrapolation."
        )

    for _ in range(maxIter):
        aM = 0.5 * (aL + aU)
        gM = g(aM)

        if abs(gM) < tol_f or (aU - aL) < tol_a:
            return float(aM)

        if gL * gM > 0:
            aL = aM
            gL = gM
        else:
            aU = aM
            gU = gM

    raise RuntimeError("Bisection did not converge. Increase maxIter or loosen tolerances.")


def run_analysis(
    A: float,
    B: float,
    Pext_list: np.ndarray,
    Nr: int,
    Nq: int,
    tol_f: float,
    tol_a: float,
    maxIter: int,
    lambda_uni: np.ndarray,
    Pnom_uni: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]], Callable[[np.ndarray], np.ndarray]]:
    """Solve all pressures and return a_sol, b_sol, stress_profiles, and W1fun."""
    if not (B > A):
        raise ValueError("Require B > A.")

    W1fun = build_W1_from_uniaxial(lambda_uni, Pnom_uni)

    a_sol = np.full(Pext_list.shape, np.nan, dtype=float)
    b_sol = np.full(Pext_list.shape, np.nan, dtype=float)
    stress_profiles: list[dict[str, np.ndarray]] = []

    for ip, Pext in enumerate(Pext_list):
        a = solve_a_marlow(Pext, A, B, W1fun, Nq, tol_f, tol_a, maxIter)
        b = np.sqrt(B**2 - A**2 + a**2)

        a_sol[ip] = a
        b_sol[ip] = b

        r = np.linspace(a, b, Nr)
        rho = (r - a) / (b - a)
        Delta_vals = delta_marlow(r, A, a, W1fun)
        sigma_r = cumtrapz(Delta_vals / r, r)
        sigma_theta = sigma_r + Delta_vals

        stress_profiles.append(
            {
                "r": r,
                "rho": rho,
                "sigma_r": sigma_r,
                "sigma_theta": sigma_theta,
            }
        )

    return a_sol, b_sol, stress_profiles, W1fun


def save_summary_table(
    output_dir: Path,
    Pext_list: np.ndarray,
    A: float,
    B: float,
    a_sol: np.ndarray,
    b_sol: np.ndarray,
) -> None:
    """Save and print the global response summary table."""
    header = "Pext_Pa,a_m,b_m,a_over_A,b_over_B"
    summary_table = np.column_stack((Pext_list, a_sol, b_sol, a_sol / A, b_sol / B))

    np.savetxt(
        output_dir / "MARLOW_hyperelastic_thickwall_summary.csv",
        summary_table,
        delimiter=",",
        header=header,
        comments="",
    )

    print(header.replace(",", "\t"))
    for row in summary_table:
        print(
            f"{row[0]:.6g}\t{row[1]:.9g}\t{row[2]:.9g}\t"
            f"{row[3]:.9g}\t{row[4]:.9g}"
        )


def plot_W1_construction(
    output_dir: Path,
    lambda_uni: np.ndarray,
    Pnom_uni: np.ndarray,
    W1fun: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Plot the constructed Marlow W1(I1) relation."""
    denom = 2 * (lambda_uni - lambda_uni**-2)
    mask = np.abs(denom) > 1e-12
    I1_uni = lambda_uni[mask] ** 2 + 2 / lambda_uni[mask]
    W1_uni = Pnom_uni[mask] / (2 * (lambda_uni[mask] - lambda_uni[mask] ** -2))
    I1_dense = np.linspace(np.min(I1_uni), np.max(I1_uni), 400)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(I1_uni, W1_uni, "o", label="Data-derived W1 points")
    ax.plot(I1_dense, W1fun(I1_dense), "-", linewidth=1.8, label="Interpolated W1(I1)")
    ax.set_xlabel("I1")
    ax.set_ylabel("W1(I1) = dW/dI1 (Pa)")
    ax.set_title("Constructed Marlow derivative from uniaxial data")
    ax.grid(True)
    ax.legend(loc="best")

    fig.savefig(output_dir / "MARLOW_W1_construction.png", dpi=300)
    plt.close(fig)


def plot_stress_profiles(
    output_dir: Path,
    Pext_list: np.ndarray,
    stress_profiles: list[dict[str, np.ndarray]],
) -> None:
    """Create the sigma_r and sigma_theta through-thickness plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].set_xlabel("rho=(r-a)/(b-a)")
    axes[0].set_ylabel(r"$\sigma_r$ (Pa)")
    axes[0].set_title(r"$\sigma_r(r)$")
    axes[0].grid(True)

    axes[1].set_xlabel("rho=(r-a)/(b-a)")
    axes[1].set_ylabel(r"$\sigma_\theta$ (Pa)")
    axes[1].set_title(r"$\sigma_\theta(r)$")
    axes[1].grid(True)

    for Pext, stress_profile in zip(Pext_list, stress_profiles):
        label = f"P={Pext / 1e3:.2f} kPa"
        axes[0].plot(stress_profile["rho"], stress_profile["sigma_r"], linewidth=1.6, label=label)
        axes[1].plot(
            stress_profile["rho"],
            stress_profile["sigma_theta"],
            linewidth=1.6,
            label=label,
        )

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    fig.savefig(output_dir / "MARLOW_stress_profiles.png", dpi=300)
    plt.close(fig)


def plot_global_response(
    output_dir: Path,
    Pext_list: np.ndarray,
    A: float,
    a_sol: np.ndarray,
) -> None:
    """Create the a/A versus Pext plot."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(Pext_list / 1e3, a_sol / A, "o-", linewidth=1.8, markersize=7)
    ax.set_xlabel(r"$P_{ext}$ (kPa)")
    ax.set_ylabel("a/A")
    ax.set_title("Deformed inner radius vs external pressure (Marlow W(I1))")
    ax.grid(True)

    fig.savefig(output_dir / "MARLOW_global_response.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the default Marlow thick-wall analysis."""
    output_dir = Path(__file__).resolve().parent

    # -------------------- USER INPUTS ----------------------------------
    A = 5e-3
    B = 10e-3

    Pext_list = np.array([0.5e3, 1e3, 2e3, 4e3, 5e3])
    Nr = 500
    Nq = 1000

    tol_f = 1e-9
    tol_a = 1e-12
    maxIter = 250

    # -------------------- UNIAXIAL DATA FOR MARLOW ---------------------
    # Replace these arrays with experimental uniaxial data.
    lambda_uni = np.linspace(1.0, 4.0, 60)
    Pnom_uni = 8e3 * (lambda_uni - lambda_uni**-2) + 1e3 * (lambda_uni**2 - 1)

    a_sol, b_sol, stress_profiles, W1fun = run_analysis(
        A, B, Pext_list, Nr, Nq, tol_f, tol_a, maxIter, lambda_uni, Pnom_uni
    )

    plot_W1_construction(output_dir, lambda_uni, Pnom_uni, W1fun)
    plot_stress_profiles(output_dir, Pext_list, stress_profiles)
    plot_global_response(output_dir, Pext_list, A, a_sol)
    save_summary_table(output_dir, Pext_list, A, B, a_sol, b_sol)

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
