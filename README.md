"""
Thick-walled tube under uniform external pressure.

Incompressible Mooney-Rivlin material, plane strain (lambda_z = 1).

For a set of applied external pressures Pext, this script solves for the
deformed inner radius a, computes the deformed outer radius b, evaluates
radial and hoop Cauchy stresses through the wall, and reproduces the plots
from the original MATLAB semi-analytical solution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pressure_from_a(a: float, A: float, B: float, mu: float) -> float:
    """
    Compute Pext predicted by the analytical solution for a given deformed
    inner radius a.

    P = mu * (F(b) - F(a))
    k = A^2 - a^2
    b^2 = B^2 - A^2 + a^2
    F(r) = ln(r) - 0.5 ln(r^2 + k) - k / (2 r^2)
    """
    if a <= 0 or a > A:
        return np.nan

    k = A**2 - a**2
    b2 = B**2 - A**2 + a**2
    b = np.sqrt(b2)

    def F(r: float | np.ndarray) -> float | np.ndarray:
        return np.log(r) - 0.5 * np.log(r**2 + k) - k / (2 * r**2)

    P = mu * (F(b) - F(a))
    return float(P)


def solve_a_for_pressure(
    Pext: float,
    A: float,
    B: float,
    mu: float,
    tol_f: float,
    tol_a: float,
    maxIter: int,
) -> tuple[float, float]:
    """
    Solve g(a) = 0 for a in (0, A] using bracketing and bisection.

    g(a) = mu * (F(b) - F(a)) - Pext
    k = A^2 - a^2
    b^2 = B^2 - A^2 + a^2
    F(r) = ln(r) - 0.5 ln(r^2 + k) - k / (2 r^2)
    """
    aU = A
    aL = 0.1 * A
    aMin = 1e-12 * A

    def g(aa: float) -> float:
        return pressure_from_a(aa, A, B, mu) - Pext

    gU = g(aU)
    if gU > 0:
        print("Warning: g(A) > 0. Check sign conventions or parameter ranges.")

    gL = g(aL)
    iterBracket = 0
    while (gL <= 0) and (aL > aMin) and (iterBracket < 80):
        aL = 0.5 * aL
        gL = g(aL)
        iterBracket += 1

    if gL <= 0:
        raise RuntimeError(
            "Failed to bracket root for a. Try smaller pressures, check "
            "parameters, or reduce aMin."
        )

    for _ in range(maxIter):
        aM = 0.5 * (aL + aU)
        gM = g(aM)

        if abs(gM) < tol_f or (aU - aL) < tol_a:
            a = aM
            b = np.sqrt(B**2 - A**2 + a**2)
            return float(a), float(b)

        if gL * gM > 0:
            aL = aM
            gL = gM
        else:
            aU = aM
            gU = gM

    raise RuntimeError(
        "Bisection did not converge within maxIter. Increase maxIter or "
        "loosen tolerances."
    )


def compute_stress_profile(
    Pext: float,
    A: float,
    B: float,
    mu: float,
    a: float,
    b: float,
    Nr: int,
) -> dict[str, np.ndarray]:
    """
    Build the current radial grid and compute sigma_r and sigma_theta.
    """
    r = np.linspace(a, b, Nr)
    k = A**2 - a**2

    Fr = np.log(r) - 0.5 * np.log(r**2 + k) - k / (2 * r**2)
    Fb = np.log(b) - 0.5 * np.log(b**2 + k) - k / (2 * b**2)

    sigma_r = -Pext + mu * (Fb - Fr)
    delta = -mu * k * (1 / (r**2 + k) + 1 / (r**2))
    sigma_theta = sigma_r + delta
    rho = (r - a) / (b - a)

    return {
        "r": r,
        "rho": rho,
        "sigma_r": sigma_r,
        "sigma_theta": sigma_theta,
    }


def run_analysis(
    A: float,
    B: float,
    C1: float,
    C2: float,
    Pext_list: np.ndarray,
    Nr: int,
    tol_f: float,
    tol_a: float,
    maxIter: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
    """
    Solve all pressures and return a_sol, b_sol, and stress_profiles.
    """
    mu = 2 * (C1 + C2)

    if not (B > A):
        raise ValueError("Require B > A.")
    if mu <= 0:
        raise ValueError("Require mu = 2*(C1+C2) > 0.")

    a_sol = np.full(Pext_list.shape, np.nan, dtype=float)
    b_sol = np.full(Pext_list.shape, np.nan, dtype=float)
    stress_profiles: list[dict[str, np.ndarray]] = []

    for ip, Pext in enumerate(Pext_list):
        a, b = solve_a_for_pressure(Pext, A, B, mu, tol_f, tol_a, maxIter)
        a_sol[ip] = a
        b_sol[ip] = b

        stress_profile = compute_stress_profile(Pext, A, B, mu, a, b, Nr)
        stress_profiles.append(stress_profile)

    return a_sol, b_sol, stress_profiles


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
    summary_table = np.column_stack(
        (Pext_list, a_sol, b_sol, a_sol / A, b_sol / B)
    )

    np.savetxt(
        output_dir / "MR_hyperelastic_thickwall_summary.csv",
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


def plot_stress_profiles(
    output_dir: Path,
    Pext_list: np.ndarray,
    stress_profiles: list[dict[str, np.ndarray]],
) -> None:
    """Create the sigma_r and sigma_theta through-thickness plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].set_xlabel(r"Normalized radius $\rho = (r-a)/(b-a)$")
    axes[0].set_ylabel(r"$\sigma_r$ (Pa)")
    axes[0].set_title(r"$\sigma_r(r)$ through thickness")
    axes[0].grid(True)

    axes[1].set_xlabel(r"Normalized radius $\rho = (r-a)/(b-a)$")
    axes[1].set_ylabel(r"$\sigma_\theta$ (Pa)")
    axes[1].set_title(r"$\sigma_\theta(r)$ through thickness")
    axes[1].grid(True)

    for Pext, stress_profile in zip(Pext_list, stress_profiles):
        label = f"P={Pext / 1e3:.1f} kPa"
        axes[0].plot(
            stress_profile["rho"],
            stress_profile["sigma_r"],
            linewidth=1.5,
            label=label,
        )
        axes[1].plot(
            stress_profile["rho"],
            stress_profile["sigma_theta"],
            linewidth=1.5,
            label=label,
        )

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    fig.savefig(output_dir / "MR_stress_profiles.png", dpi=300)
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
    ax.set_title("Deformed inner radius vs external pressure")
    ax.grid(True)

    fig.savefig(output_dir / "MR_global_response.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the default Mooney-Rivlin thick-wall analysis."""
    output_dir = Path(__file__).resolve().parent

    # -------------------- USER-DEFINED PARAMETERS -----------------------
    # Reference (undeformed) radii (meters)
    A = 5e-3
    B = 8.5e-3

    # Mooney-Rivlin parameters (Pa)
    C1 = 0.0316e6
    C2 = 0e6

    # Effective shear modulus used in derived closed-form stress relations
    mu = 2 * (C1 + C2)

    # External pressures to test (Pa)
    Pext_list = np.array([0.5e3, 1e3, 2e3, 5e3, 10e3, 15e3, 20e3])

    # Number of points through thickness for stress profiles
    Nr = 300

    # Solver tolerances for the scalar equation g(a)=0
    tol_f = 1e-10
    tol_a = 1e-10
    maxIter = 200

    a_sol, b_sol, stress_profiles = run_analysis(
        A, B, C1, C2, Pext_list, Nr, tol_f, tol_a, maxIter
    )

    save_summary_table(output_dir, Pext_list, A, B, a_sol, b_sol)
    plot_stress_profiles(output_dir, Pext_list, stress_profiles)
    plot_global_response(output_dir, Pext_list, A, a_sol)

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
