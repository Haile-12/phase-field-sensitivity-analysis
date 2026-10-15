"""
Parameter sensitivity analysis script for phase-field models.
Executes systematic sweeps over kinetic, gradient, and thermodynamic coefficients
for Cahn-Hilliard and Allen-Cahn continuum formulations using JAX.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

# Enforce 64-bit precision for energy conservation accuracy
jax.config.update("jax_enable_x64", True)

# Add local path for pfm imports if present
if "." not in sys.path:
    sys.path.insert(0, ".")


class SpectralEngine:
    """Semi-implicit Fourier spectral solver for phase-field equations."""

    def __init__(self, n: int = 64, dx: float = 1.0):
        self.n = n
        self.dx = float(dx)

        # Wavenumber coordinates
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = jnp.array(kx**2 + ky**2, dtype=jnp.float64)
        self.k4 = jnp.array(self.k2**2, dtype=jnp.float64)

    def initial_condition(self, mean: float = 0.0, noise_amp: float = 0.2, seed: int = 42) -> jnp.ndarray:
        key = jax.random.PRNGKey(seed)
        noise = jax.random.uniform(key, shape=(self.n, self.n), minval=-1.0, maxval=1.0, dtype=jnp.float64)
        return jnp.array(mean, dtype=jnp.float64) + noise_amp * noise

    def cahn_hilliard_step(self, m: float, kappa: float, epsilon: float, dt: float):
        denom = 1.0 + dt * m * kappa * self.k4
        k2 = self.k2

        @jax.jit
        def step(psi, _):
            df = -epsilon * psi + psi**3
            df_hat = jnp.fft.fftn(df)
            psi_hat = jnp.fft.fftn(psi)
            psi_hat_next = (psi_hat - dt * m * k2 * df_hat) / denom
            return jnp.real(jnp.fft.ifftn(psi_hat_next)), None

        return step

    def allen_cahn_step(self, l_phi: float, kappa: float, epsilon: float, dt: float):
        denom = 1.0 + dt * l_phi * kappa * self.k2

        @jax.jit
        def step(phi, _):
            df = -epsilon * phi + phi**3
            df_hat = jnp.fft.fftn(df)
            phi_hat = jnp.fft.fftn(phi)
            phi_hat_next = (phi_hat - dt * l_phi * df_hat) / denom
            return jnp.real(jnp.fft.ifftn(phi_hat_next)), None

        return step


def free_energy(psi: np.ndarray, epsilon: float, kappa: float, dx: float = 1.0) -> float:
    gy, gx = np.gradient(psi, dx)
    bulk = -0.5 * epsilon * (psi**2) + 0.25 * (psi**4)
    gradient = 0.5 * kappa * (gx**2 + gy**2)
    return float(np.mean(bulk + gradient))


def structure_factor_peak(psi: np.ndarray, dx: float = 1.0) -> Tuple[float, float]:
    n = psi.shape[0]
    centered = psi - np.mean(psi)
    fft_psi = np.fft.fftshift(np.fft.fft2(centered))
    psd = np.abs(fft_psi)**2 / (n * n)

    y, x = np.indices((n, n))
    r = np.hypot(x - n // 2, y - n // 2)
    dk = 2.0 * np.pi / (n * dx)
    k_radial = r * dk

    nbins = n // 2
    k_bins = np.linspace(0, float(k_radial.max()) / np.sqrt(2.0), nbins)
    bin_idx = np.digitize(k_radial.ravel(), k_bins)

    sums = np.bincount(bin_idx, weights=psd.ravel(), minlength=nbins + 1)
    counts = np.bincount(bin_idx, minlength=nbins + 1)
    s_k = np.zeros(nbins)
    for i in range(1, nbins):
        if counts[i] > 0:
            s_k[i] = sums[i] / counts[i]

    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    valid = k_centers > 0.05
    if np.any(valid):
        idx = int(np.argmax(s_k[1:][valid]))
        k_peak = float(k_centers[valid][idx])
        s_peak = float(s_k[1:][valid][idx])
        l_peak = float(2.0 * np.pi / k_peak) if k_peak > 0 else float(n * dx)
    else:
        l_peak, s_peak = float(n * dx), 1.0

    return l_peak, s_peak


SWEEPS = [
    {"key": "M", "name": "Mobility", "values": [0.1, 0.5, 1.0, 2.0, 5.0], "model": "ch"},
    {"key": "k", "name": "Gradient Penalty", "values": [0.2, 0.5, 1.0, 2.0, 4.0], "model": "ch"},
    {"key": "epsilon", "name": "Bulk Driving Force", "values": [0.3, 0.6, 0.9, 1.2, 1.5], "model": "ch"},
    {"key": "L_phi", "name": "Allen-Cahn Mobility", "values": [0.1, 0.5, 1.0, 2.0, 5.0], "model": "ac"},
    {"key": "dt", "name": "Time Step", "values": [0.005, 0.010, 0.025, 0.050, 0.075], "model": "ch"},
]

BASE_PARAMS = {
    "N": 64,
    "dx": 1.0,
    "steps": 15000,
    "dt": 0.025,
    "M": 1.0,
    "L_phi": 1.0,
    "kappa": 1.0,
    "epsilon": 0.9,
    "seed": 42,
    "noise_amp": 0.20,
}


def run_sweep(output_dir: str, steps: int = 15000, n: int = 64, dx: float = 1.0, seed: int = 42):
    log_dir = os.path.join(output_dir, "logs")
    csv_dir = os.path.join(output_dir, "csv")
    plot_dir = os.path.join(output_dir, "plots")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    raw_dir = os.path.join(output_dir, "sim_outputs")

    for path in [log_dir, csv_dir, plot_dir, ckpt_dir, raw_dir]:
        os.makedirs(path, exist_ok=True)

    master_log = os.path.join(log_dir, "parameter_sweep_execution.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(master_log, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Starting parameter sensitivity study (N=%d, dx=%.2f, steps=%d, seed=%d)", n, dx, steps, seed)
    engine = SpectralEngine(n=n, dx=dx)
    checkpoints = [int(steps * 0.1), int(steps * 0.3), steps]

    records = []
    total_runs = sum(len(s["values"]) for s in SWEEPS)
    run_idx = 0

    for sweep in SWEEPS:
        param_key = sweep["key"]
        model = sweep["model"]

        for val in sweep["values"]:
            run_idx += 1
            run_id = f"{param_key}__{str(val).replace('.', 'p')}"

            m = val if param_key == "M" else BASE_PARAMS["M"]
            kappa = val if param_key == "k" else BASE_PARAMS["kappa"]
            eps = val if param_key == "epsilon" else BASE_PARAMS["epsilon"]
            l_phi = val if param_key == "L_phi" else BASE_PARAMS["L_phi"]
            dt = val if param_key == "dt" else BASE_PARAMS["dt"]

            t0 = time.time()
            field = engine.initial_condition(noise_amp=BASE_PARAMS["noise_amp"], seed=seed)
            field_init = np.array(field)

            frames = {0: field_init}
            step_fn = engine.cahn_hilliard_step(m, kappa, eps, dt) if model == "ch" else engine.allen_cahn_step(l_phi, kappa, eps, dt)

            curr = field
            step_count = 0
            for cp in checkpoints:
                curr, _ = jax.lax.scan(step_fn, curr, None, length=cp - step_count)
                step_count = cp
                frames[cp] = np.array(curr)

            curr.block_until_ready()
            runtime = time.time() - t0
            field_final = np.array(curr)

            fe_init = free_energy(field_init, eps, kappa, dx)
            fe_final = free_energy(field_final, eps, kappa, dx)
            delta_fe = fe_init - fe_final
            l_peak, s_peak = structure_factor_peak(field_final, dx)
            sharpness = float(np.mean(np.gradient(field_final, dx)[0]**2 + np.gradient(field_final, dx)[1]**2))
            segregation = float(np.std(field_final))
            drift = float(np.abs(np.mean(field_final) - np.mean(field_init)))

            # Save state arrays
            sim_sub = os.path.join(raw_dir, run_id)
            os.makedirs(sim_sub, exist_ok=True)
            np.savetxt(os.path.join(sim_sub, "init_0.dat"), field_init, fmt="%.6e")
            np.savetxt(os.path.join(sim_sub, "last_0.dat"), field_final, fmt="%.6e")

            rec = {
                "run_id": run_id,
                "param_key": param_key,
                "param_display_name": sweep["name"],
                "param_value": float(val),
                "model": model,
                "steps": steps,
                "dt": dt,
                "init_free_energy": fe_init,
                "final_free_energy": fe_final,
                "energy_drop": delta_fe,
                "dominant_length_scale": l_peak,
                "structure_factor_peak": s_peak,
                "interface_sharpness": sharpness,
                "phase_separation_index": segregation,
                "mass_drift": drift,
                "mass_conserved": bool(drift < 1e-8),
                "wall_time_seconds": runtime,
                "status": "SUCCESS",
            }
            records.append(rec)

            # Metadata checkpoint
            with open(os.path.join(ckpt_dir, f"{run_id}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)

            # Run log
            with open(os.path.join(log_dir, f"{run_id}.log"), "w", encoding="utf-8") as f:
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {model.upper()} | Integrator: semi-implicit spectral\n")
                f.write(f"Grid: {n}x{n}, dx={dx}, dt={dt}, steps={steps}\n")
                f.write(f"Parameters: M={m}, kappa={kappa}, epsilon={eps}, L_phi={l_phi}\n")
                f.write(f"Free Energy: init={fe_init:.8e}, final={fe_final:.8e}, drop={delta_fe:.8e}\n")
                f.write(f"Microstructure: L_peak={l_peak:.4f} nm, S_peak={s_peak:.6e}, sharpness={sharpness:.6e}\n")
                f.write(f"Conservation: mass_drift={drift:.2e}, conserved={drift < 1e-8}\n")
                f.write(f"Runtime: {runtime:.3f} s ({(runtime / steps) * 1e3:.4f} ms/step)\n")

            # Plot output
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
            vmax = float(np.max(np.abs(field_final)))
            for ax, (sk, tit) in zip(axes, [(0, "t=0"), (checkpoints[0], f"t={int(checkpoints[0]*dt)}"), (checkpoints[1], f"t={int(checkpoints[1]*dt)}"), (checkpoints[2], f"t={int(checkpoints[2]*dt)}")]):
                im = ax.imshow(frames[sk], cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, origin="lower")
                ax.set_title(tit, fontsize=9)
                ax.axis("off")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
            fig.suptitle(f"{sweep['name']} = {val} | Delta_F = {delta_fe:.4e} | L* = {l_peak:.2f} nm", fontsize=10)
            plt.savefig(os.path.join(plot_dir, f"field_{run_id}.png"), bbox_inches="tight", dpi=150)
            plt.close(fig)

            logging.info("[%02d/%02d] %s (value=%.3f) -> Delta_F=%.4e, L*=%.2f nm, drift=%.1e (%.2fs)",
                         run_idx, total_runs, sweep["name"], val, delta_fe, l_peak, drift, runtime)

    df = pd.DataFrame(records)
    csv_file = os.path.join(csv_dir, "parameter_sweep_results.csv")
    df.to_csv(csv_file, index=False)
    logging.info("Sweep complete. Saved %d results to %s", len(df), csv_file)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase-Field Parameter Sensitivity Sweep")
    parser.add_argument("--output-dir", type=str, default="./phase_field_sensitivity_study-result")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_sweep(output_dir=args.output_dir, steps=args.steps, n=args.N, dx=args.dx, seed=args.seed)
