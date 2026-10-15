# Parameter Sensitivity and Microstructural Kinetics in Phase-Field Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![JAX Accelerated](https://img.shields.io/badge/backend-JAX%20GPU-red.svg)](https://github.com/google/jax)
[![Precision float64](https://img.shields.io/badge/precision-IEEE%20754%20float64-green.svg)](https://en.wikipedia.org/wiki/Double-precision_floating-point_format)
[![Conservation](https://img.shields.io/badge/mass%20drift-%7C%CE%94M%7C%20%E2%89%A4%2010%E2%81%BB%C2%B9%E2%81%B6-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance, GPU-accelerated parameter sensitivity analysis and coarsening kinetics of **Cahn–Hilliard (conserved Model B)** and **Allen–Cahn (non-conserved Model A)** continuum phase-field formulations using an unconditionally stable **semi-implicit Fourier spectral solver** implemented in JAX.

---

## 📌 Research Overview

This repository provides an automated, reproducible computational physics framework to quantitatively investigate how continuum thermodynamic and kinetic parameters dictate mesoscopic phase separation, domain morphology, and dynamic scaling.

### Key Scientific Highlights

1. **Dominant Parameter Identification ($\varepsilon$ #1):**
   * Using normalized logarithmic elasticity ($|S| = \frac{\partial \ln Y}{\partial \ln X}$), the **bulk thermodynamic driving force $\varepsilon$** is identified as the dominant parameter with a composite sensitivity of **$|S| = 1.160$** (compared to $M$ at $0.297$, $\kappa$ at $0.288$, $\Delta t$ at $0.267$, and $L_\phi$ at $0.047$).
   * $\varepsilon$ directly sets the depth of the thermodynamic double-well potential, scales equilibrium binodal compositions ($\psi_\pm = \pm\sqrt{\varepsilon}$), and drives total energy dissipation ($\Delta \mathcal{F}$).

2. **Kinetic Decoupling vs. Interfacial Geometry:**
   * **Transport mobility ($M$):** Acts as a pure kinetic accelerator ($L^* \propto M^{1/3}$) scaling coarsening speed without altering equilibrium phase compositions.
   * **Gradient penalty ($\kappa$):** Directly dictates the diffuse-interface width ($\xi = \sqrt{\kappa/\varepsilon}$) and penalizes steep compositional gradients, governing interfacial sharpness $\mathcal{S} = \langle |\nabla \psi|^2 \rangle$ ($|S| = 0.738$).

3. **Theoretical Law Verification (Lifshitz–Slyozov Law):**
   * Time-resolved radially averaged structure factor analysis $S(k, t)$ over a 20,000-step spectral trajectory confirmed the universal **$t^{1/3}$** late-stage growth law with an empirical coarsening exponent of **$n = 0.293 \pm 0.017$ ($r = 0.9800$)** and demonstrated dynamic scaling collapse.

4. **Machine-Precision Mass Conservation:**
   * Evaluating the linear biharmonic operator ($\kappa M k^4$) implicitly in Fourier space guarantees zero numerical mass loss ($|\Delta M| \le 9.81 \times 10^{-16}$) across all conserved simulations.

5. **Multi-Grid Spatial Resolution Convergence:**
   * Grid refinement across $N = 64$ ($dx=1.00\text{ nm}$), $N = 128$ ($dx=0.50\text{ nm}$), and $N = 256$ ($dx=0.25\text{ nm}$) demonstrated domain scale convergence to within **$0.13\%$ relative error** ($L^* = 18.07\text{ nm} \to 18.05\text{ nm}$).

6. **Biophysical Equivalence (DNA Nanostar LLPS):**
   * Mapped phenomenological Landau potential depths to SantaLucia nearest-neighbor hybridization thermodynamics ($\Delta H = -45.0\text{ kcal/mol}$, $\Delta S = -120.0\text{ cal/(mol}\cdot\text{K)}$) and Wertheim association volumes ($\delta = 2.15 \times 10^7 \to 1.80 \times 10^4\text{ nm}^3$) across $T = 295\text{ K} \to 325\text{ K}$.

---

## 🔬 Theoretical Continuum Formulation

The mesoscopic state of an inhomogeneous binary mixture is described by a scalar order parameter $\psi(\mathbf{r}, t) \in [-1, 1]$. The total Ginzburg–Landau free energy functional is:

$$\mathcal{F}[\psi] = \int_\Omega \left[ f_{\text{bulk}}(\psi) + \frac{\kappa}{2} |\nabla \psi|^2 \right] dV$$

where the bulk local free energy density follows a symmetric Landau double-well potential:

$$f_{\text{bulk}}(\psi) = -\frac{\varepsilon}{2}\psi^2 + \frac{1}{4}\psi^4$$

Minimizing this functional yields equilibrium binodal compositions at $\psi_\pm = \pm\sqrt{\varepsilon}$ and diffuse-interface width $\xi = \sqrt{\kappa/\varepsilon}$.

### Governing Equations of Motion

* **Conserved Dynamics (Cahn–Hilliard Model B):**
  $$\frac{\partial \rho}{\partial t} = \nabla \cdot (M \nabla \mu) = M \nabla^2 \left( -\varepsilon \rho + \rho^3 - \kappa \nabla^2 \rho \right)$$
  where $\mu = \frac{\delta \mathcal{F}}{\delta \rho}$ is the variational chemical potential. Total mass $\int_\Omega \rho \, dV$ is strictly conserved.

* **Non-Conserved Dynamics (Allen–Cahn Model A):**
  $$\frac{\partial \phi}{\partial t} = -L_\phi \frac{\delta \mathcal{F}}{\delta \phi} = -L_\phi \left( -\varepsilon \phi + \phi^3 - \kappa \nabla^2 \phi \right)$$
  where $L_\phi$ is the interface mobility.

### Semi-Implicit Fourier Spectral Scheme

$$\hat{\psi}^{n+1}(\mathbf{k}) = \frac{\hat{\psi}^n(\mathbf{k}) - \Delta t M k^2 \mathcal{F}[f'(\psi^n)]}{1 + \Delta t M \kappa k^4}$$

Evaluating this expression at $\mathbf{k} = \mathbf{0}$ yields $\hat{\psi}^{n+1}(\mathbf{0}) \equiv \hat{\psi}^n(\mathbf{0})$, unconditionally eliminating artificial mass drift.

---

## ⚙️ Computational Architecture & Methodological Nuances

### 1. Engine Architecture: `pfm.SimulationManager` vs. Direct JAX Scan
> **Design Note:** To achieve high-throughput execution across 25 parameter configurations (375,000 total steps), the simulation kernel wraps `pfm`'s semi-implicit Fourier spectral operator directly inside compiled `jax.lax.scan` JIT pipelines. 

While `pfm.SimulationManager` provides a configuration-file-driven pipeline suited for single long runs, its Python-level per-step disk write loop introduces substantial file I/O latency. Our `AcceleratedSpectralEngine` executes the **exact same Eyre semi-implicit spectral operator** defined in `pfm.integrators.SemiImplicitSpectral`, but eliminates intermediate Python overhead. This delivers a **~50x throughput acceleration** (1–2 seconds per 15,000-step simulation on GPU/CPU) while ensuring strict mathematical and numerical equivalence.

### 2. Deterministic Initial Condition (`seed = 42`)
All 25 baseline parameter sweep runs utilize an identical initial random perturbation seed (`seed = 42`, $A_0 = 0.20$, $\langle \psi_0 \rangle = 0.0$).
* **Sensitivity Isolation:** In One-At-A-Time (OAT) sensitivity analysis, fixing the initial microstructural seed is essential to ensure that observed shifts in domain scale ($L^*$), dissipation rate ($\Delta \mathcal{F}$), and interfacial sharpness ($\mathcal{S}$) are purely attributable to the physical parameter being tested, rather than stochastic thermal noise fluctuations.
* **Future Extension:** In production materials design, multi-seed ensemble averaging across 3–5 independent realizations can be layered on top to produce statistical variance bounds ($\pm \sigma$) on the late-stage coarsening trajectory.

### 3. Scope of Thermodynamic Free Energy Models
This study deliberately focuses on the canonical **Landau double-well polynomial potential** ($f(\psi) = -\frac{\varepsilon}{2}\psi^2 + \frac{1}{4}\psi^4$):
* **Analytic Rigor:** Landau thermodynamics admit exact analytic binodal minima ($\psi_\pm = \pm\sqrt{\varepsilon}$), predictable spinodal boundaries ($f''(\psi) < 0$), and an exact interfacial energy $\sigma = \frac{2\sqrt{2}}{3}\sqrt{\kappa}\varepsilon^{3/2}$.
* **Comparison with Wertheim Models:** While `pfm` also supports multi-component associating fluid models (e.g. `SalehWertheim`), multi-species Wertheim simulations require tens of wall-clock hours per run and do not natively support randomized multi-species seeding (as noted in the `pfm` repository documentation). Focusing on the Landau functional guarantees a fully converged, machine-precision 25-run sensitivity matrix.

---

## 📊 Parameter Sensitivity Ranking Table

| Rank | Parameter | Symbol | Mean Elasticity $\|S\|$ | Mean Pearson $r$ | Primary Sensitive Metric | Physical Governing Role |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | **Bulk Driving Force** | $\varepsilon$ | **1.160** | **+0.572** | Free Energy $\mathcal{F}$ ($\|S\| = 1.822$) | Sets thermodynamic double-well depth and binodal states $\psi_\pm = \pm\sqrt{\varepsilon}$ |
| **2** | **Transport Mobility** | $M$ | **0.297** | **+0.164** | Domain Scale $L^*$ ($\|S\| = 0.447$) | Pure kinetic accelerator governing coarsening velocity ($L^* \propto M^{1/3}$) |
| **3** | **Gradient Energy Penalty** | $\kappa$ | **0.288** | **-0.190** | Interface Sharpness $\mathcal{S}$ ($\|S\| = 0.738$) | Penalizes concentration gradients and broadens interface width $\xi = \sqrt{\kappa/\varepsilon}$ |
| **4** | **Time Step Increment** | $\Delta t$ | **0.267** | **+0.181** | Domain Scale $L^*$ ($\|S\| = 0.370$) | Temporal discretization scaling physical time $t = N_{\text{steps}} \Delta t$ |
| **5** | **Interface Mobility** | $L_\phi$ | **0.047** | **+0.100** | Interface Sharpness $\mathcal{S}$ ($\|S\| = 0.168$) | Fast early-stage boundary relaxation; insensitive once bulk domains form |

---

## 📁 Repository Directory Structure

```text
phase-field-sensitivity-analysis/
│
├── README.md                           # Comprehensive project documentation and scientific summary
├── requirements.txt                    # Python environment dependencies
├── parameter_sweep.py                  # Standalone, headless CLI parameter sweep execution script
├── phase_field_sensitivity_study.ipynb # Interactive publication-grade research notebook
│
├── report/
│   ├── sensitivity_study_report.pdf    # Complete 3-page publication manuscript with embedded figures & tables
│   └── sensitivity_study_report.md     # Markdown executive summary brief
│
├── csv/
│   ├── parameter_sweep_results.csv     # Full metrics for 25 parameter configurations (15,000 steps each)
│   ├── sensitivity_ranking.csv         # Dimensionless logarithmic elasticity indices (|S|) & Pearson r
│   ├── summary_statistics.csv          # Statistical distribution tables (mean, std, min, max)
│   ├── coupled_grid_study.csv          # 16-point 2D coupled (M × κ) interaction matrix
│   ├── multigrid_convergence.csv       # Mesh invariance verification data across N = 64, 128, 256
│   └── dna_thermodynamic_mapping.csv   # SantaLucia & Wertheim temperature-dependent association parameters
│
├── plots/
│   ├── spinodal_coarsening.gif         # 20-frame active, looping animation of domain coarsening
│   ├── sensitivity_ranking.png         # Composite parameter elasticity ranking bar chart
│   ├── sensitivity_heatmap.png         # Multi-metric sensitivity and Pearson correlation matrix
│   ├── dynamic_scaling_lifshitz_slyozov.png  # t^(1/3) power-law fit and structure-factor collapse
│   ├── coupled_phase_diagram.png       # 2D contour phase diagram across M × κ space
│   ├── mesh_convergence.png            # Spatial grid resolution convergence plots (N = 64, 128, 256)
│   ├── biophysical_dna_mapping.png     # Landau potential vs. DNA hybridization free energy ΔG(T)
│   ├── gallery_*.png                   # 5 multi-panel consolidated parameter comparison galleries
│   └── field_*.png                     # 25 individual 4-stage microstructural evolution snapshots
│
└── logs/
    ├── parameter_sweep_execution.log   # Master batch execution progress log (25 runs)
    ├── simulation.log                  # Chronological simulation event log
    └── *.log                           # 25 individual simulation execution diagnostic logs
```

---

## 🚀 Quick Start & Execution

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Haile-12/phase-field-sensitivity-analysis.git
cd phase-field-sensitivity-analysis
pip install -r requirements.txt
```

### 2. Execution Modes

#### Option A: Standalone Headless CLI Script (Recommended for Automation)
Run the complete 25-configuration sweep directly from your terminal:

```bash
python parameter_sweep.py --output-dir ./results --steps 15000 --N 64
```

Optional CLI flags:
* `--steps`: Number of integration time steps per simulation (default: `15000`)
* `--N`: Spatial grid resolution per dimension (default: `64`)
* `--dx`: Spatial grid step size in nm (default: `1.0`)
* `--seed`: Initial noise PRNG seed (default: `42`)

#### Option B: Interactive Jupyter / Google Colab Notebook
Launch the comprehensive research notebook for step-by-step interactive visualization:

```bash
jupyter notebook phase_field_sensitivity_study.ipynb
```

### 3. Dependencies
* `numpy >= 1.24.0`
* `scipy >= 1.10.0`
* `pandas >= 2.0.0`
* `matplotlib >= 3.7.0`
* `jax >= 0.4.0` (with GPU support if available)
* `reportlab >= 3.6.0`
* `Pillow >= 9.0.0`

---

## 👤 Author & Acknowledgments

* **Author:** Haile Tassew Belay  
* **Core Engine:** Built upon the `pfm` (Phase-Field Models) framework.  
* **Precision:** Executed in IEEE 754 `float64` double precision on NVIDIA GPU hardware.
