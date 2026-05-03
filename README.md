# Relativistic Astrophysical Jet Simulator

This repository contains a Python-based simulation suite developed for my Master's Thesis. The code models the kinematic evolution and radiative output (synchrotron emission) of relativistic astrophysical jets as they expand outward from a central source.

## 🛠️ Core Capabilities

The script is built around a central physics engine (`RelativisticJetCalculator`) that evaluates the jet's properties along its propagation axis ($z$). 

**Main Functions:**
* **Kinematic Modeling:** Calculates the evolution of the Lorentz factor ($\Gamma$), jet radius, magnetic field strength, and various pressures (thermal vs. magnetic) over distance.
* **Multi-Model Testing:** Automatically runs and compares different physical scenarios (Model A, B, and C) defined by varying external pressure gradients.
* **Synchrotron Emission:** Computes the radiative spectrum across a wide range of frequencies, adjusting for relativistic Doppler boosting at different observer viewing angles ($\theta = 10^\circ$ and $45^\circ$).
* **Automated Data Visualization:** Generates a comprehensive suite of plots, including acceleration profiles, energy evolution in log-log space, and time-lapse spectral flux curves.

## ✅ Built-in Sanity Checks

To ensure the physical validity of the numerical simulations, the code includes automated diagnostic checks at the end of the runtime:
* **Mass Conservation Check:** Verifies that the mass accretion rate ($\dot{M}$) remains strictly constant across all $z$-steps.
* **Energy Conservation Check:** Confirms that the sum of kinetic and thermal energy rates is conserved when magnetic fields are excluded.
* **Radiative Efficiency Limits:** Compares the calculated comoving synchrotron luminosity against the total absolute jet power, ensuring the radiation does not exceed physical energy limits (tracking a strict 1% efficiency threshold).

## 🚀 Usage

The simulation is entirely self-contained in `final_thesis_plot_generator.py`. 

**Dependencies:**
* `numpy` (for vectorized array operations and log-space generation)
* `matplotlib` (for generating the visual output suite)

Run the script directly to execute the engine across all models. The script will sequentially output the calculated physical parameters to the console, followed by the generation of the matplotlib figures.
