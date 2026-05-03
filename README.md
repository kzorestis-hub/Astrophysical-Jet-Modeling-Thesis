# Master's Thesis: Modeling Relativistic Astrophysical Jets

This repository contains the core numerical modeling and visualization suite developed for my Master's Thesis at the University of Barcelona, titled **"Astrophysical Jet Modelling."**

The project focuses on the physical processes within relativistic jets, specifically calculating the luminosity evolution and spectral characteristics of synchrotron emission from relativistic electrons as they propagate away from a central engine.

## 🛠 Project Overview
The code simulates the physical conditions of a jet by accounting for relativistic effects, particle distribution dynamics, and the transition from co-moving to laboratory frames of reference.

### Key Physics Features:
* **Relativistic Doppler Boosting:** Implements transformations between the jet's co-moving frame and the observer's laboratory frame using the Doppler factor $\delta = [\gamma(1 - \beta \cos \theta)]^{-1}$.
* **Synchrotron Emission Modeling:** Calculates the specific luminosity $L_{\nu}$ based on the magnetic field strength $B$, the relativistic particle density, and the volume of the emitting region.
* **Jet Energetics:** Evaluates the total jet power and monitors the "1% Threshold" for luminosity to determine the efficiency of energy conversion at different distances from the source.
* **Geometric Evolution:** Models the expansion of the jet radius $R$ as a function of the distance $z$ using specified opening angles.

### Technical Implementation:
* **Log-Log Slope Analysis:** Features a robust numerical slope calculator to determine the power-law index of luminosity decay in log-log space.
* **Vectorized Numerical Integration:** Uses NumPy to handle complex physics equations over large spatial grids ($z \in [1, 300] Z_0$) with high efficiency.
* **Publication-Quality Visualization:** Generates standardized comparative plots for various physical scenarios (e.g., "Standard," "High Density," "Low Magnetic Field") to analyze how input parameters affect the resulting emission.

## 🚀 How to Use
The main script `final_thesis_plot_generator.py` contains the simulation engine. It is designed to be modular, allowing for the comparison of multiple physical "cases" by simply adjusting the input dictionaries for magnetic fields, particle densities, and Lorentz factors.
