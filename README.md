
#  Heliogeophysical Adaptive Coupling
### A Complex Systems Framework for Solar–Terrestrial Organization  
**Author:** Pedro Guilherme Antico  
**Date:** November 2025  
**License:** MIT  

---

##  Overview

The **Heliogeophysical Adaptive Coupling (HAC)** framework provides a novel conceptual and computational model for understanding the **Sun–Earth interaction** as a *complex adaptive system*.  
It bridges **plasma physics, geophysics, and complex systems science**, introducing a method to quantify *self-organization, resilience, and feedback adaptivity* in heliogeophysical coupling.

This repository includes:
- Theoretical formulation (LaTeX article)
- Computational analysis framework (Python)
- Example synthetic dataset and results
- Visualization and comparison tools for multifractal and wavelet analyses

---

##  Scientific Motivation

The HAC Hypothesis proposes that the **Sun–Earth system** behaves as a **self-regulating dissipative structure**, capable of temporary reconfiguration to maintain dynamic equilibrium.  
By applying **multifractal analysis**, **wavelet coherence**, and **scale-dependent causality**, the framework quantifies emergent adaptive properties of the magnetosphere–ionosphere system.

Key theoretical insights include:
- **Magnetospheric pre-conditioning:** Bz polarity shifts preceding CMEs may indicate anticipatory energy regulation.
- **Cross-scale coherence:** Wavelet-based correlations between solar and geomagnetic indices reveal structured, non-random feedback.
- **Resonant coupling:** Schumann Resonances act as mediators linking solar forcing, geomagnetic dynamics, and biological rhythms.

---

##  Repository Structure

Heliogeophysical-Adaptive-Coupling/ │ ├── article/ │   ├── HAC_Theory.tex          # Complete LaTeX article │   ├── references.bib          # Bibliographic sources │   └── FCID_Appendix.pdf       # Supplemental theoretical notes │ ├── src/ │   ├── helio_analysis.py       # Core analysis class │   ├── example_analysis.py     # Example full pipeline (this file) │   └── utils/                  # Auxiliary data functions │ ├── output/ │   ├── Quiet_Geomagnetic_Conditions/ │   ├── Geomagnetic_Storm_Conditions/ │   └── comparative_analysis.png │ ├── LICENSE                     # MIT License with citation note ├── README.md                   # This document └── requirements.txt             # Dependencies

---

##  Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/PedroAntico/Heliogeophysical-Adaptive-Coupling.git
cd Heliogeophysical-Adaptive-Coupling
pip install -r requirements.txt

Dependencies

Python ≥ 3.9

NumPy

Pandas

Matplotlib

PyWavelets

scikit-learn

statsmodels



---

 Usage Example

Run the full example workflow (synthetic satellite data simulation):

python src/example_analysis.py

This will:

1. Generate synthetic signals for quiet and storm geomagnetic conditions


2. Perform multifractal and wavelet-based analysis


3. Create a comparative report with visualizations in output/



Expected output:

comparative_analysis.png

Individual reports per condition (including spectrum width, Hurst exponent, and adaptive complexity metrics)



---

 Key Metrics

Metric	Meaning	Interpretation

Δα (Spectrum Width)	Width of multifractal spectrum	Higher → greater complexity/adaptivity
Complexity Index	Derived from nonlinear entropy measures	Quantifies system adaptability
Hurst Exponent (H)	Long-term correlation	H > 0.5 → persistence, self-organization
Wavelet Coherence	Cross-scale coupling	Identifies causal relationships between solar and geomagnetic variables



---

 Cross-Planetary Extension

The HAC framework extends to planetary-scale systems, such as Jupiter’s magnetosphere (using NASA’s Juno mission data).
The comparison explores whether self-organizational signatures exist across different planetary dynamos, contributing to the idea of universal adaptive coupling in magnetized celestial bodies.


---

 Citation

If you use or reference this framework, please cite:

> Antico, P. G. (2025). Heliogeophysical Adaptive Coupling: A Complex Systems Framework for Solar–Terrestrial Organization. GitHub Repository.
Available at: https://github.com/PedroAntico/Heliogeophysical-Adaptive-Coupling



Optionally, you can also cite the forthcoming publication:

> Antico, P. G. (2025). Auto-organization and Adaptive Coupling in the Sun–Earth System: A Systemic Approach to Heliogeophysics. (submitted)




---

 License

This project is distributed under the MIT License.
You are free to use, modify, and distribute it, provided proper credit is given to:

Pedro Guilherme Antico (2025)


---

 Acknowledgements

This framework builds on interdisciplinary insights from:

Ilya Prigogine (dissipative structures theory)

James Lovelock (Gaia hypothesis)

Recent advances in nonlinear geophysics, multifractal analysis, and magnetobiology



---

 Project Vision

> “The magnetosphere does not merely endure solar storms — it learns from them.”
— Pedro G. Antico




---
