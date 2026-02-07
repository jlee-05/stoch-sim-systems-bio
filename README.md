# Stochastic Simulation of Reaction-Diffusion Systems

**Supplementary Material for Dissertation** | Durham University  
**Topic:** Stochastic Modelling for Systems Biology

---

## 📌 Overview
This repository contains the visual results (GIF animations) generated from stochastic simulations of the **Gray-Scott Reaction-Diffusion model**.

The primary goal of this research is to bridge the gap between microscopic molecular fluctuations (noise) and macroscopic spatial patterns. The simulations utilise the **Chemical Langevin Equation (CLE)** and its spatial extension (**RDLE**) to efficiently approximate the stochastic dynamics on a 2D grid, demonstrating how intrinsic noise drives pattern formation.

---

## 1. The Effect of Noise (Volume Scaling)
The "graininess" of the simulation is controlled by the system size parameter, $\Omega$ (Omega).

* **Low Population ($\Omega=50$):** High intrinsic noise. The RDLE approximation reveals how stochastic fluctuations dominate, leading to spontaneous nucleation and "ragged" pattern edges.
* **High Population ($\Omega=5000$):** Low noise. The system approaches the deterministic limit (ODE), resulting in smooth, continuous wavefronts.

| **High Noise ($\Omega=50$)** | **Deterministic Limit ($\Omega=5000$)** |
|:----------------------------:|:---------------------------------------:|
| ![Low Pop Result](Results/Labyrinth_Omega50.gif) | ![High Pop Result](Results/Labyrinth_Omega5000.gif) |
| *Note the spontaneous symmetry breaking caused by noise.* | *Smooth evolution typical of standard differential equations.* |

---

## 2. Pattern Versatility (Turing Instability)
By adjusting the **Feed ($F$)** and **Kill ($k$)** rates, the system shifts between different Turing regimes, replicating biological patterns found in nature (e.g., *Tetraodon mbu* pufferfish skin).

### 🐡 Phenotype A: Juvenile Spots
* **Parameters:** $F=0.030, k=0.062$
* **Behavior:** Isolated regions of high concentration stabilize into spots.

![Spots Animation](Results/Spots_Omega1000.gif)

### 🦓 Phenotype B: Adult Labyrinths (Stripes)
* **Parameters:** $F=0.037, k=0.060$
* **Behavior:** Spots elongate and merge to form complex, maze-like structures.

![Maze Animation](Results/Labyrinth_Omega1000.gif)

---

## 🛠️ Methodology
These results were generated using the **Euler-Maruyama method** for the Reaction-Diffusion Langevin Equation (RDLE). This approximation allows for the simulation of complex spatial patterns that would be computationally prohibitive using the exact Gillespie Algorithm (SSA).
