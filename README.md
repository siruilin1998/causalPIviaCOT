
# Causal Partial Identification via Conditional Optimal Transport

**Sirui Lin**¹, **Zijun Gao**², **Jose Blanchet**¹, **Peter Glynn**¹  

¹ Stanford University, Department of Management Science & Engineering  
² Marshall School of Business, University of Southern California  

---

**Paper Links:**  
- 📄 arXiv (2025): https://arxiv.org/abs/2506.00257  
- 🏛 AISTATS 2026 (OpenReview): https://openreview.net/forum?id=rDufBj64yQ 
---


## 📖 Overview

This repository contains code, experiments, and supplementary materials for our work on **conditional optimal transport (COT)** methods for **causal partial identification**.

In many causal inference settings, only the marginal distributions of treatment and control outcomes are observed. The joint counterfactual distribution is not point-identified, leading to a **partially identified (PI) set** for causal estimands.

We propose a **Conditional Optimal Transport (COT)** framework to:

- Characterize causal partial identification sets via conditional OT problems.
- Incorporate covariate information to obtain tighter bounds.
- Avoid instability of naive plug-in OT estimators under weak topologies.
- Develop a direct nonparametric estimator with provable convergence guarantees.

---

## 🚀 Key Contributions

- Reformulation of causal bounds as conditional optimal transport values.
- Finite-sample convergence analysis for COT estimation.
- A discretization-based estimator that avoids nuisance parameter estimation.
- Empirical validation showing tighter bounds compared to baseline PI methods.

---


## 📚 Citation

Please cite our work if you use this code:

> Lin, S., Gao, Z., Blanchet, J., & Glynn, P. (2026).  
> *Causal Partial Identification via Conditional Optimal Transport*.  
> Proceedings of AISTATS 2026.

```bibtex
@inproceedings{lin2026causal,
  title={Causal Partial Identification via Conditional Optimal Transport},
  author={Lin, Sirui and Gao, Zijun and Blanchet, Jose and Glynn, Peter},
  booktitle={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics},
  year={2026}
}

