# Related work and implementation scope

Primary bibliographic pages were checked on 2026-09-24. This is a focused context list drawn from the original study sources, not an exhaustive survey.

| Primary source | Relevance | Implementation boundary |
|---|---|---|
| Cho et al., [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), EMNLP 2014 | Gated recurrent models | B1/T1 use PyTorch GRU and GRUCell in a task-specific conditioned model; the translation system is not reproduced. |
| Sanchez-Gonzalez et al., [Learning to Simulate Complex Physics with Graph Networks](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html), ICML 2020, PMLR 119:8459–8468 | Message-passing physical simulators | The local graph GRU uses a single message round and a recurrent update. It does not reproduce GNS architecture, data or training. |
| Qing et al., [System-Aware Neural ODE Processes for Few-Shot Bayesian Optimization](https://arxiv.org/abs/2406.02352), arXiv:2406.02352v2 (2024) | Multiple trajectories used to infer system context | The local conditioned dynamics are deterministic; SANODEP's probabilistic objective and Bayesian optimization are not reproduced. |
| Jing et al., [Meta-learning Structure-Preserving Dynamics](https://arxiv.org/abs/2508.11205), arXiv:2508.11205v2 (2026 revision) | Latent modulation and Hamiltonian model families | Related structural context; no reproduction of the authors' implementation or performance. |
| Neary and Topcu, [Compositional Learning of Dynamical System Models Using Port-Hamiltonian Neural Networks](https://arxiv.org/abs/2212.00893), arXiv:2212.00893v2 (2023) | Compositional dynamics with physical interconnections | The candidate is a project-specific structured model. Port-based composition is prior work. |
| Zaheer et al., [Deep Sets](https://arxiv.org/abs/1703.06114), NeurIPS 2017 | Permutation-invariant representations of calibration episodes | Pooling episodes is established methodology, not a new architectural contribution of these studies. |

The imported baseline source describes its implementations as local adaptations and explicitly says that GNS source was not copied. Curation found no vendored upstream implementation in the imported files. The owner has confirmed sole ownership of the original project material and authorized the [repository licenses](project/licensing.md); this does not relicense the cited works or external dependencies. No weights, figures or code were downloaded from the cited works.

Numerical results must be cited to this repository's [evidence manifest](../results/evidence_manifest.json), not to the related papers. No comparison with an entire GRU, GNS, SANODEP, MSPD, Transformer or Mamba family follows from these local baselines.
