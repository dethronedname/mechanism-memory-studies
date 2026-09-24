# Task, architecture, metrics, and comparison units

## Task and observed information

Components are simulated nonlinear oscillators coupled through known lossless velocity interconnections, not ordinary spring-edge systems. The model receives calibration input/output records, local query history, known connectivity, and future external controls. It does not receive true stiffness/damping, hidden position, state derivatives, or future observations during autonomous prediction.

Calibration and query episodes share component parameters but have reset internal states. Different calibration views may have different initial conditions and excitations. Identical-parameter controls and zero-difference boundaries are part of the data definition, not excluded inconvenient examples.

The first suite uses a common component-law family and known interconnections, with noiseless interface observations. Broader application claims require other evidence.

## Candidate and baselines

A component holds a learned fixed context `c ∈ R²` and evolving state `z ∈ R²`. The history encoder has its own hidden computation; “four-dimensional” counts persistent prediction state, not the entire network or physical dimensions. Context is fixed within a forecast, not identical across components and not untrainable.

Local structured dynamics use a learned energy, skew exchange, nonnegative dissipation, and an input/output interface. The continuous-time energy accounting condition does not guarantee arbitrary-step discrete stability, accurate dynamics, or semantic identification of the latent coordinates.

PI-4 compares fixed versus free state and ordinary MSE versus behavioral supervision. B1/T1 also include:

| Code ID | Role | Persistent coordinates/component | Parameters in the recorded B1/T1 configuration |
|---|---|---:|---:|
| `fixed_behavior` | fixed context + structured dynamics + behavioral target | 4 (2+2) | 1510 |
| `free_behavior` | free four-state structured comparator, same supervision | 4 | 1892 |
| `gru4_behavior` | conditioned GRU, small | 4 (2+2) | 1198 |
| `gru32_behavior` | conditioned GRU, wider | 32 (16+16) | 4082 |
| `graph32_behavior` | conditioned graph GRU, wider | 32 (16+16) | 8786 |
| `node32_behavior` | conditioned neural ODE, wider | 32 (16+16) | 3970 |

All new baselines can continually access calibration context. Wider baselines deliberately relax the four-scalar restriction; this is a resource–quality challenge, not equal capacity. Formula-level task adaptations are not complete replications of all public methods in these families. No Transformer/Mamba result is present.

## Learning objective

All relevant comparators receive the same calibration and behavioral data. The behavioral objective changes the weighting of ordinary prediction, inter-component response differences, and same-component calibration consistency. It does not create new mechanism information. The exact frozen coefficients, normalization, batch sizes, and validation selection live in each source `protocol.json` or `config.json`, not in a new unified “default” that overwrites history.

## Metrics

For each dataset, general prediction NMSE uses the documented training-output second-moment scale. The behavioral reset normalization follows its own recorded training scale. Do not renormalize each model or test case to improve presentation.

For two paired components, let `ΔY` be their true response difference and `ΔYhat` their predicted difference. The mechanism metric is:

```text
E_delta = mean((ΔYhat - ΔY)^2) / mean(ΔY^2).
```

Zero is ideal; one can describe predicting no difference. If the true difference energy is zero, the normalized score is **undefined/null**, not a perfect zero. Preserve that boundary and report unnormalized diagnostics.

Same-component alternative calibration checks nuisance sensitivity; wrong-component calibration checks reliance on calibration. Neither alone proves accurate physical parameter recovery. More sensitivity is not necessarily better.

## Prescribed comparisons and cost

For each same-seed candidate/comparator pair, calculate the error ratio. Mechanism ratios are geometrically averaged across the two 256-step-wait tasks and the three initializations. General-task ratios are medians of paired same-seed ratios, not ratios of separately computed medians. The original quality gate uses mechanism ≤0.90, all three general-task medians ≤1.05, and at least two favorable initialization summaries. These are engineering decision rules, not significance or equivalence tests.

C1 changes data draws while freezing the recipe. B1 uses equal 1600 update counts and a separate limited baseline learning-rate screen. T1 retains those locked recipes, removes the update stop, and uses a 600-second complete-process limit including imports, data transfer, initialization, training, validation, checkpointing, and exit. Thirty seconds are reserved internally for closing. Twenty wall-scheduled validation opportunities were recorded per T1 fit. Early 150/300-second checkpoints are **validation-only**, not selectable retrospective test winners.

Historical search, shared simulation, replay, and packaging costs must not be silently included for one family and omitted for another. Equal complete-fit cap does not mean equal total historical research cost or equal realized seconds. There is no reported inference-latency win.

## Statistical and causal limits

Keep dataset, initialization, method, scenario, wait length, and source-row identity in exported data. Do not pool all trials as independent samples. Do not infer a causal effect of extra training by subtracting B1 and T1 numbers: other frozen conditions and draws differ between stages. Interpret comparisons within each stage first.

Sources: [original configs and records](../results/source_records/), [T1 costs](../results/derived/t1_fit_costs.csv), and [source provenance](../results/evidence_manifest.json).
