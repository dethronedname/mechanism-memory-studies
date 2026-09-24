# Mechanism Memory Studies

**Controlled studies of calibration, persistent context, compositional dynamics, and training-budget trade-offs.**

[中文](README_CN.md) · [Documentation](docs/README.md) · [Study map](docs/study_map.md) · [Methods](docs/methods.md) · [Results](results/derived/TABLES.md) · [Conclusions and limits](docs/conclusions.md) · [Reproduction](docs/reproduction.md)

This research collection asks how a model can learn a component's response law from interface observations, retain that information while its dynamic state changes, and reuse it after waiting or recomposition. It contains positive, negative, and mixed evidence—not a claim of a generally superior architecture.

The principal candidate stores **two fixed context coordinates and two evolving state coordinates per component**, with structured dynamics and behavioral supervision. Comparators include a free four-dimensional state, persistently conditioned GRUs, a graph GRU, and a conditioned neural ODE. The four coordinates are learned latent variables, not identified stiffness, damping, position, and velocity.

## What the current evidence says

- **PI-4 / C1:** fixed context combined with behavioral supervision improved post-wait mechanism-difference prediction relative to the implemented free-four-state comparator. C1 repeated that direction across three independently generated datasets, but only one dataset passed the complete general-prediction protection criteria.
- **B1:** models that continuously read calibration information recovered some of the capability. At equal update counts the outcome remained a trade-off.
- **T1:** under the same 600-second complete-fit upper bound, the wider conditioned GRU achieved lower error in every recorded paired mechanism and short-horizon comparison. The candidate retained lower long-horizon error in 7/9 comparisons with that GRU and 9/9 with the graph GRU, while storing fewer persistent scalars. These are descriptive paired counts, not independent-task significance tests.

Thus, **a general quality–cost advantage has not been established**. The evidence supports studying a narrower relationship between state organization, supervision, and long-horizon behavior; it does not establish a universal failure or a proven deployment advantage. See the [bounded conclusion](docs/conclusions.md).

## Three ways to use the repository

1. **Read and recompute the evidence:** use the small, immutable JSON records and the standard-library table script below. No GPU or model weights are needed.
2. **Inspect or replay frozen models:** import the corresponding versioned source and evidence assets using the release inventory. Replay is distinct from retraining, and numerical task fidelity is distinct from the historical elementwise check.
3. **Run a new replication:** use the exact stage protocol in a fresh output directory, with separately authorized GPU resources. Published test cases are historical evaluation material, not a new blind test set.

## Quick start: result records, no training

```bash
python scripts/rebuild_results.py
python -m unittest discover -s tests -v
```

The first command verifies the included evidence hashes, recomputes all prescribed comparisons, checks the recorded decisions, and checks that the committed derived tables are current. To deliberately regenerate only the tables:

```bash
python scripts/rebuild_results.py --write
```

No command above imports PyTorch, loads a checkpoint, downloads data, or schedules training. These checks do not constitute a fresh neural-model or GPU reproduction. [Scope and levels](docs/reproduction.md).

## Repository map

| Location | Purpose |
|---|---|
| `docs/` | Research documentation and [reading index](docs/README.md) |
| `docs/project/` | Authorship, licensing, citation, changelog, and project policies |
| `docs/reports/` | Repository content and verification reports |
| `results/source_records/` | Small original or explicitly field-filtered JSON evidence, with source-archive and member hashes |
| `results/derived/` | Deterministically rebuilt tables, all per-seed scores, directional counts, and fit costs |
| `catalog/` | Study scope, asset inventory, provenance, and file manifests |
| `reference/` | Versioned frozen implementations, assembled from local source packages; do not overwrite one stage with another |
| `scripts/`, `tests/` | Lightweight evidence checks and regeneration |
| `docs/historical_context.md` | Earlier Wick / LocalPolyNet / role-geometry studies, separate from PI-family empirical claims |

Frozen source snapshots and full run assets must be marked **available**, **external**, or **missing** in the asset inventory. A historical trial count is not a statement that its weights are included in the default clone.

## Scope

The main experiments use a controlled nonlinear oscillator simulator, known interconnection rules, noiseless interface observations, and a limited parameter range. They do not establish superiority over Transformer/Mamba families, production graph simulators, or external real systems. Equal state size, equal parameter count, equal updates, and equal elapsed budget are different comparisons. [Details](docs/methods.md).

## Status, attribution, and reuse

The original protocols are closed historical studies; the repository remains open to independently specified replications and new questions. No automatic extension of a closed experiment is planned.

Research and curation used AI assistance. Numerical replay described in historical records is not independent human peer review. See [AI usage](docs/project/ai_usage.md), [contributing](.github/CONTRIBUTING.md), and the [license/release checklist](docs/project/release_checklist.md). The sole author is [dethronedname](https://github.com/dethronedname), using a public pseudonym. Original code uses [MIT](LICENSE); original documentation and data use [CC BY 4.0](LICENSES/CC-BY-4.0.txt). See [license scope](docs/project/licensing.md) and [authorship](docs/project/authors.md). No DOI or release identifier is asserted before one exists.

If you use the software, protocols, data, or results in research, please [cite this project](docs/project/citation.md) and identify the study and revision used. [CITATION.cff](CITATION.cff) provides machine-readable metadata. This academic citation request does not add a condition to the standard licenses.

English is the main documentation language; the Chinese README and bilingual conclusions provide companion summaries. Historical Chinese protocol documents remain in their original language.

## Repository status

The repository is available at [dethronedname/mechanism-memory-studies](https://github.com/dethronedname/mechanism-memory-studies). It contains the four source snapshots and the small evidence collection; full weights and prediction arrays are external assets. [Citation metadata](CITATION.cff) identifies the repository without inventing a DOI or release version.

## Included implementation and verification

Four source snapshots are integrated. Run python scripts/verify_reference.py to check their identity; python scripts/check_reference_cpu.py runs bounded CPU checks. Twenty standard-library tests and 71 bounded CPU tests passed. The local assembly performed no scientific training or new GPU experiment. The [documentation index](docs/README.md) groups research, project information, and verification reports. See the [ready report](docs/reports/repository_readiness.md), [assets](catalog/assets.json), [source imports](catalog/source_imports.json), [redactions](catalog/redactions.json) and [license review](docs/project/license_review.md).
