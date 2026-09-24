# Repository content and verification

[中文报告](repository_readiness_CN.md) · [Documentation](../README.md)

Repository: [dethronedname/mechanism-memory-studies](https://github.com/dethronedname/mechanism-memory-studies). Sole author: **dethronedname**, using a public pseudonym. Original code uses MIT; original documentation and data use CC BY 4.0. See [licensing](../project/licensing.md), [citation](../project/citation.md), and [validation](validation.json).

## Included content

- Four isolated source snapshots: PI4, C1, B1, and T1 Run02. They contain 127 imported files, including 84 Python files. Imported Python and shell source remains byte-identical to the identified executed archive members.
- Models, simulators, objectives, configurations, launchers, evaluators, and tests. T1 retains the disclosed CUDA initialization repair, exact patch, and original source hashes.
- 128 small evidence files, including 1,512 per-model/seed/scenario records, 34 comparison summaries, and 54 T1 fit-cost records. Of the evidence files, 127 match original members byte for byte; the PI4 status receipt uses explicit field filtering.
- English research and project documentation, a Chinese README, and bilingual conclusions. Historical Chinese protocols retain their original language.

The root holds the two READMEs, LICENSE, and CITATION.cff. Research documentation is in docs/; project information in docs/project/; verification reports in docs/reports/; machine-readable inventories and manifests in catalog/. GitHub contribution guidance is in .github/.

## Verification

The original assembly passed 20 standard-library tests, 71 bounded CPU tests (PI4: 9; C1: 11; B1: 39; T1: 12), all four original read-only verifiers, and deterministic checks of nine derived files.

The publication update checks the reorganized manifest paths, all Markdown links, author and citation metadata, repository file hashes, and unchanged scientific content. Standard-library tests and source verification run locally and in [CI](../project/ci_scope.md). The 71 ML-dependent CPU tests remain recorded as prior assembly validation.

Repository organization and publication do not add training, GPU experiments, or neural replay. Algorithms, evidence values, decisions, failure records, and test locks remain unchanged.

## Provenance and availability

- [Assets](../../catalog/assets.json): available means included; external means verified outside the clone; missing means the expected original was not found.
- [Source imports](../../catalog/source_imports.json): archive/member identities, original delivery hashes, and the explicitly derived public compatibility manifests.
- [Redactions](../../catalog/redactions.json): prose transformations, status-field filtering, and environment summaries.
- [Curation changes](../../catalog/curation_changes.json): file organization and differences from the supplied overlay.
- [File manifest](../../catalog/file_manifest.json): hashes of the repository files.
- [License record](../project/license_review.md): the original-package scan and the owner's authorization.

Of 21 expected archives, 13 matched exactly; the original T1 startup-failure archive was also inspected. No same-name hash conflict or required mainline source gap was found. Eight optional archives remain missing, with exact filenames in the asset inventory. Related review notes are not equivalent source code.

The first T1 run retains 54 startup errors, zero training updates, and INCONCLUSIVE. Run02 retains MIXED_TIME_BUDGET_TRADEOFF. See the [incident and repair record](../incidents/t1_cuda_initialization.md).

## Scope of publication

The repository provides code and small evidence. Full weights, prediction arrays, and raw run archives remain external; their public download URLs are unset. There is no newly verified unified historical neural-replay CLI. No tagged software release, DOI, or peer-reviewed publication is asserted.

The scientific conclusion remains bounded: a general quality-cost advantage has not been established; the records retain long-horizon and smaller-state trade-offs and open questions.
