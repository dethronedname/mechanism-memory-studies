# Mechanism Memory Studies: local repository readiness

[中文报告](REPO_READY_REPORT_CN.md)

Date: 2026-09-24. Status: **PRIVATE_REVIEW_REPOSITORY / AUTHOR_AND_LICENSE_METADATA_COMPLETE**. The sole author is [chumingyzx](https://github.com/chumingyzx), using a public pseudonym. Original code is licensed under MIT; original documentation and data under CC BY 4.0. See [scope](LICENSING.md), [citation](CITATION.md), and [validation](VALIDATION_REPORT.json).

## Included

- Four isolated source snapshots: PI4, C1, B1, and T1 Run02. There are 127 imported files, including 84 Python files; imported Python and shell source is byte-identical to the identified executed archive members.
- Models, simulators, objectives, configuration, launchers, evaluators, tests, and required frozen dependencies are included. The T1 snapshot preserves the disclosed CUDA initialization repair, exact patch, and source hashes.
- 128 small evidence files preserve 1,512 per-model/seed/scenario records, 34 comparison summaries, and 54 T1 fit-cost records. Of these files, 127 match original members byte for byte; the PI4 status receipt uses explicit field filtering.
- English is the main documentation language: README, methods, study map, results, reproduction, related work, provenance, limitations, and this report. The repository also includes a Chinese README, bilingual conclusions, and Chinese historical protocol documents. The historical protocols have not been represented as fully translated.
- The author and licensing pages and citation instructions are provided in English with Chinese summaries.

## Validation and scientific scope

The original assembly passed 20 standard-library tests, 71 bounded CPU tests (PI4: 9; C1: 11; B1: 39; T1: 12), all four original read-only verifiers, and deterministic checks of nine derived files. CPU checks used small synthetic inputs, hidden CUDA, separate processes, and a 90-second limit per stage.

This metadata revision reruns the standard-library evidence tests, source identity checks, public-tree checks, and CFF schema validation. The 71 ML-dependent CPU tests remain recorded as prior assembly validation; they are not claimed as newly rerun. Export checks are recorded beside the returned ZIP.

The local assembly performed no scientific training, new GPU experiment, or remote publication. Subsequent hosting preparation created a private repository for author review; no public release has been made. Historical neural replay was not rerun. Source algorithms, scientific evidence, decisions, failed-run records, and test locks remain unchanged.

## Provenance and availability

- [Asset inventory](ASSET_INVENTORY.json): available means included; external means verified locally but outside the clone; missing means the expected original was not found.
- [Source imports](SOURCE_IMPORT_MANIFEST.json): original archive/member hashes and the explicitly derived public compatibility manifests.
- [Redactions](REDACTION_LOG.json): documented transformations of prose, status fields, and environment summaries.
- [License record](LICENSE_REVIEW.md): the historical absence of original-package notices and the owner's current authorization.
- [Reproduction](docs/reproduction.md): record rebuilding, bounded CPU checks, optional historical replay, and new experiments are separate levels.

Of 21 expected archives, 13 matched exactly; the original T1 startup-failure archive was also found and inspected. No same-name hash conflict or required mainline source gap was found. Eight optional archives remain missing; their exact filenames are recorded in the inventory and Chinese report. PI3/RG review notes are not treated as source code. Optional PI1/PI2 archives remain external.

The first T1 run retains 54 startup errors, zero training updates, and its original INCONCLUSIVE outcome. Run02 retains MIXED_TIME_BUDGET_TRADEOFF. See the [incident and repair record](docs/incidents/t1_cuda_initialization.md).

## Remaining publication work

The repository now has a private remote at [chumingyzx/mechanism-memory-studies](https://github.com/chumingyzx/mechanism-memory-studies), with main as the target branch. Its URL is recorded in the citation and publication metadata. Public visibility remains subject to author review; release version/date and an optional DOI remain unset until they exist. Full weights and prediction archives are outside this draft; publishing them would require a separately prepared public asset. There is no unified, newly verified historical neural-replay CLI.

The scientific conclusion remains bounded: a general quality-cost advantage has not been established; the records retain long-horizon and smaller-state trade-offs and open questions.

## Delivery

MechanismMemory_Repo_Attributed_20260924.zip, with SHA-256 and export-validation sidecars, preserves the reviewed local snapshot before private-hosting metadata was added. Git includes that subsequent metadata update. It excludes Git internals, training weights, generated arrays, the private handoff layer, and private local-path inventories. The earlier draft ZIP is preserved.
