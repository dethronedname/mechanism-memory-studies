# Provenance and immutable evidence

`results/evidence_manifest.json` records the source archive filename and SHA256, exact member path, local evidence path and SHA256, byte length, and transformation for every included small source record. Most are byte-identical copies. The original PI-4 supervisor receipt contains personal paths and process identifiers; the public status extract keeps a stated allowlist, while the private handoff retains the original.

`results/derived/all_scores.csv` points back to each source JSON and `/records/<index>` location. Derived values use full-precision recorded numbers; rounding is presentation only. Rebuilding a table is not verification of unprovided weights or a new experimental run.

## Raw / public / editorial layers

- **Original evidence:** retain unchanged in the local evidence vault, including failure statuses, exact protocols, stage decisions, patches, data/source hashes, curves and original tests.
- **Public reproducibility assets:** versioned code and machine-readable results, safely publishable logs, manifests, selected small checkpoints, and deterministic data generation. Redactions must be described with old/new hashes, not performed in place.
- **Editorial documentation:** explain tested questions and bounded conclusions. Narrative may be improved without altering experimental facts or gate meanings.

A report title containing “independent review” describes a specified file or numerical checking procedure. Do not represent it as an independent laboratory experiment, a second independent neural implementation, or human peer review. CPU forward replay, float64 probes, hash checks and independent retraining are separate evidence levels.

## Public-safe curation

Personal directories, usernames embedded in paths, environment secrets, credentials, scheduler/account identifiers, private mail and raw agent conversations are not necessary to support scientific conclusions. Keep them out of public history, including release archives. Preserve public-relevant numeric failures and repair semantics in a concise source-linked incident record.

Do not delete scientific anomalies while removing personal identifiers. Do not rewrite `FAIL` as `PASS`, `INCONCLUSIVE` as a negative result, or `NOT_RUN` as a test. Do not concatenate multiple stages into a fictitious sequence of improving model versions.

## Asset availability

The original T1 time_gpu_01 failed-run archive was found and checked at file/member level during assembly: 54 startup errors, no update log, no global test release. Its INCONCLUSIVE decision remains unchanged. See the [incident](incidents/t1_cuda_initialization.md) and [member hashes](../catalog/t1_failed_run_inspection.json). Eight optional historical/supplementary expected archives remain missing. Related private review notes are not equivalent source packages.

## Assembled source identities

Four isolated snapshots are in reference/. All 84 imported Python files and all shell launchers retain their identified source-member bytes. Documentation transformations are in [REDACTION_LOG.json](../REDACTION_LOG.json). Each compatibility manifest is a new public subset, never an original-delivery claim. The [source import manifest](../SOURCE_IMPORT_MANIFEST.json) records archive/member identities and original manifest hashes.
