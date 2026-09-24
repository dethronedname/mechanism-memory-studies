# Reproduction levels and verified entry points

## A. Recorded results and source identity — standard library only

From the repository root:

~~~bash
python scripts/rebuild_results.py
python scripts/verify_reference.py
python -m unittest discover -s tests -v
python scripts/check_public_tree.py
~~~

All commands were run during local assembly; 20 standard-library tests passed. These commands do not load a neural model, trained checkpoint or GPU. The source verifier checks imported member hashes and separately identified public manifests.

The deliberate refresh command was also tested and produces the same derived bytes:

~~~bash
python scripts/rebuild_results.py --write
~~~

## B. Limited CPU implementation checks — existing ML environment

~~~bash
python scripts/check_reference_cpu.py
~~~

Executed results: PI4 9, C1 11, B1 39, T1 12 tests passed (71 total). An explicit function allowlist runs in separate processes, with CUDA hidden, one CPU thread and a 90-second cap per study. It does not invoke fit/stage launchers, read historical trained weights or release a test set. Checks cover synthetic invariance, loss identities, source locks and budget selection.

Use an existing environment with NumPy, SciPy, PyTorch and pytest. The command does not install or upgrade software. Assembly used Python 3.12.12, NumPy 2.2.6, SciPy 1.16.2, PyTorch 2.8.0+cu128 and pytest 8.3.5. [Actual historical environments](../catalog/runtime_environments.json) are selected run fields, not a complete dependency lockfile. Remaining original tests, including T1's CUDA regression, are retained but outside this CPU allowlist.

## C. Historical checkpoint replay — external assets required

Weights, predictions, full curves and generated data are outside the default clone. The [inventory](../catalog/assets.json) identifies verified local originals; public URLs remain null until an authorized public derivative exists.

Public snapshots contain model, simulator, objective, evaluator, audit, configs and launchers. Their documentation and delivery manifests differ from the original full packages. Historical source-identity checks may reject using a public subset with an old run. Use exact run-matched original source/evidence or a separately validated replay adapter. This draft does not claim an operational unified historical replay CLI.

Full neural replay was not run during curation. Never alter an old SOURCE_MANIFEST or test lock to bypass a check. Recreate omitted NPZ files only in new scratch storage with the frozen generator/config and verify content hashes. CPU32/CPU64 replay and both numerical labels remain evidence from the completed historical pipeline.

## D. New replication — GPU commands not executed during curation

Original Python and shell launchers are included byte for byte. Their read-only verification commands were tested in each stage directory:

| Study | Working directory | Read-only verification | Historical GPU entry, unexecuted here |
|---|---|---|---|
| PI4 | reference/pi4 | python scripts/run.py verify | bash scripts/run_gpu.sh runs/fresh_replication |
| C1 | reference/pi4_c1 | python scripts/run.py verify | bash scripts/run_gpu.sh runs/fresh_replication |
| B1 | reference/pi4_b1 | python scripts/run.py verify | bash scripts/run_gpu.sh runs/fresh_replication |
| T1 | reference/pi4_t1_gpu02 | python scripts/run_t1.py verify | bash scripts/run_time_gpu.sh runs/fresh_replication |

Each MANIFEST_SHA256.json is a **new public-subset compatibility manifest**, declared in PUBLIC_SNAPSHOT.json. It is not an original-delivery identity. [Import records](../SOURCE_IMPORT_MANIFEST.json) retain source membership and original manifest hashes. Python and shell code is unchanged during public curation; T1's prior execution fix remains in the [incident record](incidents/t1_cuda_initialization.md).

Training requires an explicit resource allocation and fresh output directory. Historical published tests are no longer blind. An independent study must specify new data splits, information access, selection and stopping rules with new provenance. Do not train in a closed historical run.

T1's retained scripts/run.py is B1 lineage; T1 uses scripts/run_t1.py and run_time_gpu.sh. Absent CUDA yields NOT_RUN, not substituted CPU scientific scores.

## Verification scope

Recorded scientific outcomes are unchanged. Rebuilding scalar tables, CPU unit checks, model replay and independent training are distinct evidence levels. [CI](ci_scope.md) only runs standard-library checks; it does not start training or hosted GPU work.
