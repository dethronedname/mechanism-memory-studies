# CI scope

The [evidence-only workflow](../../.github/workflows/evidence.yml) checks small records, imported source-member hashes, standard-library tests, and repository files and links on pushes and pull requests. It uses read-only repository permissions and a five-minute job limit. It does not install CUDA, use a self-hosted runner, download scientific data, deserialize model checkpoints, or train.

The initial hosted run [completed successfully](https://github.com/dethronedname/mechanism-memory-studies/actions/runs/36026321902). Current results are available in [GitHub Actions](https://github.com/dethronedname/mechanism-memory-studies/actions). A successful CI run verifies the committed checks; it is not a fresh neural-model reproduction or independent scientific review.

The checkout action is pinned to upstream commit 11bd71901bbe5b1630ceea73d27597364c9af683 (v4.2.2). Keep action versions explicit and record updates. Upstream identity was checked on 2026-09-24: [release](https://github.com/actions/checkout/releases/tag/v4.2.2) and [commit](https://github.com/actions/checkout/commit/11bd71901bbe5b1630ceea73d27597364c9af683).

The 71 optional ML unit checks were run locally during assembly with CUDA hidden. CI does not install their dependencies. See [reproduction](../reproduction.md) for the distinction between scalar checks, CPU tests, historical replay, and new experiments.
