# CI scope

The supplied `evidence-only` workflow checks small records, public source-member hashes and standard-library tests. It does not install CUDA, use a self-hosted runner, fetch scientific data, deserialize models, or train. It is a draft workflow for the future repository: local equivalents were run during handoff preparation, but no hosted GitHub Actions run has occurred.

The checkout action is pinned to the verified v4.2.2 upstream commit `11bd71901bbe5b1630ceea73d27597364c9af683`, not claimed to be the latest release. Reassess upstream support/security before publication, retain read-only permissions, and record any version update. Do not automatically enable secret-bearing or training workflows.

Upstream identity checked on 2026-09-24: [release](https://github.com/actions/checkout/releases/tag/v4.2.2) and [commit](https://github.com/actions/checkout/commit/11bd71901bbe5b1630ceea73d27597364c9af683).

Workflow-equivalent commands were executed locally. The 71 optional ML unit checks were run locally with CUDA hidden; CI does not install their dependencies. No hosted Actions execution or remote repository was created.
