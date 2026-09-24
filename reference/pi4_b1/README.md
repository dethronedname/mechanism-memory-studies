# B1 public source snapshot

Source authority: baseline_gpu_01_lite.zip. File identities and transformations: [catalog/source_imports.json](../../catalog/source_imports.json).

Model, simulator, loss, protocol, launcher and test Python code is copied byte for byte. T1 contains the previously disclosed CUDA fix. This directory is a public subset. MANIFEST_SHA256.json is a NEW compatibility manifest declared in PUBLIC_SNAPSHOT.json, and must not be represented as the original full delivery manifest.

From the repository root, run: python scripts/verify_reference.py --study b1. This standard-library check does not train or load weights. Limited CPU checks and unexecuted GPU entry points: [reproduction](../../docs/reproduction.md).

Original software in this snapshot is licensed under MIT; original documentation and research records use CC BY 4.0. The sole author is dethronedname. See [license scope](../../docs/project/licensing.md), [license record](../../docs/project/license_review.md), and [citation instructions](../../docs/project/citation.md).
