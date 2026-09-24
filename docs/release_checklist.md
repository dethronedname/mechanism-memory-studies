# Release checklist

## Before public release

- Authorship is recorded as the sole human author chumingyzx, using a public pseudonym. The repository is chumingyzx/mechanism-memory-studies and is private for author review. Public visibility is a separate author decision. Do not invent ORCIDs, affiliations, DOI or publication acceptance.
- The owner's authorization has been applied: original code uses MIT; original documentation/data use CC BY 4.0. The original-package notice scan and dependency boundaries are recorded in [license review](../LICENSE_REVIEW.md).
- Keep third-party provenance current when adding material. The current draft references external dependencies without vendoring their implementations; cited methods and local implementations are distinguished in [related work](related_work.md).
- Preserve source and run identities. The T1 corrected GPU02 implementation must carry its patch provenance. Keep old failures without personal paths.
- Verify every README command in the assembled tree. A frozen package requiring a private manifest is not yet a working public quick start.
- Regenerate every public numeric table from source JSON. Preserve all prescribed comparator rows, null boundaries, dataset and seed grouping.
- Confirm what the default clone, each release asset, and optional historical snapshot actually contain. Publish a real checksum and URL only after the asset exists.
- Verify that no private scratchpads, raw chat, host paths, credentials, scheduler/account IDs, or internal agent instructions have entered Git history or release assets. Automated scanning is not a complete security review.
- Separate `LICENSE`/`NOTICE`, if authorized, from any research conclusion. Do not automatically assign one license to third-party raw data or weights.

## Size and storage

Default clone target: **under 30 MB as a project choice**, not a GitHub limit. Keep source, config, small evidence and tables in Git; regenerated arrays and full run logs belong in separate versioned assets or a local vault. Small checkpoint bundles may be release assets with hashes. Do not upload old multi-GB ZIPs wholesale.

GitHub's current documentation states that ordinary Git files above 100 MiB are blocked, recommends small repositories, and documents release assets as an alternative. Verify current hosting limits when uploading, and split an oversized original archive only with a part manifest; do not silently replace its source hash.

## Citation and metadata

[CITATION.cff](../CITATION.cff) records chumingyzx as the sole author and the applicable licenses. It records the actual private repository URL and omits a DOI, release version, and release date until those exist. The [research citation request](../CITATION.md) does not add conditions to MIT or CC BY 4.0. Keep source experiment dates distinct from curation/release dates.

## Reference documentation checked during curation (2026-09-24)

- GitHub: [large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
- GitHub: [citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).
- GitHub: [licensing a repository](https://docs.github.com/articles/licensing-a-repository).

These sources document repository mechanics, not the scientific claims in this collection.
