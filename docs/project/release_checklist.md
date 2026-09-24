# Repository maintenance and release checklist

## Metadata and documentation

- Keep the sole-author attribution, repository URL, [citation metadata](../../CITATION.cff), and [license scope](licensing.md) consistent.
- Use real version identifiers, dates, and DOIs only when they exist. Public repository access is separate from a tagged software release or a research publication.
- Keep [third-party provenance](license_review.md) current when adding material. External dependencies retain their own licenses.
- Keep English and Chinese entry points, the [documentation index](../README.md), and relative links current after moving files.

## Evidence and verification

- Preserve frozen source and run identities, the T1 repair, original failures, test locks, and all scientific decisions.
- Run the [standard-library checks](../reproduction.md) and the release-mode repository scan before distributing a revision.
- Update the source, redaction, and repository file manifests when their covered files change. Do not substitute a new public-subset hash for an original delivery hash.
- Keep all prescribed comparator rows, dataset/seed grouping, and null boundaries when regenerating tables.
- Exclude host paths, credentials, internal conversations, and unrelated local materials from commits and release assets.

## Asset storage

The default clone contains code, configuration, small evidence, and tables. Its size target is under 30 MB. Weights, generated arrays, and full logs are external assets listed in the [inventory](../../catalog/assets.json); repository publication does not make those assets available.

For each future downloadable asset, provide its actual URL, checksum, license scope, and source identity. Check current hosting limits before uploading large assets. A DOI and formal GitHub Release are optional.

## Reference documentation

- GitHub: [citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).
- GitHub: [licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
- GitHub: [large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
