#!/usr/bin/env python3
"""Verify explicitly derived public source snapshots; standard library only."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json
ROOT = Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or '\\' in name:
        raise ValueError('Unsafe manifest path: '+name)
    path = root.joinpath(*rel.parts)
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError('Symlink in manifest path: '+name)
    if not path.is_relative_to(root):
        raise ValueError('Path escapes snapshot')
    return path
def verify(root=ROOT, study='all'):
    root=Path(root).resolve()
    manifest=json.loads((root/'SOURCE_IMPORT_MANIFEST.json').read_text())
    snapshots=[s for s in manifest['snapshots'] if study=='all' or s['study']==study]
    if not snapshots: raise ValueError('Unknown study')
    reports=[]
    for s in snapshots:
        target=safe_path(root,s['path'])
        mfpath=target/'MANIFEST_SHA256.json'
        if sha(mfpath)!=s['compatibility_manifest_sha256']:raise ValueError('Public manifest changed: '+s['study'])
        mf=json.loads(mfpath.read_text())
        for rel,digest in mf.items():
            p=safe_path(target,rel)
            if not p.is_file() or sha(p)!=digest:raise ValueError('Snapshot mismatch: '+s['path']+'/'+rel)
        meta=json.loads((target/'PUBLIC_SNAPSHOT.json').read_text())
        if meta['kind']!='DERIVED_PUBLIC_SNAPSHOT' or meta['code_changes_during_public_curation']:
            raise ValueError('Snapshot provenance changed')
        entries=[e for e in manifest['entries'] if e['study']==s['study']]
        for e in entries:
            p=safe_path(root,e['path'])
            if not p.is_relative_to(target):raise ValueError('Cross-study import path')
            if not p.is_file() or sha(p)!=e['sha256'] or p.stat().st_size!=e['size']:raise ValueError('Imported member mismatch: '+e['path'])
            if p.suffix in ('.py','.sh') and e['source_sha256']!=e['sha256']:raise ValueError('Undisclosed code change')
        reports.append(dict(study=s['study'],status='PASS',public_manifest_files=len(mf),imported_members=len(entries),
                            manifest_kind=meta['kind'],original_package_claim=False))
    return reports
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=['all','pi4','c1','b1','t1'],default='all')
    args=p.parse_args()
    try:
        print(json.dumps(dict(status='PASS',snapshots=verify(study=args.study)),indent=2));return 0
    except (ValueError,OSError,KeyError) as e:
        print(json.dumps(dict(status='FAIL',error=str(e))));return 1
if __name__=='__main__':raise SystemExit(main())
