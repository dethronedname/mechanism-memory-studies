#!/usr/bin/env python3
"""Bounded public-draft hygiene scan; not a credential/security guarantee."""
from pathlib import Path
import argparse,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
TEXT_EXT={'.md','.py','.json','.csv','.yml','.yaml','.toml','.cff','.txt'}
SKIP={'.git','__pycache__','.pytest_cache','.venv','venv'}
PATTERNS=[re.compile(r'sandbox:/mnt/data/'),re.compile(r'/(?:home|Users|workspace|data)/[A-Za-z0-9_.-]+/'),
          re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}')]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',action='store_true');a=p.parse_args();issues=[];n=0
    for f in ROOT.rglob('*'):
        rel=f.relative_to(ROOT)
        if any(x in SKIP for x in rel.parts) or not f.is_file():continue
        if f.is_symlink():issues.append({'file':str(rel),'reason':'symlink requires manual inspection'});continue
        if f.stat().st_size>10*1024*1024:issues.append({'file':str(rel),'reason':'exceeds project default-clone file budget; review asset placement'})
        if rel.parts[0] in {'handoff_only','vault','private'}:issues.append({'file':str(rel),'reason':'private layer in public draft'})
        if f.suffix in TEXT_EXT:
            n+=1;s=f.read_text(errors='replace')
            # Scanner patterns are code, not leaked personal values.
            if rel.as_posix()!='scripts/check_public_tree.py':
                for pat in PATTERNS:
                    if pat.search(s):issues.append({'file':str(rel),'reason':'possible private path or credential pattern'});break
        if f.suffix=='.md':
            s=f.read_text()
            for target in re.findall(r'\]\(([^)\s]+)(?:\s+[^)]*)?\)',s):
                if '://' in target or target.startswith(('#','mailto:')):continue
                target=target.split('#',1)[0]
                if target and not (f.parent/target).exists():issues.append({'file':str(rel),'reason':'broken relative link','target':target})
    if a.release:
        for filename in ('LICENSE','CITATION.cff'):
            f=ROOT/filename
            if not f.is_file():issues.append({'file':filename,'reason':'unconfirmed publication metadata'})
            elif re.search(r'REPLACE_|TODO|PLACEHOLDER',f.read_text(errors='replace')):issues.append({'file':filename,'reason':'placeholder in publication metadata'})
    print(json.dumps({'status':'PASS_BOUNDED_DRAFT_SCAN' if not issues else 'REVIEW_REQUIRED','text_files_scanned':n,
                      'excluded_subtrees':sorted(SKIP),'not_a_full_security_or_license_audit':True,'issues':issues},indent=2));return int(bool(issues))
if __name__=='__main__':raise SystemExit(main())
