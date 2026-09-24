from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi4.greybox import run
r=Path(sys.argv[1])
if not (r/'TEST_RELEASE_LOCK.json').exists():raise RuntimeError('finalize neural trials before this scoring-stage control')
run(r/'data',r/'greybox_control')
