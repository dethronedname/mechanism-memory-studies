"""Regression for the actual fit entry in a fresh CUDA process, without training."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch

def test_cold_cuda_fit_initializes_allocator_before_reset(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('Fresh CUDA initialization regression requires allocated CUDA')
    root = Path(__file__).resolve().parents[1]
    code = r'''
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, sys.argv[1])
import torch
from tc1 import experiment
from tc1.protocol import read, config_for
assert torch.cuda.is_available()
assert not torch.cuda.is_initialized(), 'Must exercise a fresh uninitialized worker'
config = read(Path(sys.argv[1]) / 'configs/time_budget.json')
cfg = config_for(config, config['datasets'][0])
data = Path(sys.argv[2]) / 'data'
data.mkdir()
(data / 'SCALES.json').write_text(json.dumps(dict(general=1., reset=1., difference=1.)))
out = Path(sys.argv[2]) / 'trial'
observed = []
original_reset = torch.cuda.reset_peak_memory_stats
def checked_reset(device):
    assert torch.cuda.is_initialized(), 'Peak stats reset before CUDA initialization'
    original_reset(device)
    observed.append(str(device))
class DataLoadReached(Exception):
    pass
def stop_before_data_load(*args, **kwargs):
    raise DataLoadReached()
torch.cuda.reset_peak_memory_stats = checked_reset
experiment.load = stop_before_data_load
started = time.monotonic()
try:
    experiment.fit(cfg, config['methods'][0], config['initialization_seeds'][0], data, out, 'cuda:0', started)
except DataLoadReached:
    pass
else:
    raise AssertionError('Expected diagnostic stop before data loading')
assert observed == ['cuda:0']
assert not (out / 'initial.pt').exists()
assert not (out / 'training.jsonl').exists()
print(json.dumps(dict(status='PASS', initialized=torch.cuda.is_initialized(), resets=observed,
                     training_started=False, elapsed_since_parent_origin=time.monotonic()-started)))
'''
    result = subprocess.run([sys.executable, '-c', code, str(root), str(tmp_path)],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout.strip().splitlines()[-1])
    assert record['status'] == 'PASS' and record['initialized'] and not record['training_started']
