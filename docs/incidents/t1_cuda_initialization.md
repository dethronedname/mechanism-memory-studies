# T1 startup failure and disclosed repair

The original time_gpu_01 archive was found locally and inspected during this curation. Its SHA256 is 4fd094a8827572763a0631de8f63e9b79bde3761a83f6c60e97c7c8413aac1de. All 54 parent receipts report ERROR; all worker logs contain the allocator initialization error. The archive contains no training-update log and no global test-release lock. The recorded scientific status remains INCONCLUSIVE.

The failure occurred when a fresh worker called torch.cuda.reset_peak_memory_stats before completing CUDA lazy initialization. The executed GPU02 source calls torch.cuda.init in that same worker first. Initialization remains inside the original parent-origin 600-second complete-fit budget. Model definitions, losses, seeds, data and scientific gates were unchanged.

- [Inspection and member hashes](../../catalog/t1_failed_run_inspection.json)
- [Exact historical execution patch](../../reference/pi4_t1_gpu02/provenance/CUDA_INIT_REPAIR.patch)
- [Original source bytes](../../reference/pi4_t1_gpu02/provenance/ORIGINAL_T1_experiment.py.txt)
- [Corrected executed source](../../reference/pi4_t1_gpu02/tc1/experiment.py)
- [Corrected run decision](../../results/source_records/t1/run/DECISION.json)

The public declaration removes original personal paths and conversational authorization. Its before/after hashes and transformation are recorded in catalog/redactions.json. The original failed run and original repair declaration remain unchanged in the maintainer's local vault.
