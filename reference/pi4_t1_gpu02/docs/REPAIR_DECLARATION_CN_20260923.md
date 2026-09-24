# CUDA initialization repair used by T1 GPU02

The original T1 run failed all 54 worker startups before any fitted update: reset_peak_memory_stats preceded lazy CUDA initialization. GPU02 initializes CUDA inside the same fresh fit worker before resetting peak statistics. Initialization remains inside the unchanged parent-origin 600-second budget.

The exact patch is in provenance/CUDA_INIT_REPAIR.patch; original source bytes are in provenance/ORIGINAL_T1_experiment.py.txt. Models, losses, data, seeds, gates and learning rates are unchanged.

The original failed archive was located and checked during local curation. See the [public incident record](../../../docs/incidents/t1_cuda_initialization.md). Personal paths and conversational authorization are omitted from this technical public derivation.
