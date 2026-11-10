# Profiling artifacts

- `nsys_baseline.nsys-rep`: Nsight Systems capture for the engine baseline.
- `nsys_baseline.sqlite`: exported queryable Nsight Systems database.
- `ncu_blockwise.ncu-rep`: Nsight Compute baseline capture.
- `ncu_triton.ncu-rep`: Nsight Compute capture containing `_paged_decode_kernel`.
- `trace_baseline/`: FlexEngine JSONL metadata and step traces from the same run.

Regenerate the files with the exact commands in `docs/NSIGHT_BASELINE.md`.
The `.nsys-rep` and `.ncu-rep` files are the source artifacts; prose reports
must remain derivable from them and the colocated JSONL trace.
