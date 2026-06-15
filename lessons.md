---
## Lesson #1 — 2026-06-15
**Trigger:** Windows CUDA target verification failed because MSBuild saw an empty or malformed CUDA Toolkit path.
**Rule:** Before verifying a Windows CUDA build, set `CUDA_PATH`, `CUDA_PATH_V13_2`, and `/p:CudaToolkitDir=` to the CMake-selected toolkit root with a trailing backslash.
**Source:** Fix MSVC C4819 warnings in ggml-cuda build
