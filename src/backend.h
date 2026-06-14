#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. Single shared backend across modules in the
// same binary, refcounted.
//
// Implementation lives in backend.cpp — header modules include this
// for the type and function declarations only.

#include "ggml-backend.h"

#include <cstddef>

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
};

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Subsequent calls reuse the same backend (single VMM pool). Returns a
// BackendPair with .backend == NULL when initialisation fails; the caller
// must check this before passing it to any pipeline_*_load.
BackendPair backend_init(const char * label);

// Release a backend reference. Frees GPU + CPU backends when refcount hits 0.
void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend);

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
// When a GPU is present, use its host buffer type for the CPU backend.
// Pinned memory lets the scheduler keep more ops on GPU instead of
// falling back to CPU with plain malloc.
ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes);
