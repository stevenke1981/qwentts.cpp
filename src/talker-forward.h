#pragma once
// talker-forward.h : prefill forward of the Talker LM.
//
// Takes a precomputed input embedding [T, hidden] f32 row-major and runs
// the 28-layer Qwen3 decoder stack with multimodal RoPE collapsed to 1D
// NEOX, GQA attention with per-head QK-norm, and SwiGLU MLP. The final
// hidden state is RMS-normalised and projected through codec_head to
// produce codebook 0 logits over a 3072-entry vocab.
//
// Phase 4.1+ : eager attention, full F32 compute, no KV cache. The
// graph is built from scratch at every call ; a generation loop will
// later wrap this with a plain causal KV cache (the talker Python
// reference uses pure causal attention, no sliding window).
//
// Optional dump_dir captures bisect-layer activations and the final
// logits in the same f32 binary format the Python reference produces,
// for stage-by-stage cossim validation.

#include "backend.h"
#include "debug.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "talker-weights.h"

#include <cstdint>
#include <vector>

struct TalkerForwardOutput {
    // Final hidden state for the last position [hidden] f32 (post final norm).
    std::vector<float> hidden_last;

    // Codec head logits for the last position [vocab] f32.
    std::vector<float> logits_last;

    int hidden;
    int vocab;
};

// Build and run the prefill graph. The scheduler is responsible for op
// placement (GPU primary, CPU fallback) ; the caller owns its lifetime.
// dump_dir may be NULL.
bool talker_forward_prefill(const TalkerWeights * tw,
                            ggml_backend_sched_t  sched,
                            const float *         input_embed,  // [T, hidden] row-major
                            int                   T,
                            const char *          dump_dir,
                            TalkerForwardOutput * out);
