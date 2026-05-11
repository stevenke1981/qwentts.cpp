#pragma once
// talker-forward.h : prefill + decode forwards of the Talker LM, KV
// cached.
//
// Both entry points run the same 28-layer Qwen3 decoder stack with
// multimodal RoPE collapsed to 1D NEOX, GQA attention with per-head
// QK-norm, and SwiGLU MLP. The final hidden state is RMS-normalised
// and projected through codec_head to produce codebook 0 logits over a
// 3072-entry vocab.
//
//   talker_forward_prefill : feeds a [T_ctx, hidden] input embedding,
//   rewinds the KV cache to 0 and writes T_ctx positions into it. Used
//   once per utterance at the start of generation, and re-runnable for
//   bisect dumps. Optional dump_dir captures L0/7/14/21/27 hidden taps
//   plus the final hidden and logits.
//
//   talker_forward_decode : feeds a single [1, hidden] embedding,
//   appends one position to the cache at index kv->cur_len, attends to
//   the [0, cur_len+1) window. Called once per generated frame after
//   the predictor has produced its 15 acoustic codes and the loop has
//   summed the codec embeddings into next_emb.
//
// The Python reference uses pure causal attention with no sliding
// window, so the cache is a plain causal ring.

#include "backend.h"
#include "debug.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "kv-cache.h"
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

// Prefill : reset the cache and write T_ctx positions in one shot.
// input_embed is [T, hidden] f32 row-major. dump_dir may be NULL.
bool talker_forward_prefill(const TalkerWeights * tw,
                            KVCache *             kv,
                            ggml_backend_sched_t  sched,
                            const float *         input_embed,
                            int                   T,
                            const char *          dump_dir,
                            TalkerForwardOutput * out);

// Decode : feed exactly one embedding and append one position to the
// cache. Reads positions [0, kv->cur_len + 1). Caller is responsible
// for ensuring kv->cur_len + 1 <= kv->max_seq_len.
bool talker_forward_decode(const TalkerWeights * tw,
                           KVCache *             kv,
                           ggml_backend_sched_t  sched,
                           const float *         input_embed_1,
                           TalkerForwardOutput * out);
