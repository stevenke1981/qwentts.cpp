#pragma once
// code-predictor-forward.h : run the 5-layer Qwen3 code predictor over a
// growing context to produce the 15 acoustic codes of one audio frame,
// KV cached.
//
// Input :
//   talker_hidden_last [hidden] f32   -- last position hidden state from
//                                        the Talker forward (post final norm)
//   c0                                -- semantic code sampled from the
//                                        Talker codec_head (codebook 0)
// Output :
//   codes[16] = [c0, c1, ..., c15]    -- the full set of codes for one
//                                        frame, ready for decode through
//                                        the codec
//
// The predictor cache is local to a single frame : we reset it at every
// frame, prefill the first two positions (talker_hidden + embed(c0)),
// then decode 14 single-token steps. Total work drops from
// O(sum_{g=0..14} (g+2)^2) = O(1496 token-steps) to O(16) per frame,
// roughly 90x for the inner loop.

#include "code-predictor-weights.h"
#include "ggml-backend.h"
#include "kv-cache.h"
#include "sampling.h"
#include "talker-weights.h"

#include <cstdint>
#include <vector>

struct CodePredictorOutput {
    // Sixteen codes : c0 from the talker plus c1..c15 from the predictor.
    std::vector<int32_t> codes;
};

// Run the predictor for one audio frame. Caller passes the talker hidden
// state for the current frame and the already-sampled c0. Sampling
// parameters control greedy (temperature <= 0) vs stochastic. subseq_base
// is the Philox subsequence of the c0 sample for this step ; the 15
// acoustic samples consume subseq_base + 1 .. subseq_base + 15.
// Returns the full vector of 16 codes. dump_dir may be NULL.
bool code_predictor_step(const TalkerWeights *        tw,
                         const CodePredictorWeights * cw,
                         KVCache *                    kv,
                         ggml_backend_sched_t         sched,
                         const float *                talker_hidden_last,
                         int                          c0,
                         float                        temperature,
                         int                          top_k,
                         float                        top_p,
                         int64_t                      seed,
                         int64_t                      subseq_base,
                         const char *                 dump_dir,
                         CodePredictorOutput *        out);
