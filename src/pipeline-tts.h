#pragma once
// pipeline-tts.h : full TTS pipeline composition (Talker LM + code
// predictor MTP head + optional speaker encoder + 12Hz codec decoder).
//
// Phase 2.0 covers load-only: parse hyperparameters from both GGUF
// files, load every weight tensor on the configured backend, and
// expose the metadata needed to build forward graphs in later phases.
// No graph construction or sampling is wired here yet.

#include "backend.h"
#include "code-predictor-weights.h"
#include "ggml-backend.h"
#include "gguf-weights.h"
#include "kv-cache.h"
#include "pipeline-codec.h"
#include "speaker-encoder-weights.h"
#include "talker-weights.h"

#include <cstdint>
#include <string>
#include <vector>

struct CodecSpecials {
    int pad_id;
    int bos_id;
    int eos_id;
    int think_id;
    int nothink_id;
    int think_bos_id;
    int think_eos_id;
};

struct TextSpecials {
    int im_start_id;
    int im_end_id;
    int tts_pad_id;
    int tts_bos_id;
    int tts_eos_id;
};

struct LanguageEntry {
    std::string name;
    int         id;
};

// Speaker entry for CustomVoice models. id is the codec embedding row id
// inserted in the talker prefix, dialect is empty unless the speaker
// overrides the user supplied language with a dialect lang_id (eric ->
// sichuan_dialect, dylan -> beijing_dialect on the upstream checkpoint).
struct SpeakerEntry {
    std::string name;
    int         id;
    std::string dialect;
};

struct GenerationDefaults {
    bool  do_sample;
    int   top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    bool  subtalker_do_sample;
    int   subtalker_top_k;
    float subtalker_top_p;
    float subtalker_temperature;
    int   max_new_tokens;
};

struct PipelineTTS {
    GGUFModel             gguf_talker;
    TalkerWeights         talker;
    CodePredictorWeights  code_predictor;
    SpeakerEncoderWeights speaker_encoder;
    bool                  has_speaker_encoder;

    PipelineCodec codec;

    std::string tokenizer_type;
    std::string model_size;
    std::string model_type;
    int         num_code_groups;

    CodecSpecials              codec_specials;
    TextSpecials               text_specials;
    std::vector<LanguageEntry> languages;
    std::vector<SpeakerEntry>  speakers;
    GenerationDefaults         gen_defaults;

    BackendPair          bp;
    ggml_backend_t       backend;
    ggml_backend_sched_t sched;

    // Persistent KV caches : the talker holds the LM context, the
    // predictor holds one frame's 16 sub-steps and gets reset every
    // frame in code_predictor_step.
    KVCache talker_kv;
    KVCache code_predictor_kv;
};

// Open the talker GGUF and the codec GGUF, load every module on the
// shared backend. Aborts with a logged error on any missing tensor or
// invalid metadata. Caller frees with pipeline_tts_free.
bool pipeline_tts_load(PipelineTTS * pt, const char * talker_gguf_path, const char * codec_gguf_path, BackendPair bp);

void pipeline_tts_free(PipelineTTS * pt);

struct BPETokenizer;

// Parameters for one synthesis call. Lifetime constraint : text and lang
// are borrowed pointers, must outlive the call. dump_dir, when non-NULL,
// captures step 0 prefill activations plus the codes-full / output-audio
// dumps under the named directory ; debug only, slows the run.
struct PipelineTTSSynthesizeParams {
    const char * text;
    const char * lang;
    const char * instruct;
    const char * speaker;
    const char * ref_audio;
    const char * ref_text;
    int64_t      seed;
    int          max_new_tokens;
    bool         do_sample;
    float        temperature;
    int          top_k;
    float        top_p;
    float        repetition_penalty;
    bool         subtalker_do_sample;
    float        subtalker_temperature;
    int          subtalker_top_k;
    float        subtalker_top_p;
    const char * dump_dir;
};

// Output of one synthesis call. audio is a 24 kHz mono F32 PCM buffer
// already decoded through the codec ; the caller writes it to disk.
struct PipelineTTSSynthesizeOutput {
    std::vector<float> audio;
    int                sample_rate;
};

// Run the full TTS pipeline : prompt assembly, prefill, frame loop with
// sampling, codec decode. Returns false on any failure with a diagnostic
// already routed through qt_log / qt_set_error.
bool pipeline_tts_synthesize(PipelineTTS *                       pt,
                             BPETokenizer *                      tok,
                             const PipelineTTSSynthesizeParams & params,
                             PipelineTTSSynthesizeOutput *       out);
