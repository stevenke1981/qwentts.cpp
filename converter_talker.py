"""Qwen3-TTS Talker LM (autoregressive + MTP code predictor) converter → GGUF."""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

import gguf


def rename_talker_layer(name: str) -> str:
    # talker.model.layers.{i}.input_layernorm.weight             -> talker.blk.{i}.attn_norm.weight
    # talker.model.layers.{i}.post_attention_layernorm.weight    -> talker.blk.{i}.ffn_norm.weight
    # talker.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight    -> talker.blk.{i}.attn_{q,k,v,output}.weight
    # talker.model.layers.{i}.self_attn.{q,k}_norm.weight        -> talker.blk.{i}.attn_{q,k}_norm.weight
    # talker.model.layers.{i}.mlp.{gate,up,down}_proj.weight     -> talker.blk.{i}.ffn_{gate,up,down}.weight
    assert name.startswith("talker.model.layers.")
    return _xlate_lm_layer(name, prefix="talker.model.layers.", out_prefix="talker.blk")


def rename_code_predictor_layer(name: str) -> str:
    # talker.code_predictor.model.layers.{i}.<role>  ->  code_pred.blk.{i}.<role>
    assert name.startswith("talker.code_predictor.model.layers.")
    return _xlate_lm_layer(name, prefix="talker.code_predictor.model.layers.", out_prefix="code_pred.blk")


def rename_text_projection(name: str) -> str:
    # talker.text_projection.linear_fc{1,2}.{weight,bias}  ->  talker.text_proj.fc{1,2}.{weight,bias}
    assert name.startswith("talker.text_projection.")
    return name.replace("talker.text_projection.linear_fc", "talker.text_proj.fc")


# Common HF -> llama.cpp suffix table for the Qwen3 backbone, shared between
# the Talker and the Code Predictor since both follow the same architecture.
_LM_LAYER_SUFFIX = {
    "input_layernorm.weight":           "attn_norm.weight",
    "post_attention_layernorm.weight":  "ffn_norm.weight",
    "self_attn.q_proj.weight":          "attn_q.weight",
    "self_attn.k_proj.weight":          "attn_k.weight",
    "self_attn.v_proj.weight":          "attn_v.weight",
    "self_attn.o_proj.weight":          "attn_output.weight",
    "self_attn.q_norm.weight":          "attn_q_norm.weight",
    "self_attn.k_norm.weight":          "attn_k_norm.weight",
    "mlp.gate_proj.weight":             "ffn_gate.weight",
    "mlp.up_proj.weight":               "ffn_up.weight",
    "mlp.down_proj.weight":             "ffn_down.weight",
}


def _xlate_lm_layer(name: str, prefix: str, out_prefix: str) -> str:
    rest      = name[len(prefix):]
    dot       = rest.find(".")
    layer_idx = rest[:dot]
    suffix    = rest[dot + 1:]
    new_suffix = _LM_LAYER_SUFFIX.get(suffix)
    if new_suffix is None:
        raise ValueError(f"Unhandled LM layer suffix : {suffix} (full name : {name})")
    return f"{out_prefix}.{layer_idx}.{new_suffix}"


def load_bpe_vocab(checkpoint_dir: Path):
    # Load Qwen2 BPE vocab + merges from a HF checkpoint directory and
    # produce the (tokens, token_types, merges) triple expected by the
    # GGUF tokenizer convention. Special tokens listed in
    # tokenizer_config.json:added_tokens_decoder are tagged as user-defined
    # (token_type 4) so the runtime can recognise them as verbatim chunks.
    vocab = json.loads((checkpoint_dir / "vocab.json").read_text())
    tok_cfg = json.loads((checkpoint_dir / "tokenizer_config.json").read_text())
    added = tok_cfg.get("added_tokens_decoder", {})

    max_id = max(vocab.values())
    for sid in added.keys():
        max_id = max(max_id, int(sid))

    tokens = [None] * (max_id + 1)
    token_types = [1] * (max_id + 1)  # 1 = normal

    for tok, tid in vocab.items():
        tokens[tid] = tok

    for sid_str, info in added.items():
        sid = int(sid_str)
        tokens[sid] = info["content"]
        token_types[sid] = 4  # 4 = user-defined / special

    # Empty slots (gaps in id space) get a placeholder so the GGUF array
    # has no None entries.
    for i, tok in enumerate(tokens):
        if tok is None:
            tokens[i] = f"<|unused-{i}|>"
            token_types[i] = 5  # 5 = unused

    # merges.txt : first line may be a "#version" comment, skip it.
    merges_lines = (checkpoint_dir / "merges.txt").read_text().splitlines()
    merges = [ln for ln in merges_lines if ln and not ln.startswith("#")]

    return tokens, token_types, merges


def convert_talker_base(checkpoint_dir: Path, out_path: Path, model_size: str) -> int:
    cfg_path = checkpoint_dir / "config.json"
    st_path = checkpoint_dir / "model.safetensors"
    gen_path = checkpoint_dir / "generation_config.json"
    if not cfg_path.is_file() or not st_path.is_file():
        print(f"[Convert] FATAL: missing checkpoint files in {checkpoint_dir}")
        return 1

    cfg = json.loads(cfg_path.read_text())
    talker_cfg = cfg["talker_config"]
    cp_cfg = talker_cfg["code_predictor_config"]
    spk_cfg = cfg.get("speaker_encoder_config")
    gen_cfg = json.loads(gen_path.read_text()) if gen_path.is_file() else {}

    arch = "qwen3-tts"
    writer = gguf.GGUFWriter(str(out_path), arch)

    writer.add_string("general.name", f"Qwen3-TTS-12Hz-{model_size}-{cfg['tts_model_type']}")

    # Top-level TTS metadata
    writer.add_string("qwen3-tts.tokenizer_type", cfg["tokenizer_type"])
    writer.add_string("qwen3-tts.model_size", cfg["tts_model_size"])
    writer.add_string("qwen3-tts.model_type", cfg["tts_model_type"])
    writer.add_uint32("qwen3-tts.num_code_groups", talker_cfg["num_code_groups"])

    # Talker LM hyperparameters
    writer.add_uint32("qwen3-tts.talker.embedding_length", talker_cfg["hidden_size"])
    writer.add_uint32("qwen3-tts.talker.feed_forward_length", talker_cfg["intermediate_size"])
    writer.add_uint32("qwen3-tts.talker.block_count", talker_cfg["num_hidden_layers"])
    writer.add_uint32("qwen3-tts.talker.attention.head_count", talker_cfg["num_attention_heads"])
    writer.add_uint32("qwen3-tts.talker.attention.head_count_kv", talker_cfg["num_key_value_heads"])
    writer.add_uint32("qwen3-tts.talker.attention.key_length", talker_cfg["head_dim"])
    writer.add_uint32("qwen3-tts.talker.vocab_size", talker_cfg["vocab_size"])
    writer.add_uint32("qwen3-tts.talker.text_vocab_size", talker_cfg["text_vocab_size"])
    writer.add_uint32("qwen3-tts.talker.text_hidden_size", talker_cfg["text_hidden_size"])
    writer.add_uint32("qwen3-tts.talker.context_length", talker_cfg["max_position_embeddings"])
    writer.add_float32("qwen3-tts.talker.rope.freq_base", float(talker_cfg["rope_theta"]))
    writer.add_float32("qwen3-tts.talker.attention.layer_norm_rms_epsilon", float(talker_cfg["rms_norm_eps"]))
    writer.add_uint32("qwen3-tts.talker.position_id_per_seconds", talker_cfg["position_id_per_seconds"])
    rope_scaling = talker_cfg.get("rope_scaling") or {}
    if "mrope_section" in rope_scaling:
        writer.add_array("qwen3-tts.talker.rope.mrope_section", rope_scaling["mrope_section"])
        writer.add_bool("qwen3-tts.talker.mrope_interleaved", bool(rope_scaling.get("interleaved", False)))

    # Code predictor (subtalker) hyperparameters
    writer.add_uint32("qwen3-tts.code_pred.embedding_length", cp_cfg["hidden_size"])
    writer.add_uint32("qwen3-tts.code_pred.feed_forward_length", cp_cfg["intermediate_size"])
    writer.add_uint32("qwen3-tts.code_pred.block_count", cp_cfg["num_hidden_layers"])
    writer.add_uint32("qwen3-tts.code_pred.attention.head_count", cp_cfg["num_attention_heads"])
    writer.add_uint32("qwen3-tts.code_pred.attention.head_count_kv", cp_cfg["num_key_value_heads"])
    writer.add_uint32("qwen3-tts.code_pred.attention.key_length", cp_cfg["head_dim"])
    writer.add_uint32("qwen3-tts.code_pred.vocab_size", cp_cfg["vocab_size"])
    writer.add_uint32("qwen3-tts.code_pred.context_length", cp_cfg["max_position_embeddings"])
    writer.add_float32("qwen3-tts.code_pred.rope.freq_base", float(cp_cfg["rope_theta"]))
    writer.add_float32("qwen3-tts.code_pred.attention.layer_norm_rms_epsilon", float(cp_cfg["rms_norm_eps"]))

    # Speaker encoder hyperparameters (Base checkpoints only). CustomVoice
    # and VoiceDesign carry no speaker encoder so the keys are skipped
    # entirely, the runtime detects the absence via tensor lookup.
    if spk_cfg is not None:
        writer.add_uint32("qwen3-tts.spk_enc.embedding_length", spk_cfg["enc_dim"])
        writer.add_uint32("qwen3-tts.spk_enc.sample_rate", spk_cfg["sample_rate"])

    # Codec stream special tokens
    writer.add_uint32("qwen3-tts.codec.pad_id", talker_cfg["codec_pad_id"])
    writer.add_uint32("qwen3-tts.codec.bos_id", talker_cfg["codec_bos_id"])
    writer.add_uint32("qwen3-tts.codec.eos_id", talker_cfg["codec_eos_token_id"])
    writer.add_uint32("qwen3-tts.codec.think_id", talker_cfg["codec_think_id"])
    writer.add_uint32("qwen3-tts.codec.nothink_id", talker_cfg["codec_nothink_id"])
    writer.add_uint32("qwen3-tts.codec.think_bos_id", talker_cfg["codec_think_bos_id"])
    writer.add_uint32("qwen3-tts.codec.think_eos_id", talker_cfg["codec_think_eos_id"])

    # Language id table flattened to two parallel arrays. Names stay as in
    # the upstream config so the runtime can pass --lang chinese verbatim.
    lang_map = talker_cfg.get("codec_language_id") or {}
    lang_names = list(lang_map.keys())
    lang_ids = [int(lang_map[k]) for k in lang_names]
    writer.add_array("qwen3-tts.codec.language_names", lang_names)
    writer.add_array("qwen3-tts.codec.language_ids", lang_ids)

    # Speaker table for CustomVoice variants. Three parallel arrays indexed
    # by speaker position : name, codec embedding id, and optional dialect
    # name pulled from codec_language_id. Empty dialect string means the
    # speaker keeps the user supplied language. Skipped entirely for Base
    # and VoiceDesign which have no spk_id map.
    spk_map = talker_cfg.get("spk_id") or {}
    if spk_map:
        dialect_map = talker_cfg.get("spk_is_dialect") or {}
        spk_names    = list(spk_map.keys())
        spk_ids      = [int(spk_map[k]) for k in spk_names]
        spk_dialects = [dialect_map.get(k) or "" for k in spk_names]
        spk_dialects = [d if isinstance(d, str) else "" for d in spk_dialects]
        writer.add_array("qwen3-tts.codec.speaker_names", spk_names)
        writer.add_array("qwen3-tts.codec.speaker_ids", spk_ids)
        writer.add_array("qwen3-tts.codec.speaker_dialects", spk_dialects)

    # Text-side special tokens (Qwen2 BPE)
    writer.add_uint32("qwen3-tts.text.im_start_id", cfg["im_start_token_id"])
    writer.add_uint32("qwen3-tts.text.im_end_id", cfg["im_end_token_id"])
    writer.add_uint32("qwen3-tts.text.tts_pad_id", cfg["tts_pad_token_id"])
    writer.add_uint32("qwen3-tts.text.tts_bos_id", cfg["tts_bos_token_id"])
    writer.add_uint32("qwen3-tts.text.tts_eos_id", cfg["tts_eos_token_id"])

    # Default sampling parameters from generation_config.json
    if gen_cfg:
        if "do_sample" in gen_cfg:
            writer.add_bool("generation.do_sample", bool(gen_cfg["do_sample"]))
        if "top_k" in gen_cfg:
            writer.add_uint32("generation.top_k", int(gen_cfg["top_k"]))
        if "top_p" in gen_cfg:
            writer.add_float32("generation.top_p", float(gen_cfg["top_p"]))
        if "temperature" in gen_cfg:
            writer.add_float32("generation.temperature", float(gen_cfg["temperature"]))
        if "repetition_penalty" in gen_cfg:
            writer.add_float32("generation.repetition_penalty", float(gen_cfg["repetition_penalty"]))
        if "subtalker_dosample" in gen_cfg:
            writer.add_bool("generation.subtalker_do_sample", bool(gen_cfg["subtalker_dosample"]))
        if "subtalker_top_k" in gen_cfg:
            writer.add_uint32("generation.subtalker_top_k", int(gen_cfg["subtalker_top_k"]))
        if "subtalker_top_p" in gen_cfg:
            writer.add_float32("generation.subtalker_top_p", float(gen_cfg["subtalker_top_p"]))
        if "subtalker_temperature" in gen_cfg:
            writer.add_float32("generation.subtalker_temperature", float(gen_cfg["subtalker_temperature"]))
        if "max_new_tokens" in gen_cfg:
            writer.add_uint32("generation.max_new_tokens", int(gen_cfg["max_new_tokens"]))

    # BPE tokenizer payload
    bpe_tokens, bpe_token_types, bpe_merges = load_bpe_vocab(checkpoint_dir)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", bpe_tokens)
    writer.add_array("tokenizer.ggml.token_type", bpe_token_types)
    writer.add_array("tokenizer.ggml.merges", bpe_merges)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 151643)  # <|endoftext|>

    # Top level talker tensors. Renames mirror the koboldcpp TENSOR_MAP.
    TALKER_TOP = {
        "talker.model.codec_embedding.weight":         "talker.codec_embd.weight",
        "talker.model.text_embedding.weight":          "talker.text_embd.weight",
        "talker.model.norm.weight":                    "talker.output_norm.weight",
        "talker.codec_head.weight":                    "talker.codec_head.weight",
        "talker.text_projection.linear_fc1.weight":    "talker.text_proj.fc1.weight",
        "talker.text_projection.linear_fc1.bias":      "talker.text_proj.fc1.bias",
        "talker.text_projection.linear_fc2.weight":    "talker.text_proj.fc2.weight",
        "talker.text_projection.linear_fc2.bias":      "talker.text_proj.fc2.bias",
    }

    # Top level code predictor tensors.
    CP_TOP = {
        "talker.code_predictor.model.norm.weight":                       "code_pred.output_norm.weight",
        "talker.code_predictor.small_to_mtp_projection.weight":          "code_pred.mtp_proj.weight",
        "talker.code_predictor.small_to_mtp_projection.bias":            "code_pred.mtp_proj.bias",
    }

    # Speaker encoder rename table, koboldcpp SPEAKER_ENCODER_PATTERNS plus
    # standalone tensors. block 0 is the entry conv, blocks 1 to 3 hold the
    # Res2Net + SE + TDNN stack.
    SPK_TOP = {
        "speaker_encoder.blocks.0.conv.weight":     "spk_enc.conv0.weight",
        "speaker_encoder.blocks.0.conv.bias":       "spk_enc.conv0.bias",
        "speaker_encoder.asp.conv.weight":          "spk_enc.asp.conv.weight",
        "speaker_encoder.asp.conv.bias":            "spk_enc.asp.conv.bias",
        "speaker_encoder.asp.tdnn.conv.weight":     "spk_enc.asp.tdnn.weight",
        "speaker_encoder.asp.tdnn.conv.bias":       "spk_enc.asp.tdnn.bias",
        "speaker_encoder.mfa.conv.weight":          "spk_enc.mfa.weight",
        "speaker_encoder.mfa.conv.bias":            "spk_enc.mfa.bias",
        "speaker_encoder.fc.weight":                "spk_enc.fc.weight",
        "speaker_encoder.fc.bias":                  "spk_enc.fc.bias",
    }

    def rename_speaker_encoder_block(k: str) -> str:
        # speaker_encoder.blocks.{i}.res2net_block.blocks.{j}.conv.{weight,bias}  ->  spk_enc.blk.{i}.res2net.{j}.{weight,bias}
        # speaker_encoder.blocks.{i}.se_block.conv{1,2}.{weight,bias}             ->  spk_enc.blk.{i}.se.conv{1,2}.{weight,bias}
        # speaker_encoder.blocks.{i}.tdnn{1,2}.conv.{weight,bias}                 ->  spk_enc.blk.{i}.tdnn{1,2}.{weight,bias}
        parts = k.split(".")
        idx   = parts[2]
        if parts[3] == "res2net_block":
            sub    = parts[5]
            suffix = parts[-1]
            return f"spk_enc.blk.{idx}.res2net.{sub}.{suffix}"
        if parts[3] == "se_block":
            which  = parts[4]  # conv1 or conv2
            suffix = parts[-1]
            return f"spk_enc.blk.{idx}.se.{which}.{suffix}"
        if parts[3] in ("tdnn1", "tdnn2"):
            return f"spk_enc.blk.{idx}.{parts[3]}.{parts[-1]}"
        raise ValueError(f"Unhandled speaker encoder block tensor : {k}")

    # Walk safetensors
    n_added = 0
    n_unhandled = 0
    with safe_open(str(st_path), framework="pt") as f:
        all_keys = sorted(list(f.keys()))
        for k in all_keys:
            t = f.get_tensor(k)
            arr = t.float().numpy()

            # Talker transformer layers
            if k.startswith("talker.model.layers."):
                writer.add_tensor(rename_talker_layer(k), arr)
                n_added += 1
                continue

            # Talker top level (codec_embedding, text_embedding, norm,
            # codec_head, text_projection)
            if k in TALKER_TOP:
                writer.add_tensor(TALKER_TOP[k], arr)
                n_added += 1
                continue

            # Code predictor transformer layers
            if k.startswith("talker.code_predictor.model.layers."):
                writer.add_tensor(rename_code_predictor_layer(k), arr)
                n_added += 1
                continue

            # Code predictor codec embeddings (one per acoustic codebook)
            if k.startswith("talker.code_predictor.model.codec_embedding."):
                idx = k.split(".")[4]
                writer.add_tensor(f"code_pred.codec_embd.{idx}.weight", arr)
                n_added += 1
                continue

            # Code predictor lm heads (one per acoustic codebook)
            if k.startswith("talker.code_predictor.lm_head."):
                idx = k.split(".")[3]
                writer.add_tensor(f"code_pred.lm_head.{idx}.weight", arr)
                n_added += 1
                continue

            # Code predictor top level (final norm, MTP projection)
            if k in CP_TOP:
                writer.add_tensor(CP_TOP[k], arr)
                n_added += 1
                continue

            # Speaker encoder, only present in Base checkpoints
            if k in SPK_TOP:
                writer.add_tensor(SPK_TOP[k], arr)
                n_added += 1
                continue
            if k.startswith("speaker_encoder.blocks.") and not k.startswith("speaker_encoder.blocks.0."):
                writer.add_tensor(rename_speaker_encoder_block(k), arr)
                n_added += 1
                continue

            print(f"[Convert] WARNING: unhandled tensor : {k}  shape={tuple(t.shape)}")
            n_unhandled += 1

    print(f"[Convert] Tensors: {n_added} written, {n_unhandled} unhandled")
    print(f"[Convert] BPE: {len(bpe_tokens)} tokens, {len(bpe_merges)} merges")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[Convert] Wrote {out_path}")
    return 0
