#!/usr/bin/env python3
# convert.py: Qwen3-TTS HF checkpoint -> GGUF.
#
# Dispatches to converter_tokenizer and converter_talker.
# Reads a HF safetensors checkpoint and writes a GGUF F32 file.
#
# Usage: python convert.py [--outdir models] [checkpoints/]

from pathlib import Path
import os
import re
import sys

from converter_tokenizer import convert_tokenizer_12hz
from converter_talker import convert_talker_base


CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "models"

# Talker checkpoints follow the upstream pattern Qwen3-TTS-12Hz-{size}-{kind}
# where size is 0.6B or 1.7B and kind is Base, CustomVoice or VoiceDesign.
# The compiled regex captures both groups for the GGUF filename suffix and
# the model_size argument fed to convert_talker_base.
TALKER_RE = re.compile(r"^Qwen3-TTS-12Hz-([0-9.]+B)-(\w+)$")


def classify(dir_name: str):
    """Return (kind, model_size) for a known checkpoint directory or None.
    kind is 'tokenizer' or 'talker'. model_size is '0.6B' / '1.7B' for
    talker, None for tokenizer."""
    if "Tokenizer" in dir_name:
        return ("tokenizer", None)
    m = TALKER_RE.match(dir_name)
    if m:
        return ("talker", m.group(1))
    return None


def output_path_for(out_dir: Path, kind: str, dir_name: str) -> Path:
    """Map a checkpoint directory name to its F32 GGUF output path."""
    if kind == "tokenizer":
        return out_dir / "qwen-tokenizer-12hz-F32.gguf"
    m = TALKER_RE.match(dir_name)
    short = f"{m.group(1).lower()}-{m.group(2).lower()}"
    return out_dir / f"qwen-talker-{short}-F32.gguf"


def main() -> int:
    ckpt_root = Path(CHECKPOINT_DIR)
    out_dir   = Path(OUTPUT_DIR)

    if not ckpt_root.is_dir():
        print(f"[Convert] FATAL: {ckpt_root}/ not found")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_unknown: list[str] = []
    rc = 0

    for name in sorted(os.listdir(ckpt_root)):
        ckpt = ckpt_root / name
        if not ckpt.is_dir():
            continue

        info = classify(name)
        if info is None:
            skipped_unknown.append(name)
            continue

        kind, model_size = info
        out = output_path_for(out_dir, kind, name)

        if out.exists():
            print(f"[Convert] skip {out.name}: exists")
            converted += 1
            continue

        if kind == "tokenizer":
            rc |= convert_tokenizer_12hz(ckpt, out)
        else:
            rc |= convert_talker_base(ckpt, out, model_size)
        converted += 1

    if skipped_unknown:
        print(f"[Convert] skipped (unknown): {', '.join(skipped_unknown)}")
    print(f"[Convert] done : {converted} model(s) in {out_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
