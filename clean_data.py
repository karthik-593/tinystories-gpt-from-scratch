"""
clean_data.py - Clean final_data.jsonl for instruction fine-tuning
-------------------------------------------------------------------
Fixes applied:
  1. Remove exact duplicates (same instruction + same output)
  2. Cap repeated instructions to MAX_PER_INSTRUCTION outputs each
  3. Remove outputs shorter than MIN_OUTPUT_WORDS words
  4. Truncate outputs that would overflow the model's context window, measured
     in actual tokens against the exact prompt template finetune.py builds
     (BOS + instruction/input + response + EOS), not a word-count guess
  5. Fix outputs missing ending punctuation

Usage:
    python clean_data.py
    python clean_data.py --input final_data.jsonl --output final_data_cleaned.jsonl
"""

import json, argparse
from collections import defaultdict
from tokenizers import Tokenizer

from config import ModelConfig

# -- Config --------------------------------------------------------------------
MIN_OUTPUT_WORDS     = 10    # drop outputs shorter than this
MAX_PER_INSTRUCTION  = 5     # keep at most this many outputs per unique instruction

CFG            = ModelConfig()
CONTEXT_LEN    = CFG.context_len            # kept in sync with the model config
TOKENIZER_PATH = CFG.tokenizer_path
MAX_SEQ_LEN    = CONTEXT_LEN + 1            # BOS + prompt + response + EOS budget


# -- Helpers -------------------------------------------------------------------
def word_count(text: str) -> int:
    return len(text.split())


def build_prefix(instruction: str, inp: str) -> str:
    """Same prompt template finetune.py uses — must match exactly for the
    token-length measurement below to reflect what fine-tuning will actually see."""
    if inp.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def token_length(tokenizer: Tokenizer, prefix: str, output: str) -> int:
    """Full sequence length as finetune.py constructs it: BOS + prompt + response + EOS."""
    return 1 + len(tokenizer.encode(prefix + output).ids) + 1


def snap_to_sentence_end(text: str) -> str:
    """Trim text back to the last sentence-ending punctuation (if past halfway), else hard-cut."""
    for punct in (".", "!", "?", '"', "'"):
        idx = text.rfind(punct)
        if idx > len(text) // 2:
            return text[:idx + 1].strip()
    return text.rstrip(",:;") + "."


def fit_output_to_context(tokenizer: Tokenizer, prefix: str, output: str, context_len: int) -> str:
    """Truncate output so BOS + prefix + output + EOS fits within context_len + 1 tokens."""
    prefix_len = len(tokenizer.encode(prefix).ids)
    budget     = context_len - 1 - prefix_len   # reserve 1 token each for BOS and EOS
    output_ids = tokenizer.encode(output).ids
    if len(output_ids) <= budget:
        return output
    truncated_text = tokenizer.decode(output_ids[:max(budget, 0)]).strip()
    return snap_to_sentence_end(truncated_text)


def fix_ending_punctuation(text: str) -> str:
    """Add a period if the output doesn't end with sentence-ending punctuation."""
    text = text.strip()
    if text and text[-1] not in ".!?\"'":
        text += "."
    return text


# -- Main ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="final_data.jsonl")
    parser.add_argument("--output", default="final_data_cleaned.jsonl")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # Load
    raw = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    print(f"Loaded : {len(raw):,} samples from {args.input}")

    stats = defaultdict(int)

    # -- Step 1: Remove exact duplicates --------------------------------------
    seen_pairs = set()
    deduped = []
    for s in raw:
        key = (s.get("instruction", "").strip(), s.get("output", "").strip())
        if key in seen_pairs:
            stats["removed_exact_dup"] += 1
            continue
        seen_pairs.add(key)
        deduped.append(s)
    print(f"Step 1 - Removed exact duplicates      : -{stats['removed_exact_dup']:,}  -> {len(deduped):,} remain")

    # -- Step 2: Cap per-instruction outputs ----------------------------------
    inst_count = defaultdict(int)
    capped = []
    for s in deduped:
        inst = s.get("instruction", "").strip()
        if inst_count[inst] >= MAX_PER_INSTRUCTION:
            stats["removed_capped"] += 1
            continue
        inst_count[inst] += 1
        capped.append(s)
    print(f"Step 2 - Capped to {MAX_PER_INSTRUCTION} outputs/instruction : -{stats['removed_capped']:,}  -> {len(capped):,} remain")

    # -- Steps 3–5: Length filter, truncation, punctuation fix ----------------
    cleaned = []
    for s in capped:
        instruction = s.get("instruction", "").strip()
        inp         = s.get("input", "").strip()
        output      = s.get("output", "").strip()

        # Step 3: Drop very short outputs
        if word_count(output) < MIN_OUTPUT_WORDS:
            stats["removed_short"] += 1
            continue

        # Step 4: Truncate outputs that would overflow the model's context window
        prefix = build_prefix(instruction, inp)
        if token_length(tokenizer, prefix, output) > MAX_SEQ_LEN:
            output = fit_output_to_context(tokenizer, prefix, output, CONTEXT_LEN)
            stats["truncated"] += 1
            if word_count(output) < MIN_OUTPUT_WORDS:
                stats["removed_after_truncation"] += 1
                continue

        # Step 5: Fix missing ending punctuation
        fixed = fix_ending_punctuation(output)
        if fixed != output:
            stats["punct_fixed"] += 1
        output = fixed

        cleaned.append({
            "instruction": instruction,
            "input":       inp,
            "output":      output,
        })

    print(f"Step 3 - Removed short outputs (<{MIN_OUTPUT_WORDS}w)  : -{stats['removed_short']:,}  -> {len(cleaned):,} remain")
    print(f"Step 4 - Truncated outputs exceeding context budget ({MAX_SEQ_LEN} tok) : {stats['truncated']:,} modified"
          f"  (-{stats['removed_after_truncation']:,} dropped, too short after truncation)")
    print(f"Step 5 - Fixed ending punctuation       :  {stats['punct_fixed']:,} modified")

    # -- Write output ----------------------------------------------------------
    with open(args.output, "w", encoding="utf-8") as f:
        for s in cleaned:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Input  : {len(raw):,} samples")
    print(f"  Output : {len(cleaned):,} samples  ({len(raw)-len(cleaned):,} removed, {stats['truncated']} truncated)")
    print(f"  Saved  -> {args.output}")

    # -- Quick sanity check ----------------------------------------------------
    out_lens = [word_count(s["output"]) for s in cleaned]
    n = len(out_lens)
    out_lens.sort()
    print(f"\n  Output length after cleaning (words):")
    print(f"    min={out_lens[0]}  max={out_lens[-1]}  mean={sum(out_lens)/n:.1f}  "
          f"p50={out_lens[n//2]}  p95={out_lens[int(n*0.95)]}")

    seq_lens = sorted(
        token_length(tokenizer, build_prefix(s["instruction"], s["input"]), s["output"])
        for s in cleaned
    )
    over_budget = sum(1 for l in seq_lens if l > MAX_SEQ_LEN)
    print(f"\n  Full sequence length (tokens, BOS+prompt+response+EOS) vs budget={MAX_SEQ_LEN}:")
    print(f"    max={seq_lens[-1]}  p99={seq_lens[int(n*0.99)]}")
    print(f"    Samples still exceeding budget: {over_budget}  (should be 0)")

    from collections import Counter
    inst_freq = Counter(s["instruction"] for s in cleaned)
    over_cap  = sum(1 for v in inst_freq.values() if v > MAX_PER_INSTRUCTION)
    print(f"    Instructions exceeding cap: {over_cap}  (should be 0)")


if __name__ == "__main__":
    main()
