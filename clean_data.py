"""
clean_data.py - Clean final_data.jsonl for instruction fine-tuning
-------------------------------------------------------------------
Fixes applied:
  1. Remove exact duplicates (same instruction + same output)
  2. Cap repeated instructions to MAX_PER_INSTRUCTION outputs each
  3. Remove outputs shorter than MIN_OUTPUT_WORDS words
  4. Truncate outputs longer than MAX_OUTPUT_WORDS words at sentence boundary
  5. Fix outputs missing ending punctuation

Usage:
    python clean_data.py
    python clean_data.py --input final_data.jsonl --output final_data_cleaned.jsonl
"""

import json, re, argparse
from collections import defaultdict

# -- Config --------------------------------------------------------------------
MIN_OUTPUT_WORDS      = 10    # drop outputs shorter than this
MAX_OUTPUT_WORDS      = 130   # truncate outputs longer than this (context_len=256 limit)
MAX_PER_INSTRUCTION   = 5     # keep at most this many outputs per unique instruction


# -- Helpers -------------------------------------------------------------------
def word_count(text: str) -> int:
    return len(text.split())


def truncate_at_sentence(text: str, max_words: int) -> str:
    """Truncate text to at most max_words words, ending at a sentence boundary."""
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    # Walk back to find the last sentence-ending punctuation
    for punct in (".", "!", "?", '"', "'"):
        idx = truncated.rfind(punct)
        if idx > len(truncated) // 2:   # at least halfway through
            return truncated[:idx + 1].strip()

    # No sentence boundary found - hard cut and add period
    return truncated.rstrip(",:;") + "."


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
        output = s.get("output", "").strip()

        # Step 3: Drop very short outputs
        if word_count(output) < MIN_OUTPUT_WORDS:
            stats["removed_short"] += 1
            continue

        # Step 4: Truncate long outputs at sentence boundary
        if word_count(output) > MAX_OUTPUT_WORDS:
            output = truncate_at_sentence(output, MAX_OUTPUT_WORDS)
            stats["truncated"] += 1

        # Step 5: Fix missing ending punctuation
        fixed = fix_ending_punctuation(output)
        if fixed != output:
            stats["punct_fixed"] += 1
        output = fixed

        cleaned.append({
            "instruction": s.get("instruction", "").strip(),
            "input":       s.get("input", "").strip(),
            "output":      output,
        })

    print(f"Step 3 - Removed short outputs (<{MIN_OUTPUT_WORDS}w)  : -{stats['removed_short']:,}  -> {len(cleaned):,} remain")
    print(f"Step 4 - Truncated long outputs (>{MAX_OUTPUT_WORDS}w)  :  {stats['truncated']:,} modified")
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
    print(f"\n  Output length after cleaning:")
    n = len(out_lens)
    out_lens.sort()
    print(f"    min={out_lens[0]}  max={out_lens[-1]}  mean={sum(out_lens)/n:.1f}  "
          f"p50={out_lens[n//2]}  p95={out_lens[int(n*0.95)]}")

    from collections import Counter
    inst_freq = Counter(s["instruction"] for s in cleaned)
    over_cap  = sum(1 for v in inst_freq.values() if v > MAX_PER_INSTRUCTION)
    print(f"    Instructions exceeding cap: {over_cap}  (should be 0)")


if __name__ == "__main__":
    main()
