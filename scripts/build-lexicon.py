#!/usr/bin/env python3
"""Build or audit a sherpa-onnx TTS lexicon.

sherpa-onnx >= 2.0 dropped built-in espeak-ng, so a model can only synthesise
words that its lexicon covers. This script fills the gaps from an external
grapheme-to-phoneme source (espeak-ng or CMUdict) and validates every phone
against the model's tokens.txt, because sherpa silently discards a lexicon
entry whose phones are not in tokens.

Formats accepted here follow sherpa-onnx/csrc/lexicon.cc exactly:
  - one entry per line:  WORD<whitespace>PHONE<whitespace>PHONE...
  - the word is lowercased before lookup, and a duplicated word is ignored
    (only the first entry wins), so no per-word pronunciation variants
  - every phone must exist in tokens.txt or the whole line is dropped
  - the blank token (" ") cannot appear inside a lexicon

Usage:
  python3 scripts/build-lexicon.py --tokens model/tokens.txt \
      --words words.txt --backend espeak --voice en-us --out lexicon.txt

  python3 scripts/build-lexicon.py --tokens model/tokens.txt \
      --words words.txt --backend cmudict --cmudict cmudict-0.7b \
      --out lexicon.txt

  python3 scripts/build-lexicon.py --tokens model/tokens.txt --check lexicon.txt
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import subprocess
import sys

BLANK_TOKEN = " "
STRESS_MARKS = "\u02c8\u02cc"  # \u02c8 primary, \u02cc secondary (IPA)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# tokens / lexicon
# --------------------------------------------------------------------------- #
def read_tokens(path: str) -> tuple[set[str], bool]:
    """Return (phone set, has_blank).

    tokens.txt lines are either "<symbol> <id>" or a bare "<id>", where the bare
    form is how sherpa encodes the blank token " ".
    See symbol-table.cc:ReadTokens.
    """
    if not os.path.isfile(path):
        die(f"tokens file not found: {path}")

    phones: set[str] = set()
    has_blank = False
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1 and parts[0].isdigit():
                has_blank = True
                continue
            if len(parts) != 2 or not parts[1].isdigit():
                warn(f"{path}:{lineno}: expected '<phone> <id>', got {line!r}")
                continue
            phones.add(parts[0])

    if not phones:
        die(f"no phones parsed from {path}")
    return phones, has_blank


def read_lexicon(path: str) -> list[tuple[str, list[str]]]:
    """Parse an existing lexicon into (word, phones) pairs, keeping line order."""
    entries: list[tuple[str, list[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append((parts[0].lower(), parts[1:]))
    return entries


def greedy_split(text: str, vocab: set[str], max_len: int) -> list[str] | None:
    """Split a continuous phone string into the longest matching phone sequence."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for length in range(min(max_len, n - i), 0, -1):
            candidate = text[i : i + length]
            if candidate in vocab:
                matched = candidate
                break
        if matched is None:
            return None
        out.append(matched)
        i += len(matched)
    return out


def strip_stress_marks(ipa: str) -> str:
    return "".join(c for c in ipa if c not in STRESS_MARKS)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def espeak_ipa(word: str, voice: str, espeak_bin: str) -> str | None:
    """Run espeak-ng for a single word and return its IPA string."""
    try:
        proc = subprocess.run(
            [espeak_bin, "-q", "--ipa=3", "-v", voice, word],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = " ".join(proc.stdout.split())
    return out or None


def load_cmudict(path: str) -> dict[str, list[str]]:
    """Parse cmudict-0.7b, keeping the lowest-numbered pronunciation per word."""
    if not os.path.isfile(path):
        die(f"cmudict file not found: {path}")

    best: dict[str, tuple[int, list[str]]] = {}
    line_re = re.compile(r"^(?P<word>[^()]+?)(?:\((?P<idx>\d+)\))?\s+(?P<phones>.+)$")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            m = line_re.match(line)
            if not m:
                continue
            word = m.group("word").strip().lower()
            idx = int(m.group("idx") or 0)
            phones = m.group("phones").split()
            if phones and phones[0] in ("-", "."):
                phones = phones[1:]
            if not phones:
                continue
            cur = best.get(word)
            if cur is None or idx < cur[0]:
                best[word] = (idx, phones)
    if not best:
        die(f"no pronunciations parsed from {path}")
    return {w: v[1] for w, v in best.items()}


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def check_lexicon(tokens_path: str, lexicon_path: str) -> int:
    phones, _ = read_tokens(tokens_path)
    entries = read_lexicon(lexicon_path)
    unknown: collections.Counter[str] = collections.Counter()
    bad_lines = 0
    seen: set[str] = set()
    dupes = 0

    for word, entry_phones in entries:
        if word in seen:
            dupes += 1
        seen.add(word)
        missing = [p for p in entry_phones if p not in phones]
        if missing:
            bad_lines += 1
            unknown.update(missing)

    used = {p for _, ps in entries for p in ps if p in phones}
    unused = sorted(phones - used)

    print(f"lexicon entries : {len(entries)} ({len(seen)} unique words)")
    print(f"tokens          : {len(phones)} phones")
    print(f"entries dropped at load: {bad_lines}")
    if dupes:
        print(f"duplicate words : {dupes} (sherpa keeps only the first entry)")
    if unknown:
        print("\nphones NOT in tokens.txt (these lines are silently dropped by sherpa):")
        for phone, count in unknown.most_common(20):
            print(f"  {phone!r} x{count}")
    else:
        print("\nall phones are present in tokens.txt")

    if unused:
        sample = ", ".join(repr(p) for p in unused[:15])
        more = "" if len(unused) <= 15 else f", ... (+{len(unused) - 15})"
        print(f"\nphones in tokens.txt never used by this lexicon: {sample}{more}")
        print("If your target language needs them, the lexicon is missing words.")

    return 1 if (bad_lines or unknown) else 0


def build_lexicon(args: argparse.Namespace) -> int:
    phones, has_blank = read_tokens(args.tokens)
    max_len = max(len(p) for p in phones)

    if BLANK_TOKEN in phones:
        warn("tokens.txt contains a blank token; it cannot be used inside a lexicon")

    words: list[str] = []
    if args.words:
        if not os.path.isfile(args.words):
            die(f"words file not found: {args.words}")
        with open(args.words, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
    elif args.text:
        if not os.path.isfile(args.text):
            die(f"text file not found: {args.text}")
        with open(args.text, "r", encoding="utf-8") as f:
            words = f.read().split()
    else:
        die("either --words or --text is required (unless --check is used)")
    if not words:
        die("no words to process")

    existing: list[tuple[str, list[str]]] = []
    known: set[str] = set()
    if args.base_lexicon:
        for word, entry in read_lexicon(args.base_lexicon):
            missing = [p for p in entry if p not in phones]
            if missing:
                # Never carry a line sherpa would silently drop into the merged
                # file: leave the word unregistered so it can be regenerated.
                warn(
                    f"{args.base_lexicon}: dropping {word!r}, "
                    f"{', '.join(missing)} not in tokens.txt"
                )
                continue
            existing.append((word, entry))
            known.add(word)

    results: list[tuple[str, list[str]]] = []
    skipped: collections.Counter[str] = collections.Counter()
    skipped_detail: list[str] = []
    seen: set[str] = set(known)

    for raw in words:
        word = raw.strip().lower()
        if not word or any(ch.isspace() for ch in word):
            skipped["invalid"] += 1
            continue
        if word in seen:
            skipped["duplicate"] += 1
            continue

        if args.backend == "espeak":
            ipa = espeak_ipa(word, args.voice, args.espeak)
            if not ipa:
                skipped["espeak_failed"] += 1
                skipped_detail.append(f"{word}: espeak-ng returned nothing")
                continue
            entry = greedy_split(ipa, phones, max_len)
            if entry is None and args.strip_stress:
                entry = greedy_split(strip_stress_marks(ipa), phones, max_len)
            if entry is None:
                skipped["unmatched_ipa"] += 1
                skipped_detail.append(f"{word}: cannot segment {ipa!r} into tokens")
                continue
        else:
            pron = load_cmudict(args.cmudict).get(word)
            if pron is None:
                skipped["not_in_cmudict"] += 1
                skipped_detail.append(f"{word}: not in cmudict")
                continue
            entry = [p.rstrip("012") if args.strip_stress else p for p in pron]
            missing = [p for p in entry if p not in phones]
            if missing:
                skipped["phone_not_in_tokens"] += 1
                skipped_detail.append(
                    f"{word}: {', '.join(missing)} not in tokens.txt"
                )
                continue

        seen.add(word)
        results.append((word, entry))

    sep = "\t" if args.tab else " "

    if args.dry_run:
        print(f"[dry-run] would add {len(results)} entries")
    else:
        if args.out:
            out_path = args.out
        elif args.base_lexicon:
            out_path = args.base_lexicon
        else:
            die("either --out or --base-lexicon is required")

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for word, entry in existing:
                f.write(word + sep + " ".join(entry) + "\n")
            for word, entry in results:
                f.write(word + sep + " ".join(entry) + "\n")
        os.replace(tmp_path, out_path)
        print(f"wrote {len(existing) + len(results)} entries to {out_path}")

    print(f"added {len(results)}, kept {len(existing)}, skipped {sum(skipped.values())}")
    if skipped:
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")

    if args.report and skipped_detail:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_detail) + "\n")
        print(f"skip details written to {args.report}")

    if skipped and args.strict:
        print("error: --strict set and words were skipped", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or audit a sherpa-onnx TTS lexicon (2.0+ has no espeak-ng)."
    )
    parser.add_argument("--tokens", required=True, help="model tokens.txt")
    parser.add_argument("--words", help="one target word per line")
    parser.add_argument("--text", help="free text; split on whitespace")
    parser.add_argument("--out", help="output lexicon path")
    parser.add_argument(
        "--base-lexicon",
        help="existing lexicon to merge into; already-present words are kept as-is",
    )
    parser.add_argument("--backend", choices=["espeak", "cmudict"], default="espeak")
    parser.add_argument("--voice", default="en-us", help="espeak-ng voice (default: en-us)")
    parser.add_argument("--espeak", default="espeak-ng", help="espeak-ng binary path")
    parser.add_argument("--cmudict", default="cmudict-0.7b", help="cmudict-0.7b file")
    parser.add_argument(
        "--keep-stress",
        dest="strip_stress",
        action="store_false",
        help="keep IPA stress marks / ARPABET digits (only if tokens.txt has them); "
        "stripped by default",
    )
    parser.add_argument("--tab", action="store_true", help="use TAB instead of space")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", help="write skipped words with reasons here")
    parser.add_argument("--strict", action="store_true", help="exit 1 if anything skipped")
    parser.add_argument("--check", metavar="LEXICON", help="only audit an existing lexicon")
    args = parser.parse_args()

    if args.check:
        return check_lexicon(args.tokens, args.check)

    if args.backend == "espeak" and shutil.which(args.espeak) is None:
        die(
            f"{args.espeak} not found in PATH. Install it (brew install espeak-ng) "
            "or use --backend cmudict."
        )

    return build_lexicon(args)


if __name__ == "__main__":
    sys.exit(main())
