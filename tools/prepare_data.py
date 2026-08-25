"""Download a corpus, train a byte-level BPE tokenizer, and tokenize to .bin.

Defaults to the TinyStories *validation* split (~21 MB, ~5M tokens) which is a
fast, fully-on-laptop subset for getting an end-to-end run working. Use --full
for the ~2 GB train split once the pipeline is proven, or --input for any local
.txt file.

Output (per vocab size, so presets with different vocabs don't clash):
  data/v{vocab}/tokenizer.json
  data/v{vocab}/train.bin   (uint16 token stream)
  data/v{vocab}/val.bin
  data/v{vocab}/meta.json
"""
import argparse
import json
import os
import sys

import numpy as np

EOT = "<|endoftext|>"
URLS = {
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
    "full":  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
}


def download(url, dest):
    import requests
    from tqdm import tqdm
    if os.path.exists(dest):
        print(f"[data] using cached {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    print(f"[data] downloading {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def read_stories(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    stories = [s.strip() for s in text.split(EOT)]
    return [s for s in stories if s]


def train_tokenizer(stories, vocab_size, out_path, max_train_chars):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def corpus_iter():
        used = 0
        for s in stories:
            used += len(s)
            if used > max_train_chars:
                break
            yield s

    print(f"[tok] training BPE vocab={vocab_size} on up to {max_train_chars/1e6:.0f}M chars")
    tok.train_from_iterator(corpus_iter(), trainer)
    tok.save(out_path)
    print(f"[tok] saved {out_path} (actual vocab={tok.get_vocab_size()})")
    return tok


def encode_to_bin(stories, tok, out_dir, val_frac=0.1):
    eot_id = tok.token_to_id(EOT)
    print(f"[enc] encoding {len(stories)} stories (eot_id={eot_id})")
    ids = []
    encs = tok.encode_batch(stories)
    for e in encs:
        ids.extend(e.ids)
        ids.append(eot_id)
    arr = np.array(ids, dtype=np.uint16)
    n_val = int(len(arr) * val_frac)
    val, train = arr[:n_val], arr[n_val:]
    train.tofile(os.path.join(out_dir, "train.bin"))
    val.tofile(os.path.join(out_dir, "val.bin"))
    print(f"[enc] total {len(arr):,} tokens -> train {len(train):,} / val {len(val):,}")
    return len(arr), eot_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--source", choices=["valid", "full"], default="valid")
    ap.add_argument("--input", type=str, default=None, help="local .txt instead of download")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--max_train_chars", type=int, default=40_000_000)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(here, args.data_dir, "raw")
    out_dir = os.path.join(here, args.data_dir, f"v{args.vocab_size}")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if args.input:
        src = args.input
    else:
        src = download(URLS[args.source], os.path.join(raw_dir, f"TinyStories-{args.source}.txt"))

    stories = read_stories(src)
    if not stories:
        sys.exit("no stories parsed from source")
    print(f"[data] {len(stories):,} stories")

    tok = train_tokenizer(stories, args.vocab_size,
                          os.path.join(out_dir, "tokenizer.json"),
                          args.max_train_chars)
    total, eot_id = encode_to_bin(stories, tok, out_dir)

    meta = dict(vocab_size=tok.get_vocab_size(), eot_id=eot_id,
                total_tokens=total, source=args.source if not args.input else args.input)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # plain continuation model (no instruction template)
    with open(os.path.join(out_dir, "template.txt"), "w") as f:
        f.write("raw")
    # emit the C-side tokenizer.bin too, so prep is a single step
    from tokbin import write_tokenizer_bin
    write_tokenizer_bin(out_dir)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
