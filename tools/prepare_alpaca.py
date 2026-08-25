"""Turn an Alpaca-format JSON into tokenized .bin files for tinylm (C or Python).

Reads a list of {instruction, input, output} records, formats each with a fixed
instruction template, trains a byte-level BPE, and writes:

  <out_dir>/tokenizer.json   (HF tokenizer)
  <out_dir>/train.bin        (uint16 token stream)
  <out_dir>/val.bin
  <out_dir>/meta.json

Then run  tools/export_tokenizer.py --dir <out_dir>  to make tokenizer.bin for the C trainer.

Example:
  python tools/prepare_alpaca.py --input alpaca_data_cleaned.json \\
                           --vocab_size 2048 --out_dir data/alpaca
"""
import argparse
import json
import os

import numpy as np

EOT = "<|endoftext|>"

# Standard Alpaca prompt template. The model learns this exact format, so use the
# same prefix at generation time (up to and including "### Response:\n").
TPL_IN = ("### Instruction:\n{instruction}\n\n"
          "### Input:\n{input}\n\n### Response:\n{output}")
TPL_NO = ("### Instruction:\n{instruction}\n\n### Response:\n{output}")


def format_record(r):
    instr = (r.get("instruction") or "").strip()
    inp = (r.get("input") or "").strip()
    out = (r.get("output") or "").strip()
    if inp:
        return TPL_IN.format(instruction=instr, input=inp, output=out)
    return TPL_NO.format(instruction=instr, output=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Alpaca-format JSON: a list of {instruction, input, output} records")
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--out_dir", default="data/alpaca")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--max_train_chars", type=int, default=60_000_000)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(here, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[data] loading {args.input}")
    records = json.load(open(args.input, encoding="utf-8"))
    docs = [format_record(r) for r in records]
    print(f"[data] {len(docs):,} records")

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, special_tokens=[EOT], show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def corpus_iter():
        used = 0
        for d in docs:
            used += len(d)
            if used > args.max_train_chars:
                break
            yield d

    print(f"[tok] training BPE vocab={args.vocab_size}")
    tok.train_from_iterator(corpus_iter(), trainer)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    eot_id = tok.token_to_id(EOT)
    print(f"[tok] vocab={tok.get_vocab_size()} eot={eot_id}")

    print("[enc] encoding...")
    ids = []
    for e in tok.encode_batch(docs):
        ids.extend(e.ids)
        ids.append(eot_id)
    arr = np.array(ids, dtype=np.uint16)
    n_val = int(len(arr) * args.val_frac)
    val, train = arr[:n_val], arr[n_val:]
    train.tofile(os.path.join(out_dir, "train.bin"))
    val.tofile(os.path.join(out_dir, "val.bin"))
    json.dump(dict(vocab_size=tok.get_vocab_size(), eot_id=eot_id,
                   total_tokens=len(arr), source=os.path.basename(args.input)),
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    # tells the C trainer which chat template to bake into the checkpoint
    open(os.path.join(out_dir, "template.txt"), "w").write("alpaca")
    # emit the C-side tokenizer.bin too, so prep is a single step
    from tokbin import write_tokenizer_bin
    write_tokenizer_bin(out_dir)
    print(f"[done] {out_dir}  total {len(arr):,} tok -> train {len(train):,} / val {len(val):,}")
    print(f"[next] c\\tinylm.exe train m1 {args.out_dir}")


if __name__ == "__main__":
    main()
