"""Write tokenizer.bin for an existing data dir (the prepare_* scripts now do this
automatically; this stays for upgrading dirs made before that change).

  python tools/export_tokenizer.py --dir data/alpaca
  python tools/export_tokenizer.py --vocab_size 512        # -> data/v512
"""
import argparse
import os
import sys

# import the shared writer from the tinylm/ root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokbin import write_tokenizer_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--data_dir", type=str, default=None,
                    help="defaults to ../data relative to this file")
    ap.add_argument("--dir", type=str, default=None,
                    help="explicit data folder (overrides --vocab_size/--data_dir)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.dir:
        vdir = os.path.abspath(args.dir)
    else:
        data_dir = args.data_dir or os.path.join(here, "..", "data")
        vdir = os.path.join(data_dir, f"v{args.vocab_size}")
    write_tokenizer_bin(vdir)


if __name__ == "__main__":
    main()
