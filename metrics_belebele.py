"""Tokenization metrics, byte premium and training-data shares for every Belebele language.

Same definitions as metrics_flores.py, on the same FLORES-200 devtest sentences:
  premium    content tokens for the language / content tokens for the same English sentences
  fertility  tokens per whitespace word, counted inside the sentence (common.tokens_per_word)
  strr       share of words kept as one token
  vocab_use  distinct token types used
Fertility and STRR are left empty for scripts written without spaces (belebele_langs.has_words).
Writes {RESULTS_DIR}/metrics.csv, byte_premium.csv and proportions.csv. Each family shares one
tokenizer across its four sizes, so the 7B tokenizer stands for the family, as in Paper 1.

proportions.csv columns:
  share          XGLM: corpus tokens (millions); BLOOM: bytes. What Paper 1 used.
  share_sampled  XGLM only: the sampling ratio with low-resource upsampling (model card).
  share_bp       BLOOM only: bytes divided by the byte premium (English-byte equivalents)."""
import os, tarfile, urllib.request
import pandas as pd
from transformers import AutoTokenizer
from common import tokens_per_word
from belebele_langs import XGLM, BLOOM, EVAL, has_words

SAVE = os.environ.get("RESULTS_DIR", "results_belebele"); os.makedirs(SAVE, exist_ok=True)
FDIR = "flores200_dataset"
if not os.path.exists(FDIR):
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", "flores.tar.gz")
    with tarfile.open("flores.tar.gz") as t: t.extractall(".")

codes = [c for c in EVAL if os.path.exists(f"{FDIR}/devtest/{c}.devtest")]
missing = sorted(set(EVAL) - set(codes))
if missing: print("Not in FLORES-200 devtest, skipped:", " ".join(missing))
sents = {c: [x.strip() for x in open(f"{FDIR}/devtest/{c}.devtest", encoding="utf-8") if x.strip()]
         for c in codes}
assert len({len(v) for v in sents.values()}) == 1, "FLORES devtest files should be parallel"

FAMS = {"xglm": "facebook/xglm-7.5B", "bloom": "bigscience/bloom-7b1"}
rows = []
for fam, name in FAMS.items():
    tok = AutoTokenizer.from_pretrained(name)
    assert tok.is_fast, f"{name}: in-context word counts need a fast tokenizer (offset mapping)"
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    en_total = sum(len(enc(s)) for s in sents["eng_Latn"])
    for code in codes:
        if code == "eng_Latn": continue
        total, ids, counts = 0, set(), []
        for s in sents[code]:
            t = enc(s); total += len(t); ids.update(t)
            if has_words(code):
                counts += tokens_per_word(tok, s)
        fert = round(sum(counts) / len(counts), 4) if counts else None
        strr = round(sum(c == 1 for c in counts) / len(counts), 4) if counts else None
        rows.append(dict(family=fam, lang=code, premium=round(total / en_total, 4),
                         fertility=fert, strr=strr, vocab_use=len(ids)))
pd.DataFrame(rows).to_csv(f"{SAVE}/metrics.csv", index=False)

en_bytes = sum(len(s.encode("utf-8")) for s in sents["eng_Latn"])
bp = {c: round(sum(len(s.encode("utf-8")) for s in sents[c]) / en_bytes, 4) for c in codes if c != "eng_Latn"}
pd.DataFrame([{"lang": c, "byte_premium": v} for c, v in bp.items()]).to_csv(f"{SAVE}/byte_premium.csv", index=False)

prows = [{"family": "xglm", "lang": c, "share": tok_m, "share_sampled": ratio, "share_bp": None}
         for c, (tok_m, ratio) in XGLM.items() if c in bp]
prows += [{"family": "bloom", "lang": c, "share": b, "share_sampled": None, "share_bp": round(b / bp[c])}
          for c, b in BLOOM.items() if c in bp]
pd.DataFrame(prows).to_csv(f"{SAVE}/proportions.csv", index=False)
print(f"{len(codes)} languages; metrics.csv, byte_premium.csv and proportions.csv written to {SAVE}")
