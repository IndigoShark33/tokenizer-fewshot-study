"""Seam check (v2, Nov 2026). For every 0-shot prompt/option pair in the three benchmarks, compares how many
option tokens v1's boundary scored with how many the v2 boundary (common.encode_pair) scores, per tokenizer
family and language. Also counts pairs where appending the option changed the prompt's tokens.
CPU only: needs the tokenizers and datasets, not the models. Output: {RESULTS_DIR}/seam_check.csv."""
import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from common import copa_text, encode_pair, SEAM

SAVE = os.environ.get("RESULTS_DIR", "results_v2"); os.makedirs(SAVE, exist_ok=True)
TOKS = {"xglm": "facebook/xglm-564M", "bloom": "bigscience/bloom-560m"}  # one tokenizer per family, all sizes
XCOPA_LANGS = ["et","ht","id","it","sw","ta","th","tr","vi","zh","qu"]
XSC_LANGS = ["ru","zh","es","ar","hi","id","te","sw","eu","my"]

def pairs():
    for it in load_dataset("aps/super_glue", "copa")["validation"]:
        for ch in ("choice1", "choice2"):
            p, c = copa_text(it["premise"], it["question"], it[ch]); yield "encopa", "en", p, c
    for lang in XCOPA_LANGS:
        for it in load_dataset("cambridgeltl/xcopa", lang)["test"]:
            for ch in ("choice1", "choice2"):
                p, c = copa_text(it["premise"], it["question"], it[ch]); yield "xcopa", lang, p, c
    for lang in XSC_LANGS:
        for it in load_dataset("juletxara/xstory_cloze", lang)["eval"]:
            story = " ".join(it[f"input_sentence_{i}"] for i in range(1, 5)) + " "
            for q in ("sentence_quiz1", "sentence_quiz2"):
                yield "xsc", lang, story, it[q]

data = list(pairs())
rows = []
for fam, name in TOKS.items():
    tok = AutoTokenizer.from_pretrained(name)
    for bench, lang, p, o in data:
        v1 = len(tok(p + o).input_ids) - len(tok(p).input_ids)     # v1: prompt encoded with its trailing space
        before = SEAM["prompt_changed"]
        _, cont = encode_pair(tok, p, o)
        rows.append(dict(family=fam, benchmark=bench, lang=lang, v1_scored=v1, v2_scored=len(cont),
                         prompt_changed=SEAM["prompt_changed"] > before))
    ex_p, ex_o = "The man broke his toe because ", "he dropped a hammer on his foot."
    old = len(tok(ex_p).input_ids); full = tok(ex_p + ex_o).input_ids
    print(f"{fam}: v1 scores {tok.convert_ids_to_tokens(full[old:])[:3]} | "
          f"v2 scores {tok.convert_ids_to_tokens(encode_pair(tok, ex_p, ex_o)[1])[:3]}")

df = pd.DataFrame(rows)
df["dropped"] = df.v2_scored - df.v1_scored
out = df.groupby(["family", "benchmark", "lang"], as_index=False).agg(
    pairs=("dropped", "size"),
    share_v1_dropped=("dropped", lambda x: round((x > 0).mean(), 3)),
    mean_tokens_dropped=("dropped", lambda x: round(x.mean(), 3)),
    v1_scored_nothing=("v1_scored", lambda x: int((x <= 0).sum())),
    prompt_changed=("prompt_changed", "sum"))
out.to_csv(f"{SAVE}/seam_check.csv", index=False)
pd.set_option("display.width", 160)
print(out.to_string(index=False))
print("\nBy family:")
print(df.groupby("family").agg(pairs=("dropped", "size"),
                               share_v1_dropped=("dropped", lambda x: round((x > 0).mean(), 3)),
                               v1_scored_nothing=("v1_scored", lambda x: int((x <= 0).sum())),
                               prompt_changed=("prompt_changed", "sum")).to_string())
