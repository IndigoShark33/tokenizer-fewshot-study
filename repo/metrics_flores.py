"""Computes tokenization metrics (fertility, STRR, premium, vocab_use) on FLORES-200 devtest,
and writes training-data proportions from published documentation. Outputs results/metrics.csv, results/proportions.csv."""
import os, tarfile, urllib.request
import pandas as pd
from transformers import AutoTokenizer

SAVE = "results"; os.makedirs(SAVE, exist_ok=True)
FDIR = "flores200_dataset"
if not os.path.exists(FDIR):
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", "flores.tar.gz")
    with tarfile.open("flores.tar.gz") as t: t.extractall(".")

CODES = {"en":"eng_Latn","et":"est_Latn","ht":"hat_Latn","id":"ind_Latn","it":"ita_Latn",
 "sw":"swh_Latn","ta":"tam_Taml","th":"tha_Thai","tr":"tur_Latn","vi":"vie_Latn",
 "zh":"zho_Hans","qu":"quy_Latn","ru":"rus_Cyrl","es":"spa_Latn","ar":"arb_Arab",
 "hi":"hin_Deva","te":"tel_Telu","eu":"eus_Latn","my":"mya_Mymr"}
NO_WORDS = {"zh","th","my"}  # written without spaces: word-level metrics undefined
sents = {l: [x.strip() for x in open(f"{FDIR}/devtest/{c}.devtest", encoding="utf-8") if x.strip()]
         for l, c in CODES.items()}

FAMS = {"xglm": "facebook/xglm-7.5B", "bloom": "bigscience/bloom-7b1"}
rows = []
for fam, name in FAMS.items():
    tok = AutoTokenizer.from_pretrained(name)
    en_total = sum(len(tok(s)["input_ids"]) for s in sents["en"])
    for lang in CODES:
        if lang == "en": continue
        total = 0; ids = set(); words = 0; single = 0
        for s in sents[lang]:
            t = tok(s)["input_ids"]; total += len(t); ids.update(t)
            if lang not in NO_WORDS:
                for w in s.split():
                    words += 1
                    if len(tok(w, add_special_tokens=False)["input_ids"]) == 1: single += 1
        # fertility via per-word tokenization for consistency with STRR
        fert = strr = None
        if lang not in NO_WORDS:
            wtok = sum(len(tok(w, add_special_tokens=False)["input_ids"]) for s in sents[lang] for w in s.split())
            fert = round(wtok / words, 4); strr = round(single / words, 4)
        rows.append(dict(model=f"{fam}-7B", family=fam, lang=lang,
                         premium=round(total / en_total, 4), fertility=fert, strr=strr,
                         vocab_use=len(ids)))
pd.DataFrame(rows).to_csv(f"{SAVE}/metrics.csv", index=False)

# Training-data shares from model documentation.
# XGLM: Table A10 token counts in millions (Lin et al., 2022).
# BLOOM: Table 1 byte counts (BigScience Workshop, 2022); zh = simplified + traditional.
XGLM_TOKENS_M = {"et":3287,"ht":87,"id":15424,"it":41930,"sw":908,"ta":1477,"th":10842,
 "tr":12413,"vi":11199,"zh":132770,"qu":3,"ru":147792,"es":87303,"ar":12249,
 "hi":3448,"te":689,"eu":105,"my":101}
BLOOM_BYTES = {"id":19972325222,"ta":7989206220,"vi":43709279959,"zh":261781923042,
 "sw":236482543,"es":175098365045,"ar":74854900600,"hi":24622119985,
 "te":2993407159,"eu":2360470848}
prows = [{"model":"xglm-7.5B","family":"xglm","lang":l,"proportion":v} for l,v in XGLM_TOKENS_M.items()]
prows += [{"model":"bloom-7b1","family":"bloom","lang":l,"proportion":v} for l,v in BLOOM_BYTES.items()]
pd.DataFrame(prows).to_csv(f"{SAVE}/proportions.csv", index=False)
print("metrics.csv and proportions.csv written")
