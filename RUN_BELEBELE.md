# Belebele runs (v5)

Belebele adds one benchmark with up to 59 languages: every language XGLM or BLOOM was trained on
(`belebele_langs.py`) that Belebele also has, each run on all eight models, so each family is also scored on languages it
never saw. Prompts stay in English, as in `sweep.py`.

Runtime: Colab Pro+, A100 GPU. About 7 A100-hours in all (roughly 40 compute units); every step is
checkpointed, so after a disconnect you rerun the same cell and it carries on.

## 1. Setup

```
!git clone -b belebele https://github.com/IndigoShark33/tokenizer-fewshot-study.git
%cd tokenizer-fewshot-study
!pip -q install datasets
!python -m pytest -q tests          # expect 26 passed
```

Keep results on Drive so a disconnect loses nothing:

```
from google.colab import drive; drive.mount("/content/drive")
import os; os.environ["RESULTS_DIR"] = "/content/drive/MyDrive/tokenizer-fewshot-study/results_belebele"
```

## 2. Pilot (about 20 minutes)

Both formats (cloze and letters), the two 7B models, six languages, the first 200 items each. The rule is fixed
in advance: the format with the higher mean English accuracy across the two 7B models is used everywhere; a
tie goes to cloze. Only English decides, so the choice cannot favour any cross-language pattern.

```
!BELEBELE_PILOT=1 python belebele.py
!cat $RESULTS_DIR/format.txt
```

## 3. Full run (about 6 hours; fine to split across sessions)

```
!python belebele.py 2>&1 | tee -a $RESULTS_DIR/run.log
```

One model at a time also works: `!BELEBELE_MODELS=xglm-7.5B python belebele.py`.
The log lists any languages Belebele does not have, and counts truncated prompts (expected: none at 0-shot).

## 4. Tokenization metrics (CPU is fine, about 5 minutes)

```
!python metrics_belebele.py
```

## 5. Analysis

```
!python analysis_belebele.py | tee $RESULTS_DIR/analysis.txt
```

It reports the harness check (English accuracy; languages above the chance band), the primary correlations for
the 7B models with permutation-tested partials and a Bonferroni line, the partials with XGLM's sampled shares and
BLOOM's byte-premium shares, reverse partials, summed scoring, the size trend, seen versus unseen, and power at
the realized n. It also writes `belebele_zdata.csv` for pooling with `results_v2/zdata.csv`.

## 6. Results into the repository

Copy the Drive folder into `results_belebele/` in this repository and commit it with the analysis output.
