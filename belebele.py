"""Belebele extension (v5): a third benchmark, with many more languages per model.

Belebele (Bandarkar et al., 2024) asks the same 900 four-way reading-comprehension questions, about
the same FLORES-200 passages, in every language. Every model here is run on every Belebele language
that XGLM or BLOOM was trained on (belebele_langs.EVAL, kept only where the dataset has it), so each
family is also tested on languages it never saw.

Formats (BELEBELE_FORMAT, default cloze):
  cloze    each answer is scored as the continuation of passage + question, as XCOPA and XStoryCloze
           are in sweep.py; the sum of log-probabilities and the token count are saved, so mean
           scoring (main analysis) and summed scoring (Appendix B) both come from one run.
  letters  the lm-evaluation-harness prompt: answers labelled A-D, scored by the log-probability
           of " A" ... " D" after "Answer:".
Pilot (BELEBELE_PILOT=1): both formats, the two 7B models, belebele_langs.PILOT, first 200 items.
  Decision rule, fixed before the full run: the format with the higher mean English accuracy over
  the two 7B models is used for every model and language; a tie goes to cloze. The rule looks only
  at English, so it cannot favour any cross-language pattern. The choice goes to format.txt.
Shots (BELEBELE_SHOTS, default "0"; e.g. "0,1"): Belebele has no training split, so k-shot examples
  come from other passages in the same language, five seeds when k > 0.
Prompts are in English for every language, as in sweep.py.
Checkpointed: {RESULTS_DIR}/{model}.json after every language; per-item scores in
{RESULTS_DIR}/items/{model}/{run}/{code}.json. Rerunning skips finished work.
Environment: RESULTS_DIR (default results_belebele), BELEBELE_MODELS (comma list of tags),
BELEBELE_N (items per language; default all 900), HF_TOKEN (optional; everything here is public)."""
import json, os, time
import numpy as np
from common import score_options, score_letters, SEAM
from belebele_langs import EVAL, PILOT

SAVE = os.environ.get("RESULTS_DIR", "results_belebele")
MODELS = ["facebook/xglm-564M", "bigscience/bloom-560m",
          "facebook/xglm-1.7B", "bigscience/bloom-1b7",
          "facebook/xglm-2.9B", "bigscience/bloom-3b",
          "facebook/xglm-7.5B", "bigscience/bloom-7b1"]
SEEDS = [0, 1, 2, 3, 4]
MAX_LEN = 2048          # XGLM's position table; BLOOM was trained at this length too
LETTERS = (" A", " B", " C", " D")


def cloze_prompt(item):
    """Passage, question and an empty answer slot. The trailing space moves to the answer in
    encode_pair, so each answer is scored from its first token, with its leading space."""
    return f"{item['flores_passage'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: "


def letters_prompt(item):
    """The lm-evaluation-harness `belebele` template."""
    a = [item[f"mc_answer{i}"] for i in range(1, 5)]
    return (f"P: {item['flores_passage'].strip()}\nQ: {item['question'].strip()}\n"
            f"A: {a[0]}\nB: {a[1]}\nC: {a[2]}\nD: {a[3]}\nAnswer:")


def answers(item):
    return [item[f"mc_answer{i}"].strip() for i in range(1, 5)]


def gold(item):
    return int(item["correct_answer_num"]) - 1


def shot_block(examples, fmt):
    """k solved examples, one blank line apart, in the same format as the test item."""
    out = ""
    for ex in examples:
        if fmt == "cloze":
            out += cloze_prompt(ex) + answers(ex)[gold(ex)] + "\n\n"
        else:
            out += letters_prompt(ex) + LETTERS[gold(ex)] + "\n\n"
    return out


def pick_shots(items, i, k, seed):
    """k examples for item i, never from item i's own passage (several questions share one)."""
    if k == 0:
        return []
    pool = [j for j, it in enumerate(items) if it["link"] != items[i]["link"]]
    rng = np.random.default_rng(seed * 100003 + i)      # fixed per seed and item
    return [items[pool[x]] for x in rng.choice(len(pool), k, replace=False)]


def run_language(tok, mdl, items, fmt, k, seed):
    """Accuracy for one language and its per-item scores.
    cloze rows: [sum_A, sum_B, sum_C, sum_D, ntok_A, ntok_B, ntok_C, ntok_D, gold]
    letters rows: [lp_A, lp_B, lp_C, lp_D, gold]"""
    rows, right_mean, right_sum, right = [], 0, 0, 0
    for i, it in enumerate(items):
        shots = shot_block(pick_shots(items, i, k, seed), fmt)
        g = gold(it)
        if fmt == "cloze":
            res = score_options(tok, mdl, shots + cloze_prompt(it), answers(it), max_len=MAX_LEN)
            sums, ntok = [s for s, _ in res], [n for _, n in res]
            right_mean += int(np.argmax([s / n for s, n in res]) == g)
            right_sum += int(np.argmax(sums) == g)
            rows.append([round(s, 4) for s in sums] + ntok + [g])
        else:
            lps = score_letters(tok, mdl, shots + letters_prompt(it), LETTERS, max_len=MAX_LEN)
            right += int(np.argmax(lps) == g)
            rows.append([round(x, 4) for x in lps] + [g])
    n = len(items)
    if fmt == "cloze":
        stats = {"acc_mean": round(100 * right_mean / n, 2), "acc_sum": round(100 * right_sum / n, 2), "n": n}
    else:
        stats = {"acc": round(100 * right / n, 2), "n": n}
    return stats, rows


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto")
    return tok, mdl.eval()


def available_codes():
    from datasets import get_dataset_config_names
    have = set(get_dataset_config_names("facebook/belebele"))
    missing = [c for c in EVAL if c not in have]
    if missing:
        print("Not in Belebele, skipped:", " ".join(missing))
    return [c for c in EVAL if c in have]


def load_items(code, n=None):
    from datasets import load_dataset
    items = list(load_dataset("facebook/belebele", code, split="test"))
    return items[:n] if n else items


def main():
    import torch
    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login
        login(os.environ["HF_TOKEN"])
    os.makedirs(SAVE, exist_ok=True)
    pilot = os.environ.get("BELEBELE_PILOT") == "1"
    if pilot:
        models = ["facebook/xglm-7.5B", "bigscience/bloom-7b1"]
        fmts, codes, n_items, shots = ["cloze", "letters"], PILOT, 200, [0]
        save = os.path.join(SAVE, "pilot")
    else:
        wanted = os.environ.get("BELEBELE_MODELS")
        models = [m for m in MODELS if not wanted or m.split("/")[-1] in wanted.split(",")]
        fmt_file = os.path.join(SAVE, "format.txt")
        fmt = os.environ.get("BELEBELE_FORMAT") or (open(fmt_file).read().strip()
                                                    if os.path.exists(fmt_file) else "cloze")
        fmts, codes = [fmt], available_codes()
        n_items = int(os.environ["BELEBELE_N"]) if os.environ.get("BELEBELE_N") else None
        shots = [int(x) for x in os.environ.get("BELEBELE_SHOTS", "0").split(",")]
        save = SAVE
    os.makedirs(save, exist_ok=True)
    for name in models:
        tag = name.split("/")[-1]
        fpath = os.path.join(save, f"{tag}.json")
        done = json.load(open(fpath)) if os.path.exists(fpath) else {}
        runs = [(f, k, s) for f in fmts for k in shots for s in (SEEDS if k > 0 else [0])]
        if all(len(done.get(f"{f}_{k}shot_s{s}", {})) == len(codes) for f, k, s in runs):
            continue
        tok, mdl = load_model(name)
        for f, k, s in runs:
            key = f"{f}_{k}shot_s{s}"
            done.setdefault(key, {})
            for code in codes:
                if code in done[key]:
                    continue
                t0 = time.time()
                stats, rows = run_language(tok, mdl, load_items(code, n_items), f, k, s)
                item_dir = os.path.join(save, "items", tag, key)
                os.makedirs(item_dir, exist_ok=True)
                json.dump(rows, open(os.path.join(item_dir, f"{code}.json"), "w"))
                done[key][code] = stats
                json.dump(done, open(fpath, "w"), indent=1)
                print(f"{tag} | {key} | {code}: {stats} ({time.time() - t0:.0f}s; "
                      f"truncated so far {SEAM['truncated']})", flush=True)
        del mdl
        torch.cuda.empty_cache()
    if pilot:
        decide(save)


def decide(save):
    """Apply the pilot rule and write the chosen format to {RESULTS_DIR}/format.txt."""
    eng = {"cloze": [], "letters": []}
    for tag in ["xglm-7.5B", "bloom-7b1"]:
        d = json.load(open(os.path.join(save, f"{tag}.json")))
        eng["cloze"].append(d["cloze_0shot_s0"]["eng_Latn"]["acc_mean"])
        eng["letters"].append(d["letters_0shot_s0"]["eng_Latn"]["acc"])
    c, l = np.mean(eng["cloze"]), np.mean(eng["letters"])
    choice = "letters" if l > c else "cloze"
    open(os.path.join(SAVE, "format.txt"), "w").write(choice + "\n")
    print(f"PILOT English accuracy, mean of the two 7B models: cloze {c:.1f}, letters {l:.1f} -> {choice}")


if __name__ == "__main__":
    main()
