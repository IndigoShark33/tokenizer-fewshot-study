"""Main evaluation harness. 8 models x (COPA-en + XCOPA + XStoryCloze) x {0,4}-shot x 5 seeds.
Checkpointed: every score writes to results/{model}.json immediately; rerunning skips finished work."""
import torch, json, os
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE"))

SAVE = "results"
os.makedirs(SAVE, exist_ok=True)
XCOPA_LANGS = ["et","ht","id","it","sw","ta","th","tr","vi","zh","qu"]
XSC_LANGS = ["ru","zh","es","ar","hi","id","te","sw","eu","my"]
MODELS = ["facebook/xglm-564M","bigscience/bloom-560m",
          "facebook/xglm-1.7B","bigscience/bloom-1b7",
          "facebook/xglm-2.9B","bigscience/bloom-3b",
          "facebook/xglm-7.5B","bigscience/bloom-7b1"]
SHOTS = [0, 4]
SEEDS = [0, 1, 2, 3, 4]
XSC_SUBSAMPLE = 500

def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto")
    mdl.eval()
    return tok, mdl

def score(tok, mdl, prompt, option):
    """Average log-probability of option tokens given prompt (prompt masked from loss)."""
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + option, return_tensors="pt").to(mdl.device)
    labels = full.input_ids.clone()
    labels[0, :p_ids.shape[1]] = -100
    with torch.no_grad():
        return -mdl(**full, labels=labels).loss.item()

def copa_text(premise, question, choice):
    conn = " because " if question == "cause" else " so "
    return premise.rstrip(".") + conn, choice[0].lower() + choice[1:]

def pick(pool, k, seed):
    if k == 0: return []
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.choice(len(pool), k, replace=False)]

def run_xcopa(tok, mdl, lang, k, seed):
    d = load_dataset("cambridgeltl/xcopa", lang)
    test, val = list(d["test"]), list(d["validation"])
    shots = ""
    for s in pick(val, k, seed):
        p, c = copa_text(s["premise"], s["question"], s["choice1"] if s["label"] == 0 else s["choice2"])
        shots += p + c + "\n\n"
    correct = 0
    for it in test:
        p, c1 = copa_text(it["premise"], it["question"], it["choice1"])
        _, c2 = copa_text(it["premise"], it["question"], it["choice2"])
        pred = 0 if score(tok, mdl, shots + p, c1) > score(tok, mdl, shots + p, c2) else 1
        correct += int(pred == it["label"])
    return round(correct / len(test) * 100, 2)

def run_encopa(tok, mdl, k, seed):
    d = load_dataset("aps/super_glue", "copa")
    test, pool = list(d["validation"]), list(d["train"])
    shots = ""
    for s in pick(pool, k, seed):
        p, c = copa_text(s["premise"], s["question"], s["choice1"] if s["label"] == 0 else s["choice2"])
        shots += p + c + "\n\n"
    correct = 0
    for it in test:
        p, c1 = copa_text(it["premise"], it["question"], it["choice1"])
        _, c2 = copa_text(it["premise"], it["question"], it["choice2"])
        pred = 0 if score(tok, mdl, shots + p, c1) > score(tok, mdl, shots + p, c2) else 1
        correct += int(pred == it["label"])
    return round(correct / len(test) * 100, 2)

def run_xsc(tok, mdl, lang, k, seed):
    d = load_dataset("juletxara/xstory_cloze", lang)
    test, pool = list(d["eval"])[:XSC_SUBSAMPLE], list(d["train"])
    shots = ""
    for s in pick(pool, k, seed):
        story = " ".join(s[f"input_sentence_{i}"] for i in range(1, 5))
        ans = s["sentence_quiz1"] if s["answer_right_ending"] == 1 else s["sentence_quiz2"]
        shots += story + " " + ans + "\n\n"
    correct = 0
    for it in test:
        story = " ".join(it[f"input_sentence_{i}"] for i in range(1, 5))
        pred = 1 if score(tok, mdl, shots + story + " ", it["sentence_quiz1"]) > \
                    score(tok, mdl, shots + story + " ", it["sentence_quiz2"]) else 2
        correct += int(pred == it["answer_right_ending"])
    return round(correct / len(test) * 100, 2)

def do(done, fpath, tag, key, fn):
    if key in done: return
    done[key] = fn()
    json.dump(done, open(fpath, "w"), indent=2)
    print(f"{tag} | {key}: {done[key]}")

if __name__ == "__main__":
    for model_name in MODELS:
        tag = model_name.split("/")[-1]
        fpath = f"{SAVE}/{tag}.json"
        done = json.load(open(fpath)) if os.path.exists(fpath) else {}
        tok, mdl = load_model(model_name)
        for k in SHOTS:
            for seed in (SEEDS if k > 0 else [0]):
                do(done, fpath, tag, f"en_copa_{k}shot_s{seed}",
                   lambda k=k, seed=seed: run_encopa(tok, mdl, k, seed))
        for lang in XCOPA_LANGS:
            for k in SHOTS:
                for seed in (SEEDS if k > 0 else [0]):
                    do(done, fpath, tag, f"xcopa_{lang}_{k}shot_s{seed}",
                       lambda lang=lang, k=k, seed=seed: run_xcopa(tok, mdl, lang, k, seed))
        for lang in XSC_LANGS:
            for k in SHOTS:
                for seed in (SEEDS if k > 0 else [0]):
                    do(done, fpath, tag, f"xsc_{lang}_{k}shot_s{seed}",
                       lambda lang=lang, k=k, seed=seed: run_xsc(tok, mdl, lang, k, seed))
        del mdl
        torch.cuda.empty_cache()
        print(f"=== {tag} COMPLETE ===")
