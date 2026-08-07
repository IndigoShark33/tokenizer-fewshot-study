"""Robustness check: rerun 0-shot for both 7B models with SUMMED (not averaged) log-probability scoring.
Averaging normalizes by candidate token length, which is what fertility measures; this checks whether
the correlation pattern depends on that normalization. Deterministic (no seeds). Output: results/{tag}_sum0.json."""
import torch, json, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
login(os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE"))

SAVE = "results"; os.makedirs(SAVE, exist_ok=True)
XCOPA_LANGS = ["et","ht","id","it","sw","ta","th","tr","vi","zh","qu"]
XSC_LANGS = ["ru","zh","es","ar","hi","id","te","sw","eu","my"]
MODELS = ["facebook/xglm-7.5B","bigscience/bloom-7b1"]
XSC_SUBSAMPLE = 500

def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto")
    mdl.eval(); return tok, mdl

def score_sum(tok, mdl, prompt, option):
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + option, return_tensors="pt").to(mdl.device)
    labels = full.input_ids.clone(); labels[0, :p_ids.shape[1]] = -100
    n_opt = full.input_ids.shape[1] - p_ids.shape[1]
    with torch.no_grad():
        return -mdl(**full, labels=labels).loss.item() * n_opt

def copa_text(premise, question, choice):
    conn = " because " if question == "cause" else " so "
    return premise.rstrip(".") + conn, choice[0].lower() + choice[1:]

def run_xcopa(tok, mdl, lang):
    test = list(load_dataset("cambridgeltl/xcopa", lang)["test"]); c = 0
    for it in test:
        p, c1 = copa_text(it["premise"], it["question"], it["choice1"])
        _, c2 = copa_text(it["premise"], it["question"], it["choice2"])
        c += int((0 if score_sum(tok,mdl,p,c1) > score_sum(tok,mdl,p,c2) else 1) == it["label"])
    return round(c/len(test)*100, 2)

def run_encopa(tok, mdl):
    test = list(load_dataset("aps/super_glue", "copa")["validation"]); c = 0
    for it in test:
        p, c1 = copa_text(it["premise"], it["question"], it["choice1"])
        _, c2 = copa_text(it["premise"], it["question"], it["choice2"])
        c += int((0 if score_sum(tok,mdl,p,c1) > score_sum(tok,mdl,p,c2) else 1) == it["label"])
    return round(c/len(test)*100, 2)

def run_xsc(tok, mdl, lang):
    test = list(load_dataset("juletxara/xstory_cloze", lang)["eval"])[:XSC_SUBSAMPLE]; c = 0
    for it in test:
        story = " ".join(it[f"input_sentence_{i}"] for i in range(1,5)) + " "
        pred = 1 if score_sum(tok,mdl,story,it["sentence_quiz1"]) > score_sum(tok,mdl,story,it["sentence_quiz2"]) else 2
        c += int(pred == it["answer_right_ending"])
    return round(c/len(test)*100, 2)

if __name__ == "__main__":
    for model_name in MODELS:
        tag = model_name.split("/")[-1]; fpath = f"{SAVE}/{tag}_sum0.json"
        done = json.load(open(fpath)) if os.path.exists(fpath) else {}
        tok, mdl = load_model(model_name)
        if "en_copa" not in done:
            done["en_copa"] = run_encopa(tok, mdl); json.dump(done, open(fpath,"w")); print(tag,"en:",done["en_copa"])
        for lang in XCOPA_LANGS:
            k = f"xcopa_{lang}"
            if k not in done:
                done[k] = run_xcopa(tok, mdl, lang); json.dump(done, open(fpath,"w")); print(tag,k,done[k])
        for lang in XSC_LANGS:
            k = f"xsc_{lang}"
            if k not in done:
                done[k] = run_xsc(tok, mdl, lang); json.dump(done, open(fpath,"w")); print(tag,k,done[k])
        del mdl; torch.cuda.empty_cache()
    print("SUM-SCORED 0-SHOT DONE")
