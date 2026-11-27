"""Tests for the Belebele extension: common.score_options and common.score_letters against score(),
prompt construction, shot selection, the language table and the pilot rule. Same toy tokenizers and
tiny random models as test_common.py; no downloads. Run from the repo root: python -m pytest -q tests"""
import json, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_common import CORPUS, TOKS, _manual_logprob
from common import SEAM, encode_pair, score, score_letters, score_options
import belebele as bb
import belebele_langs as bl

PROMPTS = ["the man broke his toe because ", "the woman smiled so ", "li te kase zòtèy li because "]
OPTIONS = ["he dropped a hammer", "the child laughed", "the dog ran home", "she opened the door"]


def tiny_model(kind):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    torch.manual_seed(0)
    tok = TOKS[kind]
    if kind == "bloom_like":
        return tok, tf.BloomForCausalLM(tf.BloomConfig(vocab_size=len(tok), hidden_size=32, n_layer=2, n_head=2)).eval()
    if not hasattr(tf, "XGLMForCausalLM"):
        pytest.skip("this transformers version has no XGLM")
    v = len(tok)
    cfg = tf.XGLMConfig(vocab_size=v + 1, d_model=32, num_layers=2, attention_heads=2, ffn_dim=64,
                        max_position_embeddings=256, pad_token_id=v)
    return tok, tf.XGLMForCausalLM(cfg).eval()


@pytest.fixture(params=["bloom_like", "xglm_like"])
def tm(request):
    return tiny_model(request.param)


def test_score_options_matches_score(tm):
    tok, mdl = tm
    for p in PROMPTS:
        res = score_options(tok, mdl, p, OPTIONS)
        for (s, n), o in zip(res, OPTIONS):
            assert abs(s - score(tok, mdl, p, o, "sum")) < 1e-3
            assert n == len(encode_pair(tok, p, o)[1])


def test_score_options_left_truncation(tm):
    tok, mdl = tm
    p, o = PROMPTS[0] * 3, OPTIONS[0]
    ctx, cont = encode_pair(tok, p, o)
    max_len = len(cont) + 5
    before = SEAM["truncated"]
    (s, n), = score_options(tok, mdl, p, [o], max_len=max_len)
    assert SEAM["truncated"] == before + 1
    assert abs(s - _manual_logprob(mdl, ctx[-5:], cont)) < 1e-3


def single_token_labels(tok, prompt, words):
    out = []
    for w in words:
        ctx, cont = encode_pair(tok, prompt, " " + w)
        if len(cont) == 1 and ctx == tok(prompt).input_ids:
            out.append(" " + w)
    return out


def test_score_letters_fast_path_matches_score(tm):
    tok, mdl = tm
    prompt = "the man broke his toe\nAnswer:"
    words = sorted(set(" ".join(CORPUS[:7]).split()))
    labels = single_token_labels(tok, prompt, words)[:4]
    if len(labels) < 2:
        pytest.skip("toy vocabulary has too few single-token labels")
    before = SEAM["letter_fallback"]
    lps = score_letters(tok, mdl, prompt, tuple(labels))
    assert SEAM["letter_fallback"] == before
    for l, lp in zip(labels, lps):
        assert abs(lp - score(tok, mdl, prompt, l, "sum")) < 1e-3


def test_score_letters_falls_back_on_multi_token_labels(tm):
    tok, mdl = tm
    prompt = "the man broke his toe\nAnswer:"
    labels = (" hammer on his foot", " the child laughed")
    before = SEAM["letter_fallback"]
    lps = score_letters(tok, mdl, prompt, labels)
    assert SEAM["letter_fallback"] == before + 1
    for l, lp in zip(labels, lps):
        assert abs(lp - score(tok, mdl, prompt, l, "sum")) < 1e-6


ITEM = {"link": "p1", "flores_passage": "the man broke his toe. ", "question": "what broke? ",
        "mc_answer1": "his toe", "mc_answer2": "the door", "mc_answer3": "a hammer", "mc_answer4": "the bell",
        "correct_answer_num": "1"}


def test_cloze_prompt_leaves_the_space_to_the_answer():
    p = bb.cloze_prompt(ITEM)
    assert p == "the man broke his toe.\nQuestion: what broke?\nAnswer: "
    tok = TOKS["bloom_like"]
    _, cont = encode_pair(tok, p, bb.answers(ITEM)[0])
    assert tok.decode(cont) == " his toe"            # scored from its first token, space included


def test_letters_prompt_and_gold():
    p = bb.letters_prompt(ITEM)
    assert p.startswith("P: the man broke his toe.\nQ: what broke?\nA: his toe\nB: the door\n")
    assert p.endswith("D: the bell\nAnswer:")
    assert bb.gold(ITEM) == 0


def items(n_passages=6, per=3):
    out = []
    for p in range(n_passages):
        for q in range(per):
            it = dict(ITEM); it["link"] = f"p{p}"; it["question"] = f"question {q}?"
            it["correct_answer_num"] = str(1 + (p + q) % 4)
            out.append(it)
    return out


def test_pick_shots_avoids_the_items_passage_and_is_seeded():
    its = items()
    assert bb.pick_shots(its, 0, 0, 0) == []
    for i in range(len(its)):
        shots = bb.pick_shots(its, i, 3, seed=2)
        assert len(shots) == 3 and all(s["link"] != its[i]["link"] for s in shots)
        assert shots == bb.pick_shots(its, i, 3, seed=2)
    assert any(bb.pick_shots(its, 0, 3, 0) != bb.pick_shots(its, 0, 3, s) for s in range(1, 5))


def test_run_language_shapes(tm):
    tok, mdl = tm
    its = items(3, 2)
    stats, rows = bb.run_language(tok, mdl, its, "cloze", 0, 0)
    assert stats["n"] == len(its) and 0 <= stats["acc_mean"] <= 100 and 0 <= stats["acc_sum"] <= 100
    assert all(len(r) == 9 and r[-1] == bb.gold(it) for r, it in zip(rows, its))
    stats, rows = bb.run_language(tok, mdl, its, "letters", 1, 0)
    assert stats["n"] == len(its) and 0 <= stats["acc"] <= 100
    assert all(len(r) == 5 for r in rows)


def test_language_table():
    import re
    assert len(bl.XGLM) == 30 and len(bl.BLOOM) == 45      # BLOOM's 46 with Chinese counted once
    for code in bl.EVAL:
        assert re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", code), code
    assert all(t > 0 and 0 < r < 1 for t, r in bl.XGLM.values())
    assert all(b > 0 for b in bl.BLOOM.values())
    assert abs(sum(r for _, r in bl.XGLM.values()) - 1) < 0.01  # sampling ratios sum to one
    assert not bl.has_words("zho_Hans") and not bl.has_words("mya_Mymr") and bl.has_words("hin_Deva")
    assert bl.seen("xglm-7.5B", "hat_Latn") and not bl.seen("bloom-7b1", "hat_Latn")
    assert bl.seen("bloom-560m", "yor_Latn") and not bl.seen("xglm-564M", "yor_Latn")
    assert set(bl.PILOT) <= set(bl.XGLM) & set(bl.BLOOM)


def test_pilot_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(bb, "SAVE", str(tmp_path))
    pilot = tmp_path / "pilot"; pilot.mkdir()
    for tag, c, l in [("xglm-7.5B", 40.0, 30.0), ("bloom-7b1", 36.0, 45.0)]:
        json.dump({"cloze_0shot_s0": {"eng_Latn": {"acc_mean": c}},
                   "letters_0shot_s0": {"eng_Latn": {"acc": l}}}, open(pilot / f"{tag}.json", "w"))
    bb.decide(str(pilot))
    assert (tmp_path / "format.txt").read_text().strip() == "cloze"     # cloze 38.0, letters 37.5
    for letters_acc, want in [(46.0, "cloze"), (47.0, "letters")]:     # 38.0 v 38.0 is a tie: cloze
        json.dump({"cloze_0shot_s0": {"eng_Latn": {"acc_mean": 36.0}},
                   "letters_0shot_s0": {"eng_Latn": {"acc": letters_acc}}}, open(pilot / "bloom-7b1.json", "w"))
        bb.decide(str(pilot))
        assert (tmp_path / "format.txt").read_text().strip() == want
