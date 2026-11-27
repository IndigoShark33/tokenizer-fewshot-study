"""Tests for common.py, on toy tokenizers that copy each family's pre-tokenization:
  - bloom_like: byte-level BPE, the space attached to the following word (BLOOM)
  - xglm_like: SentencePiece-style Metaspace pieces with a </s> prefix (XGLM)
No downloads. Run from repo/:  python -m pytest -q tests
The two score() tests need torch (Colab has it) and build tiny random models; elsewhere they skip."""
import sys, warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, Regex
from transformers import PreTrainedTokenizerFast
from common import SEAM, copa_text, encode_pair, score, tokens_per_word

CORPUS = ["the man broke his toe because he dropped a hammer on his foot",
          "the woman smiled so the child laughed and the dog ran home",
          "she opened the door because the bell rang twice in the morning",
          "today she smiled and he laughed",
          "他 打破 了 脚趾 因为 他 把 锤子 掉 在 脚 上", "今天天气很好我们去公园散步",
          "li te kase zòtèy li paske li te lage yon mato sou pye li"] * 50

# (prompt, option): the first four prompts end in a space, as every prompt in sweep.py does
CASES = [("the man broke his toe because ", "he dropped a hammer"),
         ("the woman smiled so ", "the child laughed"),
         ("他 打破 了 脚趾 because ", "他把锤子掉在脚上"),
         ("li te kase zòtèy li because ", "li te lage yon mato"),
         ("today she smiled", " and he laughed")]


def bloom_like():
    t = Tokenizer(models.BPE(unk_token=None))
    t.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r" ?[^\s.,!?…。，、।۔،]+"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(CORPUS, trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["</s>"]))
    return PreTrainedTokenizerFast(tokenizer_object=t, eos_token="</s>")


def xglm_like():
    t = Tokenizer(models.Unigram())
    t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    t.train_from_iterator(CORPUS, trainers.UnigramTrainer(
        vocab_size=400, special_tokens=["</s>", "<unk>"], unk_token="<unk>"))
    sid = t.token_to_id("</s>")
    t.post_processor = processors.TemplateProcessing(single="</s> $A", special_tokens=[("</s>", sid)])
    return PreTrainedTokenizerFast(tokenizer_object=t, eos_token="</s>", unk_token="<unk>")


TOKS = {"bloom_like": bloom_like(), "xglm_like": xglm_like()}


@pytest.fixture(params=list(TOKS))
def tok(request):
    return TOKS[request.param]


def test_pair_is_the_joint_encoding(tok):
    for p, o in CASES:
        ctx, cont = encode_pair(tok, p, o)
        assert ctx + cont == tok(p + o).input_ids
        assert ctx == tok(p.rstrip()).input_ids      # no merge across the seam on these cases


def test_v1_boundary_dropped_the_first_option_token(tok):
    for p, o in CASES:
        v1 = len(tok(p + o).input_ids) - len(tok(p).input_ids)
        _, cont = encode_pair(tok, p, o)
        if p != p.rstrip():
            assert len(cont) >= v1 + 1, (p, o)      # v1 masked at least the option's first token
        else:
            assert len(cont) == v1, (p, o)          # no trailing space: nothing changes


def test_seam_counter_counts_pairs():
    before = SEAM["pairs"]
    encode_pair(TOKS["bloom_like"], *CASES[0])
    assert SEAM["pairs"] == before + 1


def test_tokens_per_word_covers_every_token(tok):
    for s in CORPUS[:4] + [CORPUS[6]]:
        counts = tokens_per_word(tok, s)
        assert len(counts) == len(s.split())
        assert sum(counts) == len(tok(s, add_special_tokens=False).input_ids)
        assert min(counts) >= 1


def test_bare_words_differ_from_context_only_for_byte_level():
    def bare(t, s):
        return [len(t(w, add_special_tokens=False).input_ids) for w in s.split()]
    s = CORPUS[0]
    assert bare(TOKS["xglm_like"], s) == tokens_per_word(TOKS["xglm_like"], s)
    assert bare(TOKS["bloom_like"], s) != tokens_per_word(TOKS["bloom_like"], s)


def test_copa_text_strips_the_scripts_final_mark():
    assert copa_text("The man broke his toe.", "cause", "He dropped a hammer on his foot.") == \
        ("The man broke his toe because ", "he dropped a hammer on his foot.")
    assert copa_text("我的身体投下了阴影。", "cause", "太阳正在升起。") == ("我的身体投下了阴影 because ", "太阳正在升起。")
    assert copa_text("राम घर गया।", "effect", "वह सो गया।")[0] == "राम घर गया so "


def _manual_logprob(mdl, ctx, cont):
    import torch
    ids = torch.tensor([ctx + cont])
    with torch.no_grad():
        logp = torch.log_softmax(mdl(input_ids=ids).logits.float(), -1)[0]
    return sum(logp[len(ctx) + i - 1, t].item() for i, t in enumerate(cont))


def _check_score(tok, mdl):
    for p, o in CASES:
        ctx, cont = encode_pair(tok, p, o)
        manual = _manual_logprob(mdl, ctx, cont)
        assert abs(score(tok, mdl, p, o, "sum") - manual) < 1e-3
        assert abs(score(tok, mdl, p, o, "mean") - manual / len(cont)) < 1e-4


def test_score_bloom_tiny_random_model():
    torch = pytest.importorskip("torch")
    from transformers import BloomConfig, BloomForCausalLM
    torch.manual_seed(0)
    tok = TOKS["bloom_like"]
    mdl = BloomForCausalLM(BloomConfig(vocab_size=len(tok), hidden_size=32, n_layer=2, n_head=2)).eval()
    _check_score(tok, mdl)


def test_score_xglm_tiny_random_model():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    if not hasattr(tf, "XGLMForCausalLM"):
        pytest.skip("this transformers version has no XGLM")
    torch.manual_seed(0)
    tok = TOKS["xglm_like"]
    v = len(tok)
    cfg = tf.XGLMConfig(vocab_size=v + 1, d_model=32, num_layers=2, attention_heads=2, ffn_dim=64,
                        max_position_embeddings=256, pad_token_id=v)   # pad id unused by the toy vocab
    mdl = tf.XGLMForCausalLM(cfg).eval()
    _check_score(tok, mdl)
