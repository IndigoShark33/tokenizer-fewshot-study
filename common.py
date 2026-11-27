"""Helpers shared by sweep.py, sum_robustness.py, metrics_flores.py and seam_check.py.

v2 (Nov 2026) changes, each tested in tests/test_common.py:
  - encode_pair: token-level prompt/option split. v1 masked the option's first token.
  - tokens_per_word: word counts taken inside the sentence. v1 tokenized bare words.
  - copa_text: strips the sentence-final mark of the premise's script, not only the ASCII period.
  - score (26 Nov): the output layer runs only at the positions that predict option tokens, so a long
    prompt no longer builds a (length x vocabulary) logit matrix. Same log-probabilities up to float
    rounding.
v5 (Dec 2026), for belebele.py, also tested there:
  - score_options: score() for several options in one padded batch; returns sums and token counts.
  - score_letters: the log-probability of each single-token label (" A" ... " D") from one pass.
"""
from collections import Counter

# Sentence-final marks stripped from a COPA premise before the English connector.
# v1 stripped only ".", so Chinese premises kept "。" before " because".
FINAL_PUNCT = ".。।။"

# Running count of prompt/option pairs, and of pairs whose prompt tokens changed when the
# option was appended (a merge across the seam). Printed by sweep.py and seam_check.py.
SEAM = Counter()


def copa_text(premise, question, choice):
    conn = " because " if question == "cause" else " so "
    return premise.rstrip().rstrip(FINAL_PUNCT) + conn, choice[0].lower() + choice[1:]


def encode_pair(tok, prompt, option):
    """Split prompt + option at the token level, as the reference harness does
    (lm-evaluation-harness, `_encode_pair`): trailing whitespace moves from the prompt to the
    option, the pair is encoded once, and the split falls at the prompt's own token length.
    The option's first token, with its leading space, is therefore scored.

    v1 encoded the prompt with its trailing space. That made the prompt one token too long,
    so the loss mask covered the option's first token."""
    n_ws = len(prompt) - len(prompt.rstrip())
    if n_ws:
        option, prompt = prompt[-n_ws:] + option, prompt[:-n_ws]
    ctx = tok(prompt).input_ids
    full = tok(prompt + option).input_ids
    SEAM["pairs"] += 1
    if full[:len(ctx)] != ctx:
        SEAM["prompt_changed"] += 1
    return full[:len(ctx)], full[len(ctx):]


def score(tok, mdl, prompt, option, reduce="mean"):
    """Log-probability of the option given the prompt. Only the positions that predict option tokens
    go through the output layer: float32 logits over BLOOM's 250,880-token vocabulary take about 1 MB
    per position, and Burmese 4-shot prompts run to thousands of tokens.
    reduce="mean": average over option tokens (main analysis). reduce="sum": total (Appendix B)."""
    import torch
    ctx, cont = encode_pair(tok, prompt, option)
    ids = torch.tensor([ctx + cont], device=mdl.device)
    with torch.no_grad():   # logits only where an option token is predicted (memory fix, 26 Nov)
        h = mdl.base_model(input_ids=ids, use_cache=False).last_hidden_state[0, len(ctx) - 1:-1]
        logits = mdl.get_output_embeddings()(h).float()
        target = torch.tensor(cont, device=logits.device)
        mean_logprob = torch.log_softmax(logits, -1).gather(1, target[:, None]).mean().item()
    return mean_logprob if reduce == "mean" else mean_logprob * len(cont)


def tokens_per_word(tok, sent):
    """Number of tokens in each whitespace-delimited word, counted inside the sentence: the
    forms the model actually sees. Needs a fast tokenizer (offset mapping).

    v1 tokenized each word on its own. For byte-level BPE (BLOOM) that measures `word`
    while the model sees the space-led `Ġword`, so fertility and STRR were off."""
    offs = tok(sent, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    bounds, pos = [], 0
    for w in sent.split():
        s = sent.index(w, pos)
        bounds.append((s, s + len(w)))
        pos = s + len(w)
    counts, j = [0] * len(bounds), 0
    for a, _ in offs:
        while j < len(bounds) and bounds[j][1] <= a:
            j += 1          # a space-led token belongs to the next word
        if j < len(bounds):
            counts[j] += 1
    return counts


def score_options(tok, mdl, prompt, options, max_len=None):
    """score(..., reduce="sum") for every option, in one right-padded batch, plus each option's
    token count, so mean and sum scoring can both be computed afterwards. Each pair is encoded with
    encode_pair, exactly as score() does. If a pair is longer than max_len, the context is cut from
    the left (the option always stays whole) and the pair is counted in SEAM["truncated"].
    Right padding leaves every real position as it is under causal attention."""
    import torch
    pairs = []
    for o in options:
        ctx, cont = encode_pair(tok, prompt, o)
        if max_len and len(ctx) + len(cont) > max_len:
            ctx = ctx[len(ctx) + len(cont) - max_len:]
            SEAM["truncated"] += 1
        pairs.append((ctx, cont))
    width = max(len(c) + len(t) for c, t in pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(pairs), width), pad, dtype=torch.long)
    att = torch.zeros((len(pairs), width), dtype=torch.long)
    for i, (c, t) in enumerate(pairs):
        ids[i, :len(c) + len(t)] = torch.tensor(c + t)
        att[i, :len(c) + len(t)] = 1
    out = []
    with torch.no_grad():
        h = mdl.base_model(input_ids=ids.to(mdl.device), attention_mask=att.to(mdl.device),
                           use_cache=False).last_hidden_state
        head = mdl.get_output_embeddings()
        for i, (c, t) in enumerate(pairs):
            logits = head(h[i, len(c) - 1:len(c) + len(t) - 1]).float()
            target = torch.tensor(t, device=logits.device)
            lp = torch.log_softmax(logits, -1).gather(1, target[:, None]).sum().item()
            out.append((lp, len(t)))
    return out


def score_letters(tok, mdl, prompt, letters=(" A", " B", " C", " D"), max_len=None):
    """Log-probability of each label after the prompt, from a single forward pass. Every label must
    encode as one token appended to the prompt's own tokens (no merge across the seam); otherwise
    this falls back to score(..., "sum") per label, which handles any tokenization."""
    import torch
    pairs = [encode_pair(tok, prompt, l) for l in letters]
    ctx = pairs[0][0]
    if all(c == ctx and len(t) == 1 for c, t in pairs):
        if max_len and len(ctx) + 1 > max_len:
            ctx = ctx[len(ctx) + 1 - max_len:]
            SEAM["truncated"] += 1
        with torch.no_grad():
            h = mdl.base_model(input_ids=torch.tensor([ctx], device=mdl.device),
                               use_cache=False).last_hidden_state[0, -1:]
            lp = torch.log_softmax(mdl.get_output_embeddings()(h).float(), -1)[0]
        return [lp[t[0]].item() for _, t in pairs]
    SEAM["letter_fallback"] += 1
    return [score(tok, mdl, prompt, l, "sum") for l in letters]
