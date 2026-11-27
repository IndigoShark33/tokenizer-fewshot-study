# Which Tokenizer Metric Predicts Few-Shot Performance Across Languages?

Code and results for the paper (Roy Seligman, Pioneer Academics NLP Research Concentration, mentored by Prof. Fatma Tarlaci, July 2026).

**Summary (v1, July 2026):** Across 8 models (XGLM 564M-7.5B, BLOOM 560m-7b1), 2 benchmarks (XCOPA, XStoryCloze), and 21 language conditions with 5 prompt seeds per result, tokenization metrics (fertility, STRR, premium, vocabulary usage) significantly predict few-shot accuracy at scale, but no metric survives controlling for each language's share of training data, fertility and STRR prove empirically redundant (rho <= -0.93), the link tracks task capability, and languages absent from training stay at chance regardless of scale.

**Version 2 (November 2026), after fixing the scoring seam and rerunning all eight models:** no tokenizer metric stays significant once training-data share is controlled and multiple tests are corrected for; XGLM's byte premium comes closest (partial rho = -0.51, permutation p = .043 at 0-shot, not significant after Bonferroni). XGLM's smallest model now shows a weak, non-significant link, and in BLOOM, data share keeps its effect against every metric. Full output: `results_v2/analysis_v2.txt`.

## Repository layout

```
common.py           # shared helpers: prompt/option split, scoring, in-context word counts, COPA template
sweep.py            # main evaluation harness: 8 models x 2 benchmarks x {0,4}-shot x 5 seeds, checkpointed
metrics_flores.py   # fertility / STRR / premium / vocab_use on FLORES-200; writes metrics.csv, proportions.csv, byte_premium.csv
analysis.py         # seed averaging, z-scoring, Spearman + bootstrap CI + partial Spearman + permutation p,
                    # scale table, seen/unseen table, per-benchmark check, intercorrelations,
                    # reverse partials, power table, byte-premium control
sum_robustness.py   # 0-shot rerun of both 7B models with SUMMED (not averaged) log-prob scoring
seam_check.py       # how often v1's prompt/option boundary dropped an option token (CPU only)
figures.py          # the paper's two figures, computed from the result files
tests/              # tests for common.py on toy tokenizers (no downloads)
results/            # v1 result files, unchanged
results_v2/         # v2 result files (written by the scripts above)
```

## Version 2 (November 2026)

What changed, and why:

- **Scoring seam** (`common.encode_pair`). v1 tokenized the prompt with its trailing space, so the loss mask also covered the option's first token. v2 moves trailing whitespace from the prompt to the option and splits the joint encoding at the prompt's own token length, as lm-evaluation-harness does. `seam_check.py` reports how often v1 dropped a token, by tokenizer family and language.
- **Word-level metrics** (`common.tokens_per_word`). Fertility and STRR count each word's tokens inside its sentence. v1 tokenized bare words, which for BLOOM's byte-level BPE measured `word` instead of the space-led `Ġword` the model sees.
- **Premium and vocab_use** count content tokens only (no BOS/EOS).
- **Premise punctuation.** Chinese "。" (and the final marks of other scripts) is stripped before the English connector, as the ASCII period always was.
- **Byte premium.** `metrics_flores.py` writes each language's byte premium and BLOOM's byte shares divided by it (`proportion_bp`); `analysis.py` reports a byte-premium control for BLOOM.
- **Analysis.** `analysis.py` adds reverse partials (does data share survive a control for each metric?) and the smallest correlation the design could detect. It now also runs on the v1 files: v1 stopped with a `KeyError` because the v1 `proportions.csv` has no `family` column.
- **Figures.** `figures.py` computes the plotted values from the result files; v1 hard-coded them.
- **Results.** `results/` keeps the v1 files exactly as produced. The two 7B files also hold 44 unseeded keys from pilot runs made before seeding was added; `analysis.py` skips them. v2 writes to `results_v2/`.

## Reproducing (v2)

1. `pip install torch transformers datasets scipy pandas matplotlib pytest`
2. `python -m pytest -q tests`
3. `python seam_check.py` (CPU; tokenizers and datasets only)
4. `python metrics_flores.py` (CPU, ~10 min; downloads FLORES-200)
5. `python sweep.py` (GPU; resumable, every score checkpoints to `results_v2/` as it completes)
6. `python sum_robustness.py` (robustness check, 0-shot summed scoring, both 7B models)
7. `python analysis.py`, then `python figures.py`

Settings (environment variables): `RESULTS_DIR` (default `results_v2`; `RESULTS_DIR=results` analyses the v1 files); `XSC_N` (default `500`, the first 500 XStoryCloze eval items; `all` uses the full split and roughly doubles GPU time); `HF_TOKEN` (optional; every model and dataset is public).

## Reproducibility note

Experiments were run interactively in Google Colab Pro+ on a single NVIDIA A100 40GB in float16. The full sweep totaled approximately 20 GPU-hours (111.47 units), with the two 7B models making up the majority. The scripts here are consolidated equivalents of the executed notebook cells; results in `results/` are the exact files produced by the original runs. Seed-to-seed SD averaged 0.79-1.10 accuracy points per model.

v2 rerun (November 2026): float16 on NVIDIA T4, L4 and A100 GPUs in Google Colab Pro+; rescoring one condition on a different GPU type moved its accuracy by 0.8 points (4 of 500 items). Seed-to-seed SD averaged 0.76-1.21 accuracy points per model. Results are in `results_v2/`, and `results_v2/analysis_v2.txt` is the output of `analysis.py`.

## Data sources

- XCOPA: cambridgeltl/xcopa (HF) · XStoryCloze: juletxara/xstory_cloze (HF, public) · COPA: aps/super_glue
- FLORES-200 devtest: official Meta release tarball
- Training-data shares: BLOOM Table 1 byte counts (BigScience Workshop, 2022); XGLM Table A10 token counts (Lin et al., 2022)

## License

MIT
