# Which Tokenizer Metric Predicts Few-Shot Performance Across Languages?

Code and results for the paper (Roy Seligman, Pioneer Academics NLP Research Concentration, mentored by Prof. Fatma Tarlaci, July 2026).

**summary:** Across 8 models (XGLM 564M-7.5B, BLOOM 560m-7b1), 2 benchmarks (XCOPA, XStoryCloze), and 21 language conditions with 5 prompt seeds per result, tokenization metrics (fertility, STRR, premium, vocabulary usage) significantly predict few-shot accuracy at scale — but no metric survives controlling for each language's share of training data, fertility and STRR prove empirically redundant (rho <= -0.93), the link tracks task capability, and languages absent from training stay at chance regardless of scale.

## Repository layout

```
sweep.py            # main evaluation harness: 8 models x 2 benchmarks x {0,4}-shot x 5 seeds, checkpointed
metrics_flores.py   # computes fertility / STRR / premium / vocab_use on FLORES-200; writes metrics.csv + proportions.csv
analysis.py         # seed averaging, z-scoring, Spearman + bootstrap CI + partial Spearman + permutation p,
                    # scale table, seen/unseen table, per-benchmark check, intercorrelations, external validation
sum_robustness.py   # 0-shot rerun of both 7B models with SUMMED (not averaged) log-prob scoring
figures.py          # the four paper figures
results/            # per-model result JSONs, master_seeds.csv, metrics.csv, proportions.csv
```

## Reproducing

1. `pip install torch transformers datasets scipy pandas matplotlib`
2. Set an HF token in `sweep.py` (XStoryCloze is gated).
3. `python metrics_flores.py` (CPU, ~10 min; downloads FLORES-200)
4. `python sweep.py` (GPU; resumable — every score checkpoints to `results/` as it completes)
5. `python analysis.py`
6. `python sum_robustness.py` (robustness check, 0-shot summed scoring, both 7B models)

## Reproducibility note

Experiments were run interactively in Google Colab Pro+ on a single NVIDIA A100 40GB in float16. The full sweep totaled approximately 10 GPU-hours (50.69 Colab compute units at ~5.3 units/hour), with the two 7B models making up the majority. The scripts here are consolidated equivalents of the executed notebook cells; results in `results/` are the exact files produced by the original runs. Seed-to-seed SD averaged 0.79-1.10 accuracy points per model.

## Data sources

- XCOPA: cambridgeltl/xcopa (HF) · XStoryCloze: juletxara/xstory_cloze (HF, gated) · COPA: aps/super_glue
- FLORES-200 devtest: official Meta release tarball
- Training-data shares: BLOOM Table 1 byte counts (BigScience Workshop, 2022); XGLM Table A10 token counts (Lin et al., 2022)

## License

MIT
