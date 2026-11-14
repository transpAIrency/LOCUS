# Dimension analysis

Code for the four section-level dimension scores (enforcement discretion, opacity, paternalism, problem salience). A GPT-5.4-nano judge compares sampled pairs of sections on each axis, in both orderings. The judgments are fit with TrueSkill, and each section's mu, min-max scaled to 0-1, is the training target for a ModernBERT-base regressor with mean pooling (8,550 train / 450 val / 1,000 test). The regressors then score every v1.1 section at 512 tokens. Scores are z-standardized over the scored (non-Structural) rows. Opacity and paternalism are also residualized on length: `axis_resid = axis_z - (a + b * ln(L))`, with L the input length in characters and a, b fit by OLS (opacity: a = -5.0353, b = 0.7552; paternalism: a = -2.2718, b = 0.3407).

Models: [discretion](https://huggingface.co/ChristopherVu/DiscretionBERTDistillation), [paternalism](https://huggingface.co/ChristopherVu/PaternalismBERTDistillation), [salience](https://huggingface.co/ChristopherVu/SalienceBERTDistillation). Opacity model available on request.

## Pipeline

    pip install -r requirements.txt
    python sample.py --input-glob "<postprocessed jsonl glob>" --output sampled.jsonl -k 10000
    python pairwise_compare.py --input sampled.jsonl --output-dir output/opacity --axis opacity
    python rater.py --results output/opacity/results.jsonl --laws sampled.jsonl
    python distill.py --leaderboard output/opacity/leaderboard.csv --bert-model answerdotai/ModernBERT-base --pooling mean --test-size 1000 --normalize 0-1 --skip-ridge
    modal run score.py::prep
    modal run score.py::full
    modal run analyze.py::assemble
    modal run analyze.py::aggregates

The paper's run used 10,000 laws; set --n-laws and --n-pairs accordingly. The judge needs `OPENAI_API_KEY`. The scoring steps run on Modal and expect the release zip at `/raw/LOCUS_hf_v1.1.zip` and the four model directories under `/models/` on the `locus-v11-scoring` volume.
