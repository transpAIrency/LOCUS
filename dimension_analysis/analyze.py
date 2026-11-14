"""Assemble v1.1 dimension scores (z-scores, length residuals) and state/county aggregates (Modal).

  modal run analyze.py::assemble --run v11_hash2_512
  modal run analyze.py::aggregates

Outputs land in /final/ on volume locus-v11-scoring.

Residualization (opacity and paternalism):
  L      = number of characters of the model input text (header + "\n" + content)
  axis_z = (raw - mean_raw) / sd_raw, mean/sd over non-Structural rows
  fit OLS axis_z = a + b * ln(L) on non-Structural rows
  axis_resid = axis_z - (a + b * ln(L))
"""
import json
import os

import modal

VOL = modal.Volume.from_name("locus-v11-scoring")
image = modal.Image.debian_slim(python_version="3.12").pip_install("pyarrow", "pandas", "numpy")
app = modal.App("locus-v11-dimension-analysis", image=image)

AXES = ["enforcement_discretion", "opacity", "paternalism", "problem_salience"]
MODEL_IDS = {
    "enforcement_discretion": "hf:ChristopherVu/DiscretionBERTDistillation@fabf5a79110e931abd47c2d92bb4cd83aa815821",
    "opacity": "local:Opacity_Distillation(not on HF)#sha256=97b57e7a69933b8e3fc80fbf224df3af401d21fa55138d378c42a74a7aa2d141",
    "paternalism": "hf:ChristopherVu/PaternalismBERTDistillation@b5f911e0169e2ffa1d85dce2fd3fce4842dda426",
    "problem_salience": "hf:ChristopherVu/SalienceBERTDistillation@a586946a117afb34f1c6388de7739413d3c902ab",
}
SECTIONS = "/vol/v11/LOCUS_hf_v1.1/sections"
FINAL = "/vol/final"


def _save(name, obj):
    os.makedirs(FINAL, exist_ok=True)
    with open(f"{FINAL}/{name}", "w") as f:
        json.dump(obj, f, indent=2, default=float)


def _load_meta(columns):
    import glob

    import pandas as pd
    import pyarrow.parquet as pq

    return pd.concat([pq.read_table(f, columns=columns).to_pandas() for f in sorted(glob.glob(SECTIONS + "/*.parquet"))],
                     ignore_index=True)


@app.function(volumes={"/vol": VOL}, cpu=8, memory=65536, timeout=3600)
def assemble(run: str = "v11_hash2_512"):
    import glob

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    VOL.reload()
    outs = sorted(glob.glob(f"/vol/out/{run}/shard_*.parquet"))
    sc = pd.concat([pd.read_parquet(f) for f in outs], ignore_index=True).sort_values("row_id")
    meta = _load_meta(["document_id", "section_index", "header", "content", "function_majority"])
    meta["row_id"] = np.arange(len(meta))  # prep assigned row_id in this same file/row order
    hdr = meta.header.fillna("").str.strip()
    meta["char_len"] = np.where(hdr != "", hdr.str.len() + 1, 0) + meta.content.fillna("").str.len()
    meta = meta.drop(columns=["header", "content"])
    df = meta.merge(sc, on="row_id", how="left", validate="one_to_one")
    missing = int(df[AXES].isna().any(axis=1).sum())
    df["is_structural"] = df.function_majority.eq("Structural")
    df["log_char_len"] = np.log(df.char_len.clip(lower=1))
    ins = ~df.is_structural & df[AXES].notna().all(axis=1)
    norm = {}
    for a in AXES:
        mu, sd = float(df.loc[ins, a].mean()), float(df.loc[ins, a].std())
        df[f"{a}_z"] = (df[a] - mu) / sd
        norm[a] = {"raw_mean_in_scope": mu, "raw_sd_in_scope": sd}
    resid = {}
    for a in ["opacity", "paternalism"]:
        x, y = df.loc[ins, "log_char_len"].values, df.loc[ins, f"{a}_z"].values
        b, a0 = np.polyfit(x, y, 1)
        df[f"{a}_resid"] = df[f"{a}_z"] - (a0 + b * df.log_char_len)
        r = float(np.corrcoef(x, y)[0, 1])
        resid[a] = {"intercept": float(a0), "slope_per_log_char": float(b), "pearson_r_with_log_len": r, "r2": r * r,
                    "formula": f"{a}_resid = {a}_z - ({a0:.6f} + {b:.6f} * ln(char_len))"}
    for a in AXES:
        df[f"{a}_model_id"] = pd.Categorical([MODEL_IDS[a]] * len(df))
    cols = (["document_id", "section_index"] + AXES + [f"{a}_z" for a in AXES] + ["opacity_resid", "paternalism_resid",
            "char_len", "log_char_len", "n_tokens", "truncated", "is_structural", "function_majority"]
            + [f"{a}_model_id" for a in AXES])
    out = df[cols]
    os.makedirs(FINAL, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), f"{FINAL}/locus_v1.1_dimension_scores.parquet",
                   compression="zstd")
    info = {"run": run, "rows": len(out), "shards_found": len(outs), "rows_missing_scores": missing,
            "in_scope_rows(non-Structural, scored)": int(ins.sum()), "structural_rows": int(df.is_structural.sum()),
            "truncated_rows_at_512_tokens": int(df.truncated.sum()),
            "normalization": norm, "residualization": resid, "model_ids": MODEL_IDS}
    _save("assemble_info.json", info)
    VOL.commit()
    print(json.dumps(info, indent=2, default=float))
    return info


@app.function(volumes={"/vol": VOL}, cpu=8, memory=65536, timeout=3600)
def aggregates():
    import pandas as pd

    VOL.reload()
    df = pd.read_parquet(f"{FINAL}/locus_v1.1_dimension_scores.parquet")
    meta = _load_meta(["state", "jurisdiction_type", "represented_county_fips"])
    df = pd.concat([df, meta], axis=1)
    ins = ~df.is_structural & df.opacity.notna()

    st = df[ins].groupby("state").agg(n=("opacity", "size"), **{f"{c}": (c, "mean") for c in
          ["opacity_z", "opacity_resid", "paternalism_z", "paternalism_resid", "enforcement_discretion_z", "problem_salience_z"]})
    st.round(4).to_csv(f"{FINAL}/state_aggregates.csv")

    # a document's sections count toward each county it represents
    ce = df.loc[ins, ["represented_county_fips", "opacity_resid", "paternalism_resid"]].explode("represented_county_fips")
    county = ce.groupby("represented_county_fips").agg(n=("opacity_resid", "size"), opacity_resid=("opacity_resid", "mean"),
                                                       paternalism_resid=("paternalism_resid", "mean"))
    county.round(5).to_csv(f"{FINAL}/county_aggregates.csv")

    by_type = df[ins].groupby("jurisdiction_type")[["opacity_z", "opacity_resid", "paternalism_z", "paternalism_resid"]].mean()
    _save("jurisdiction_type_means.json", by_type.round(4).to_dict())
    VOL.commit()
    return {"states": len(st), "counties": len(county)}
