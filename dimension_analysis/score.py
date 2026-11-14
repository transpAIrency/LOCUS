"""Score LOCUS v1.1 sections on the four dimension axes with the distilled ModernBERT regressors (Modal).

Volume layout (locus-v11-scoring):
  /raw/LOCUS_hf_v1.1.zip         release zip
  /models/<Axis>/                model dirs (model.safetensors, head.pt, tokenizer)
  /v11/LOCUS_hf_v1.1/...         unzipped release
  /shards/shard_00000.parquet    row_id, document_id, section_index, header, content
  /out/<run>/shard_00000.parquet row_id + 4 raw scores

Usage:
  modal run score.py::prep
  modal run score.py::full
"""
import os
import time

import modal

APP = "locus-v11-dimension-scoring"
VOL = modal.Volume.from_name("locus-v11-scoring", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("locus-v11-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.7.0",
        "huggingface_hub",
        "pyarrow",
        "pandas",
        "numpy",
        "scipy",
        "tqdm",
    )
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "true"})
)
app = modal.App(APP, image=image)

# axis -> (volume path, model ID recorded in outputs, expected sha256 of model.safetensors)
MODELS = {
    "enforcement_discretion": ("/vol/models/Discretion", "hf:ChristopherVu/DiscretionBERTDistillation@fabf5a79110e931abd47c2d92bb4cd83aa815821",
                               "217538d655a220d24c8a5b8cfabc158872ebebf6edbe0521e71269404b961e8a"),
    "opacity": ("/vol/models/Opacity", "local:Opacity_Distillation(not on HF)#sha256=97b57e7a69933b8e3fc80fbf224df3af401d21fa55138d378c42a74a7aa2d141",
                "97b57e7a69933b8e3fc80fbf224df3af401d21fa55138d378c42a74a7aa2d141"),
    "paternalism": ("/vol/models/Paternalism", "hf:ChristopherVu/PaternalismBERTDistillation@b5f911e0169e2ffa1d85dce2fd3fce4842dda426",
                    "71d795e8d336930115402cc3ac81dabbeec87f4bd6d3f1b89e279b4696b3c476"),
    "problem_salience": ("/vol/models/Salience", "hf:ChristopherVu/SalienceBERTDistillation@a586946a117afb34f1c6388de7739413d3c902ab",
                         "c55deb2661ba9f34b108d949e575ce36d834214c1d7eca0d7ce5841248a9df7d"),
}
AXES = list(MODELS)
MAX_LENGTH = 512          # training max_length
SHARD_ROWS = 50_000
GPU = "L4"


def build_text(header, content, variant="plain"):
    """Model input. Training texts were '<#..> header\\ncontent'. v1.1 strips the # marks."""
    header = (header or "").strip()
    content = content or ""
    if variant == "hash2" and header:
        header = "## " + header
    return f"{header}\n{content}" if header else content


@app.function(volumes={"/vol": VOL}, cpu=4, memory=32768, timeout=3600)
def prep():
    import glob
    import zipfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    if not os.path.exists("/vol/v11/LOCUS_hf_v1.1/manifest.json"):
        with zipfile.ZipFile("/vol/raw/LOCUS_hf_v1.1.zip") as z:
            z.extractall("/vol/v11", members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    files = sorted(glob.glob("/vol/v11/LOCUS_hf_v1.1/sections/*.parquet"))
    os.makedirs("/vol/shards", exist_ok=True)
    row_id, shard, buf = 0, 0, []

    def flush(rows):
        nonlocal shard
        t = pa.Table.from_pylist(rows)
        pq.write_table(t, f"/vol/shards/shard_{shard:05d}.parquet")
        shard += 1

    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=20_000, columns=["document_id", "section_index", "header", "content"]):
            d = b.to_pydict()
            for i in range(len(d["document_id"])):
                buf.append({
                    "row_id": row_id,
                    "document_id": d["document_id"][i],
                    "section_index": d["section_index"][i],
                    "header": d["header"][i],
                    "content": d["content"][i],
                })
                row_id += 1
                if len(buf) == SHARD_ROWS:
                    flush(buf)
                    buf = []
    if buf:
        flush(buf)
    VOL.commit()
    print(f"rows={row_id} shards={shard}")
    return row_id, shard


class _Scorer:
    def load(self, dtype_name="float16"):
        import hashlib

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dtype = getattr(torch, dtype_name)
        self.models = {}
        tok_hashes = set()
        for axis, (path, _model_id, want_sha) in MODELS.items():
            h = hashlib.sha256()
            with open(os.path.join(path, "model.safetensors"), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            assert h.hexdigest() == want_sha, f"{axis}: weights sha256 mismatch"
            tok_hashes.add(hashlib.sha256(open(os.path.join(path, "tokenizer.json"), "rb").read()).hexdigest())
            bert = AutoModel.from_pretrained(path, dtype=self.dtype, attn_implementation="sdpa").cuda().eval()
            head = torch.nn.Linear(768, 1)
            head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cpu"))
            head = head.cuda().float().eval()
            self.models[axis] = (bert, head)
            if axis == "opacity":
                self.tok = AutoTokenizer.from_pretrained(path)
        assert len(tok_hashes) == 1, "tokenizers differ across axis models"

    def score_texts(self, texts, max_tokens=65_536, max_length=MAX_LENGTH):
        """Length-sorted dynamic batching. Returns {axis: list[float]} in input order + token lengths."""
        torch = self.torch
        enc = self.tok(texts, max_length=max_length, truncation=True, add_special_tokens=True)["input_ids"]
        lens = [len(x) for x in enc]
        order = sorted(range(len(texts)), key=lambda i: lens[i])
        out = {a: [0.0] * len(texts) for a in self.models}
        i = 0
        pad = self.tok.pad_token_id
        with torch.inference_mode():
            while i < len(order):
                # grow batch while padded size fits the token budget
                j = i
                while j < len(order) and (j - i + 1) * lens[order[j]] <= max_tokens:
                    j += 1
                j = max(j, i + 1)
                idx = order[i:j]
                Lb = max(lens[k] for k in idx)
                ids = torch.full((len(idx), Lb), pad, dtype=torch.long)
                mask = torch.zeros((len(idx), Lb), dtype=torch.long)
                for r, k in enumerate(idx):
                    ids[r, : lens[k]] = torch.tensor(enc[k])
                    mask[r, : lens[k]] = 1
                ids, mask = ids.cuda(non_blocking=True), mask.cuda(non_blocking=True)
                m = mask.unsqueeze(-1).float()
                for axis, (bert, head) in self.models.items():
                    h = bert(input_ids=ids, attention_mask=mask).last_hidden_state.float()
                    rep = (h * m).sum(1) / m.sum(1)          # mean pooling over non-pad tokens (as trained)
                    y = head(rep).squeeze(-1).tolist()
                    for r, k in enumerate(idx):
                        out[axis][k] = y[r]
                i = j
        return out, lens


@app.cls(gpu=GPU, volumes={"/vol": VOL, "/hf": HF_CACHE}, timeout=3600, max_containers=20, cpu=4, memory=16384)
class Scorer(_Scorer):
    @modal.enter()
    def setup(self):
        self.load("float16")
        HF_CACHE.commit()

    @modal.method()
    def score_shard(self, shard: int, run: str, variant: str = "plain"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_path = f"/vol/out/{run}/shard_{shard:05d}.parquet"
        VOL.reload()
        if os.path.exists(out_path):
            return {"shard": shard, "skipped": True}
        t0 = time.time()
        d = pq.read_table(f"/vol/shards/shard_{shard:05d}.parquet").to_pydict()
        texts = [build_text(h, c, variant) for h, c in zip(d["header"], d["content"])]
        scores, lens = self.score_texts(texts)
        cols = {"row_id": d["row_id"], "n_tokens": lens, "truncated": [l >= MAX_LENGTH for l in lens]}
        cols.update({a: scores[a] for a in AXES})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        pq.write_table(pa.table(cols), tmp)
        os.replace(tmp, out_path)                     # atomic: a crash never leaves a partial shard
        VOL.commit()
        return {"shard": shard, "rows": len(texts), "secs": time.time() - t0}


def _save_report(name, obj):
    import json
    os.makedirs("/vol/reports", exist_ok=True)
    with open(f"/vol/reports/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    VOL.commit()


@app.function(volumes={"/vol": VOL}, cpu=2, memory=8192, timeout=24 * 3600)
def run_full(run: str = "v11_hash2_512", variant: str = "hash2"):
    """Resumable: finished shards are skipped; rerun after any crash."""
    VOL.reload()
    n_shards = len([f for f in os.listdir("/vol/shards") if f.endswith(".parquet")])
    log, t0 = [], time.time()
    for r in Scorer().score_shard.map(range(n_shards), kwargs={"run": run, "variant": variant},
                                      order_outputs=False, return_exceptions=True):
        r = r if isinstance(r, dict) else {"error": repr(r)}
        log.append(r)
        print(f"[{len(log)}/{n_shards}] {r} elapsed={time.time()-t0:.0f}s", flush=True)
    _save_report(f"full_run_{run}", {"n_shards": n_shards, "wall_secs": time.time() - t0, "shards": log})
    return {"done": sum("rows" in x or x.get("skipped") for x in log), "errors": [x for x in log if "error" in x]}


@app.local_entrypoint()
def full(run: str = "v11_hash2_512", variant: str = "hash2"):
    print(run_full.remote(run, variant))
