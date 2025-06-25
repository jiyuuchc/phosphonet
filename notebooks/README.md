# Data Pre‑processing & Embedding Construction

This repository contains three Jupyter pipelines that take raw public datasets and turn them into ready‑to‑use **199‑dimensional representations of human phosphorylation sites**.  The steps below let you reproduce *every* intermediate artifact, from the cleaned PPI graph all the way to the final JSON lookup table.

---

## 0  Repository layout

```
.
├── data/               # raw input datasets (not version‑controlled)
├── notebooks/          # the three Jupyter notebooks
├── results/            # node2vec outputs & final JSON lookup
├── scripts/            # optional helper bash script (see §3.4)
├── README.md           # this file
└── requirements.txt    # exact package versions
```

---

## 1  Notebook summary

| Notebook                      | Purpose                                                                                                                                                                     | Key Outputs                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **`Node2Vec_training.ipynb`** | Builds a high‑confidence human PPI graph (STRING v12) and trains **node2vec** embeddings (grid‑search over $p,q$; 128‑dim).                                                 | `results/ppi_node2vec.emb.txt`, `results/ppi_node2vec.kv` |
| **`join_with_gps.ipynb`**     | Parses the KinPred GPS5 tyrosine‑kinase score matrix; exposes helper `get_kinpred_feat(site_grp_id)` that averages the 71‑dim family scores for a given phospho‑site group. | in‑memory feature vectors                                 |
| **`build_phos_table.ipynb`**  | Merges node2vec vectors with KinPred features to create a **lookup table** that maps each `SITE_GRP_ID` to a 199‑dim concatenated embedding (71 GPS5 + 128 PPI).            | `grp_id_to_embedding.json`                                |

---

## 2  Data sources

| File                                            | Where to download                | Notes                                                                       |
| ----------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------- |
| `9606.protein.physical.links.full.v12.0.txt.gz` | [STRING](https://string-db.org/) | Human physical interactions; we keep edges with **`combined_score ≥ 200`**. |
| `GPS5_2020-02-26_all_matrix.csv`                | KinPred GPS5 release             | 71 tyrosine‑kinase “Proba” scores per potential phospho‑site.               |
| `Phosphorylation_site_dataset.tsv`              | KinPred (auxiliary)              | Maps `SITE_GRP_ID` → Ensembl protein IDs.                                   |

Download the raw files, then place or symlink them under `data/` (or edit the paths at the top of each notebook).

---

## 3  Reproducing the full data‑preprocessing pipeline

\### 3.1  Set up a clean Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # Installs exact, tested versions
```

> **Tested with** Python 3.10; package versions are listed in §5.

\### 3.2  Fetch the raw datasets

```bash
mkdir -p data/
# 1) STRING physical links v12.0
wget https://stringdb-downloads.org/download/protein.physical.links.full.v12.0/9606.protein.physical.links.full.v12.0.txt.gz -P data/

# 2) KinPred GPS5 releases
curl -L -o data/GPS5_2020-02-26_all_matrix.csv "https://figshare.com/ndownloader/files/24125156"
# download the Phosphorylation_site_dataset from my repo
```


\### 3.3  Run the notebooks **in order**

```bash
# ≈ 90 min, <4 GB RAM in total
jupyter nbconvert --execute notebooks/Node2Vec_training.ipynb --to notebook
jupyter nbconvert --execute notebooks/join_with_gps.ipynb     --to notebook
jupyter nbconvert --execute notebooks/build_phos_table.ipynb  --to notebook
```

By default the executed notebooks are saved next to the originals with the suffix `-out.ipynb`, preserving all cell outputs for inspection.


\### 3.5  Expected artifacts

| File / folder                  | Produced by            | Description                                        |
| ------------------------------ | ---------------------- | -------------------------------------------------- |
| `results/ppi_node2vec.emb.txt` | **Node2Vec\_training** | 128‑dim text embeddings ("word2vec" format).       |
| `results/ppi_node2vec.kv`      | **Node2Vec\_training** | Binary keyed‑vectors (gensim).                     |
| `grp_id_to_embedding.json`     | **build\_phos\_table** | Final 199‑dim lookup table keyed by `SITE_GRP_ID`. |

---

## 4  Quick‑start (TL;DR)

```bash
# clone & cd
pip install -r requirements.txt
bash scripts/run_preprocessing.sh   # or follow §3.3
```

`grp_id_to_embedding.json` will appear under `results/` when finished.

---

## 5  Package versions tested

| Package  | Version |
| -------- | ------- |
| python   | 3.10    |
| pandas   | 2.2     |
| numpy    | 1.26    |
| networkx | 3.3     |
| node2vec | 0.4     |
| tqdm     | 4.66    |
| seaborn  | 0.13    |

> Node2vec training speed scales with CPU cores; tweak the `workers` parameter in `Node2Vec_training.ipynb` if you have more than 4 cores.

---

## 6  Exact experimental protocol

1. **Edge filtering** – keep STRING edges with `combined_score ≥ 200` to balance coverage vs. noise.
2. **Node2vec hyper‑search** – grid $(p,q) \in \{(1,0.5),(1,1),(4,1),(4,4)\})$.  The model with the highest mean cosine similarity across a held‑out set of known interacting pairs is selected.
3. **KinPred feature engineering** – average all KinPred rows that share the same `SITE_GRP_ID`, producing a single 71‑dim feature vector per group.
4. **Embedding fusion** – concatenate GPS5 (71) + PPI (128) ⇒ **199‑dim final representation**.

---

## 7  Citation

* Szklarczyk *et al.* **STRING v12**. *Nucleic Acids Res.* 2023.
* Tupper *et al.* **KinPred**. *Cell Syst.* 2021.

---

## 8  License

This repository is released under the MIT License (see `LICENSE`).  The raw datasets remain under their respective upstream licenses.

---

## 9  Troubleshooting

* **nbconvert fails with `MemoryError`** – decrease `walk_length` or `num_walks` in the node2vec grid.
* **No such file or directory: `...emb.txt`** – confirm you ran `Node2Vec_training.ipynb` successfully before `build_phos_table.ipynb`.

If you hit something obscure, open an issue or drop me an email.
