# title – *Data Pre‑processing*

This repository contains **three pipelines** that prepare all inputs needed for downstream representation‑learning experiments on phosphorylation sites.

| Notebook                      | Purpose                                                                                                                                                                       | Key Outputs                                                                               |
| ----------------------------  | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **`Node2Vec_training.ipynb`** | Builds a high‑confidence human PPI graph (STRING v12) and trains **node2vec** embeddings (grid‑search over $p,q$; 128‑dim).                                                   | `results/ppi_node2vec.emb.txt` (text), `results/ppi_node2vec.kv` (binary word2vec format) |
| **`join_with_gps.ipynb`**     | Parses the KinPred GPS5 tyrosine‑kinase score matrix; exposes a helper `get_kinpred_feat(site_grp_id)` that averages the 71‑dim family scores for a given phospho‑site group. | In‑memory feature vectors                                                                 |
| **`build_phos_table.ipynb`**  | Merges node2vec vectors with KinPred features to create a **lookup table** that maps each `SITE_GRP_ID` to a 199‑dim concatenated embedding (71 GPS5 + 128 PPI).              | `grp_id_to_embedding.json`                                                                |

---

## 1  Data sources

| File                                            | Where to download                      | Notes                                                                        |
| ----------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------- |
| `9606.protein.physical.links.full.v12.0.txt.gz` | String                                 | Human physical interactions; we keep edges with combined_score ≥ 200.        | 
| `GPS5_2020-02-26_all_matrix.csv`                | KinPred GPS5 release                   | 71 tyrosine‑kinase “Proba” scores per potential phospho‑site.                |
| `Phosphorylation_site_dataset.tsv`              | KinPred (auxiliary)                    | Maps `SITE_GRP_ID` ⟶ Ensembl protein IDs.                                    |

Place raw files under `data/` (or edit the paths in the notebooks).

---

## 2  Quick‑start

```bash
# 0) Create & activate any Python ≥3.9 environment (no Conda required)
python -m venv venv
source venv/bin/activate

# 1) Train PPI embeddings  (≈ 90 min, <4 GB RAM)
jupyter nbconvert --execute notebooks/Node2Vec_training.ipynb --to notebook

# 2) Load KinPred GPS5 features
jupyter nbconvert --execute notebooks/join_with_gps.ipynb --to notebook

# 3) Build the merged lookup table
jupyter nbconvert --execute notebooks/build_phos_table.ipynb --to notebook
```

`grp_id_to_embedding.json` is now ready for model training or analysis.

---

## 3  Repository layout

```
.
├── data/               # raw input datasets (not version‑controlled)
├── notebooks/          # the three Jupyter notebooks
├── results/            # node2vec outputs & final JSON lookup
├── README.md           # this file
└── requirements.txt    # exact package versions
```

---

## 4  Package versions tested

| Package  | Version |
| -------- | ------- |
| python   | 3.10    |
| pandas   | 2.2     |
| numpy    | 1.26    |
| networkx | 3.3     |
| node2vec | 0.4     |
| tqdm     | 4.66    |
| seaborn  | 0.13    |

> *Tip*: node2vec training speed scales with CPU cores; set the `workers` parameter in `Node2Vec_training.ipynb.ipynb` if you have more than 4 cores.

---

## 5  Reproducing the exact experiment

1. **Edge filtering** – We restrict STRING edges to `combined_score ≥ 200` to balance coverage and noise.
2. **Node2vec search** – The grid `[(1,0.5), (1,1), (4,1), (4,4)]` explores local vs. global walks. We pick the pair with the highest average cosine similarity among known interacting pairs.
3. **KinPred feature engineering** – For each `SITE_GRP_ID`, we average all rows sharing that ID to obtain a single 71‑dim vector.
4. **Embedding fusion** – Concatenate GPS5 (71) + PPI (128) ⇒ **199‑dim** final representation.
---

## 6  Citing data sources

* Szklarczyk *et al.* **STRING v12**: *Nucleic Acids Res.* 2022.
* Tupper *et al.* **KinPred**: *Cell Syst.* 2021.

---

## 7  License & authorship



---

## 8  Troubleshooting
