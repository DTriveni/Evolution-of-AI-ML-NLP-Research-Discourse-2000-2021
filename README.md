# Tracking the Evolution of AI, Machine Learning, and NLP Research Discourse (2000–2021)
### From Theory to Application

**Module:** Introduction to AI and Text Analytics (SEMTM0045 2025 TB-2)  
**Institution:** University of Bristol  
**Authors:** Gagandeep Shashi, Musa Khumalo, Thrijwal Krishnappa, Triveni Dhamdhere, Zahra Aghazada  
**Submitted:** May 2026

---

## Overview

This project examines how the language of AI, ML, and NLP research shifted over three decades using 24,677 arXiv abstracts (2000–2021). Five independent text analysis methods are applied to track three types of change: the move from theoretical to applied research framing, the evolution of how limitations are described (including the rise of ethics language), and the diffusion of technical terminology across time.

The key finding — that **2014–2016 was the primary inflection point** — is validated independently across all five methods, making it robust to the assumptions of any single approach.

---

## Research Questions

1. How did AI/ML/NLP research abstracts shift focus from theory to real-world application between 2000 and 2021?
2. Have the ways limitations are described in abstracts changed over time, and has large-scale AI deployment brought new ethical concerns into the discourse?
3. How does technical terminology spread — primarily through increased usage frequency, or through quiet shifts in meaning?
4. Do different text analysis methods agree on when and how these changes occurred?

---

## Dataset

**Source:** [`gfissore/arxiv-abstracts-2021`](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) on HuggingFace Hub  
**Original size:** 1,999,486 preprints  
**Filtered corpus:** 24,677 abstracts

**Inclusion criteria:**
- Primary category in `cs.LG`, `cs.CL`, or `cs.AI`
- Minimum 50 tokens per abstract
- Year and category metadata present

**Temporal structure:** Five-year periods (2000–04 through 2020–21), chosen because yearly bins were too sparse before 2005 and decade bins would mask the 2014–2015 transition.

> **Note:** The corpus is not a uniform sample of AI research — it shifts from predominantly `cs.CL` in early years to predominantly `cs.LG` post-2015. All analyses account for this subfield composition change.

---

## Repository Structure

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_thread1_focus_shift_lexical.ipynb
│   ├── 03_thread1_focus_shift_topics.ipynb
│   ├── 04_thread2_limitations.ipynb
│   ├── 05_thread3_terminology.ipynb
│   └── 06_pca_temporal_drift.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── report/
│   └── final_report.pdf
│
└── README.md
```

---

## Methods

| ID | Method | Tool(s) | Contribution |
|---|---|---|---|
| M1 | Topic Modelling | LDA (Gensim), NMF (scikit-learn) | Theory/application topic share over time |
| M2 | Terminology Diffusion | TF-IDF, Word2Vec, SPECTER2 | How individual terms spread and shift meaning *(Triveni)* |
| M3 | Limitation Framing | Keyword classification, Logistic Regression | Evolution of failure, ethics, hedging language |
| M4 | Representation Geometry | Sentence-BERT, PCA, UMAP | Cross-representation centroid drift |
| M5 | Lexical Keyword Analysis | TF-IDF frequency | Theory vs. application keyword prevalence baseline |

---

## M2 — Terminology Diffusion (Triveni Dhamdhere)

This component tracks how 36 individual technical terms evolved across periods using three complementary signals:

### Signals

| Signal | Method | What It Captures |
|---|---|---|
| TF-IDF delta | Sublinear TF-IDF, per-period mean frequency per 10,000 words | Surface adoption — how much more often a term appears |
| Word2Vec neighbourhood cosine | Skip-gram models (size=300, window=12, epochs=30) aligned via Procrustes transformation on 1,436 anchor terms | Contextual drift — whether the term's surrounding words changed |
| SPECTER2 discourse drift | `allenai/specter2` embeddings (batch=64, L2-normalised), bootstrap aggregated | Argumentative repositioning — whether abstracts using the term changed in discourse structure |

All three signals are scaled to [0, 1] and thresholded at 0.5 to produce a four-quadrant typology:

| Quadrant | TF-IDF Rise | Semantic Drift | Interpretation |
|---|---|---|---|
| **Diffused** | High | High | Genuinely spreading — new concept gaining adoption |
| **Semantic Shift Only** | Low | High | Quiet repositioning — meaning changed without frequency spike |
| **Actively Diffusing** | High | Low | Template replication — adopted widely but with fixed meaning |
| **Stable / Peripheral** | Low | Low | No meaningful change |

### Key Findings

- **13 of 36 terms** showed semantic shift without frequency rise — the dominant diffusion pathway is invisible to frequency-only analysis
- **`encoder`** showed the clearest convergent evidence: highest consensus score (0.63), highest SPECTER2 drift (0.0424), and near-total neighbourhood replacement (Jaccard ≈ 0.001)
- **`pre-training`** was the sole Actively Diffusing term: highest TF-IDF rise (~0.84) but near-zero SPECTER2 drift (0.0021), identifying it as template replication rather than conceptual spread — the BERT fine-tuning structure was reproduced at scale with minimal variation

### Setup Notes

- 174 multi-word compounds (e.g. `word_embeddings`, `pre_training`) merged into underscored tokens before Word2Vec training to prevent spurious decomposition
- Candidate terms: top-60 rising TF-IDF terms per period, reduced to ~48 after removing subsumed n-grams
- SPECTER2 coverage constraint: minimum 5 abstracts per (term, period) cell; 30 of 48 candidates have full coverage
- Procrustes alignment basis: 1,436 shared anchor terms (6.9% of combined vocabulary); reliability scores 0.27 and 0.49 for the two earliest periods

---

## Key Results

### Theory-to-Application Shift
All five methods independently place the crossover at **2014–2015**, with sustained application dominance through 2021. The Pearson correlation between TF-IDF and Sentence-BERT centroid distance series is **r = 0.863**, confirming the shift is a structural feature of the data rather than a methodological artefact.

### Limitation Framing
- Ethics language (`bias`, `fairness`) was near-absent before 2015 and reached ~13% of abstracts by 2020–2021
- Failure language (`fail`, `degrade`) grew from ~6% to ~24%, becoming the second most common limitation category
- Limitation framing rates are nearly identical across method-introducing and non-method abstracts (51.4% vs. 51.1%), indicating it has become a normalised feature of academic writing

### Terminology Diffusion
The dominant mode of change is **quiet semantic repositioning**, not frequency growth — a finding invisible to keyword counting alone.

---

## Preprocessing Pipeline

```
Load HuggingFace Dataset
→ Standardise IDs (YY.MM format)
→ Extract year (yy ≥ 19 → 20yy; yy < 19 → 19yy)
→ Clean text (ASCII, punctuation removal, lowercase)
→ Filter to primary category in {cs.LG, cs.CL, cs.AI}
→ Assign 5-year period
→ Tokenise + lemmatise (NLTK + WordNet)
→ Filter: ≥ 50 tokens, year and category present
→ Output: id, year, title, abstract, category, period
```

Raw abstracts are preserved separately for SentenceBERT and SPECTER2 (no lemmatisation applied) to retain sentence structure.

---

## Tools and Libraries

| Library | Purpose |
|---|---|
| HuggingFace `datasets` | Corpus loading |
| NLTK + WordNet | Tokenisation, lemmatisation |
| Gensim | LDA topic modelling, Word2Vec |
| scikit-learn | NMF, TF-IDF, logistic regression, PCA |
| `sentence-transformers` | Sentence-BERT (all-MiniLM-L6-v2), SPECTER2 |
| scipy | Procrustes alignment (`orthogonal_procrastes`) |
| UMAP-learn | UMAP projection |
| matplotlib, seaborn | Visualisation |

---

## Hypotheses and Outcomes

| Hypothesis | Outcome |
|---|---|
| H1: Topic models recover the lexical shift and reveal transitional hybrid themes | ✅ Verified — NMF recovered the paradigm shift and hybrid topics missed by keywords |
| H2: Embedding methods detect conceptual shifts earlier than TF-IDF | ✅ Confirmed on the `encoder` case; SPECTER2 coverage constraints blocked testing the most emergent terms |
| H3: Centroid drift captures fine-grained transitions; classification reflects broader separability | ⚠️ Partially confirmed — centroid drift captured the 2017 lag; classification conflated topical and framing effects |
| H4: Major temporal patterns are broadly consistent across methods | ✅ Strongly supported — five-fold convergence on 2014–2016 |

---

## Licence

This repository is submitted as group coursework for the University of Bristol MSc Data Science and AI programme and is intended for academic use only. The arXiv abstracts dataset is made available by HuggingFace under its original terms; no redistribution of raw data is permitted.
