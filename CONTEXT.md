# Contexte projet — Classification d'intentions client (Deep Learning)

## Vue d'ensemble

Projet académique de **classification d'intentions client** sur des tweets de support client.
Dataset : **Twitter Customer Support** (`Projet/twcs.csv`, ~3M tweets, non labellisé).
Objectif : classer chaque tweet client dans l'une des **10 classes d'intention**.

---

## Structure du projet

```
notebooks/
  01a_clustering_v7.ipynb       → EDA + clustering KMeans (découverte des classes)
  01b_data_preparation_v8.ipynb → Pipeline complet : KMeans + FAISS + split + tokenisation
  02_model_rnn.ipynb            → RNN baseline (mean pooling, 57.6%)
  03_model_lstm.ipynb           → LSTM + BiLSTM (59.5%)
  04_model_transformer.ipynb    → Transformer from scratch (57.3%)
  05_model_distilbert.ipynb     → DistilBERT fine-tuning (63.2%) ← meilleur modèle
  06_evaluation_comparison.ipynb → Comparaison des 6 modèles
  07_model_cnn_lstm.ipynb       → Architecture hybride CNN + BiLSTM (59.6%)
  08_optimisation.ipynb         → Optimisation DistilBERT (SGD/Adam/AdamW, schedulers, grid search)

Documentation/
  01_preparation_donnees.md     → Pipeline données, décisions clustering, historique v1→v8
  02_architectures.md           → Théorie + paramètres justifiés pour chaque architecture

requirements.txt
```

---

## Pipeline de labellisation (pas de labels dans le dataset)

```
twcs.csv (3M tweets)
    ↓ 01a : KMeans clustering (cardiffnlp/twitter-roberta-base, K=11 client, K=9 company)
    ↓ 01b : KMeans direct (200K échantillon) → FAISS label propagation (K=10, seuil=0.85)
    ↓ Équilibrage N_MAX=100K/classe → suppression short_ack → 10 classes finales
    ↓ Split stratifié 70/15/15
    ↓ Tokenisation custom (vocab 15K) + textes bruts pour DistilBERT
```

---

## 10 classes client (après suppression short_ack)

| ID | Classe |
|---|---|
| 0 | general_complaint |
| 1 | food_delivery_complaint |
| 2 | non_english |
| 3 | account_issue |
| 4 | ios_device_issue |
| 5 | billing_refund |
| 6 | service_complaint |
| 7 | transport_complaint |
| 8 | tech_general |
| 9 | no_response |

---

## Résultats actuels

| Modèle | Test Accuracy | Notebook |
|---|---|---|
| **DistilBERT** | **63.2%** | 05 |
| CNN+BiLSTM | 59.6% | 07 |
| LSTM | 59.5% | 03 |
| BiLSTM | 59.5% | 03 |
| RNN | 57.6% | 02 |
| Transformer | 57.3% | 04 |

---

## Décisions techniques clés

- **Modèle d'embedding** : `cardiffnlp/twitter-roberta-base` (768d) — pré-entraîné sur 58M tweets, utilisé dans 01a ET 01b pour cohérence
- **RNN fix** : `hidden[-1]` → `output.mean(dim=1)` (12.7% → 57.6%)
- **short_ack supprimé** : cluster de courts remerciements ("Thanks!", "Done") sans signal sémantique → dégradait les métriques (precision ~0.44)
- **DistilBERT** : entraîné sur 600K samples, 3 epochs (~5h), LR=2e-5, AdamW
- **GPU** : Scaleway L4-1-24G — `ssh root@51.15.234.209`, notebooks dans `~/notebooks/`

---

## Artefacts (sur le serveur Scaleway, dans `~/data/`)

| Fichier | Contenu |
|---|---|
| `client_data.pkl` | `[(tokens_int, label)]` train/val/test — RNN/LSTM/Transformer |
| `client_text_data.pkl` | `[(text_str, label)]` train/val/test — DistilBERT |
| `vocab.pkl` | Vocabulaire 15K tokens |
| `label_maps.json` | `{0: "general_complaint", ...}` |
| `results_*.pkl` | Résultats de chaque modèle (chargés par 06) |

---

## Ce qui reste à faire

1. **API FastAPI** `/predict` + Swagger (10 pts code — obligatoire)
2. **Analyse d'erreurs** sur exemples concrets + latence d'inférence (10 pts rapport)
3. **Rapport PDF** 10-15 pages (60% de la note — le plus important)
4. README.md complet

---

## GitHub

https://github.com/Luigi2317/DeepLearning_Project
