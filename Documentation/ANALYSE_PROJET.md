# Analyse du Projet — Classification d'Intentions Client

---

## 1. Présentation du sujet

> **Objectif** : concevoir, entraîner, comparer et déployer un système de classification d'intentions basé sur le Deep Learning pour router automatiquement des messages clients vers la bonne catégorie.

| Élément | Détail |
|---|---|
| **Dataset principal** | Bitext Customer Support LLM (~27 000 paires message/intention, 27 classes, 11 catégories) |
| **Dataset alternatif** | Customer Support on Twitter (Kaggle) — bruité, nécessite labellisation |
| **Option FR** | MASSIVE Amazon (~20 000 énoncés, 60 intentions) ou Amazon Reviews Multi (fr) |
| **Architectures requises** | RNN basique, LSTM/BiLSTM, Transformer from scratch, Transformer pré-entraîné (DistilBERT/CamemBERT) |
| **Livrables** | Rapport PDF 10–15 pages + code Git + démo API |

### Grille de notation résumée

| Axe | Poids | Critères clés |
|---|---|---|
| Documentation (rapport) | **60 %** | Justification dataset, explication architectures, analyse résultats, analyse d'erreurs |
| Code & implémentation | **40 %** | 4+ architectures correctes, optimisation effective, API FastAPI fonctionnelle |

---

## 2. Décomposition des besoins

### 2.1 Préparation des données

| Besoin du projet | Concept cours | TP de référence |
|---|---|---|
| Nettoyage (casse, ponctuation, emojis) | Cours II p. 2–33 — Word embeddings, tokenisation | TP5 `VOCAB_SIZE=10000`, nettoyage du texte IMDB |
| Tokenisation + vocabulaire | Cours II — `nn.Embedding(vocab_size, embed_dim)` | TP5 : `vocab.get(w, vocab['<UNK>'])` |
| Padding, longueurs variables | Cours II — `pad_sequence`, `collate_fn` | TP5 : `pad_sequence(..., padding_value=0)` + `collate_fn` |
| Split train/val/test stratifié | Cours III — validation set pour early stopping | TP2 : `train_test_split(..., stratify=y)` |
| Analyse exploratoire (distribution, longueurs) | Cours I — diagnostic overfitting/underfitting | TP2 : distribution des 6 classes, heatmap confusion |
| Discussion déséquilibre des classes | Cours I — métriques, cross-entropy | TP2 : 6 classes déséquilibrées (112/72/61/52/49/20) |

### 2.2 Modélisation et comparaison des 4 architectures

| Architecture | Détails attendus | Cours | TP |
|---|---|---|---|
| **Baseline RNN** | Embedding + RNN + tête MLP | Cours II p. 2–33 : cellule RNN, `h(t) = tanh(W·h(t-1) + U·x(t) + b)` | TP5 : architecture complète à compléter |
| **LSTM ou BiLSTM** | Embedding + LSTM/BiLSTM + MLP | Cours II p. 34–46 : 3 portes, `c(t) = f⊙c(t-1) + i⊙c̃(t)` | TP6 : 2-layer LSTM + BiLSTM corrigé |
| **Transformer from scratch** | Embedding + PE + Transformer encoder + MLP | Cours II p. 47–78 : Q/K/V, multi-head, PE sin/cos | TP7 : architecture complète à implémenter |
| **Transformer pré-entraîné** | DistilBERT ou CamemBERT fine-tuné | Cours III p. 248 : fine-tuning, transfert de représentations | TP8 : principe ResNet pré-entraîné + fine-tuning |

### 2.3 Optimisation et régularisation

| Besoin | Concept cours | TP de référence |
|---|---|---|
| Comparaison SGD / Adam / AdamW | Cours III p. 8–11 : 7 algorithmes, Adam = moment 1 + 2 | TP8 : `SGD(lr=1e-2, momentum=0.9)` + Adam |
| Scheduler cosine / ReduceLROnPlateau / warmup | Cours III p. 12–56 : stratégie 1 (plateau) vs 2 (continu) | TP8 partie 2 : `ReduceLROnPlateau` + `CosineAnnealingLR` |
| Dropout + Weight Decay | Cours III p. 57–83 : dropout = ensemble de modèles | TP8 partie 4 : `nn.Dropout(0.5)` + `weight_decay=1e-4` |
| Early Stopping | Cours III p. 275–279 : `θ*` = best val loss | TP8 partie 3 : `copy.deepcopy(model.state_dict())` |
| Grid/random search documenté | Cours III p. 238–239 : log-uniforme, anytime | TP8 bonus : grid 3×3 + random 9 trials |
| Courbes d'apprentissage commentées | Cours I — diagnostic overfitting | TP5/6/8 : `plot_history(histories, labels)` |

### 2.4 Évaluation

| Besoin | Concept cours | TP de référence |
|---|---|---|
| Accuracy, Précision, Rappel, F1 macro/weighted | Cours I — métriques multiclasse | TP2 : matrice de confusion + métriques |
| Matrice de confusion analysée | Cours I — confusion 5↔3 (MNIST) | TP3 : `ConfusionMatrixDisplay`, TP2 : heatmap seaborn |
| Analyse d'erreurs sur exemples concrets | Cours I — retour critique sur les erreurs | TP5 bonus : `predict_sentiment(critique, model, vocab)` |
| Mesure latence par modèle | Cours IV p. 17–19 : compromis perf/latence | TP10 : `mesure_latence(model, input_tensor, n=200)` |

### 2.5 Déploiement

| Besoin | Concept cours | TP de référence |
|---|---|---|
| Export TorchScript ou ONNX | Cours IV p. 12–16 : trace vs script, dynamic_axes | TP10 parties 2 et 3 |
| API FastAPI `/predict` texte → intention + score | Cours IV p. 20–27 : FastAPI, Pydantic, Swagger | TP10 bonus : `@app.post("/predict")` |
| Documentation Swagger | Cours IV — `/docs` auto-générée par FastAPI | TP10 : `FastAPI(title=...)` + `PredictionResponse` |
| Bonus : Dockerfile, `/health`, batch, quantization | Cours IV p. 20–35 : Docker Compose, quantization | TP10 : `@app.get("/health")`, `quantize_dynamic` |

---

## 3. Liens cours → projet (ce qu'il faut maîtriser)

### Partie I — Fondations (utilisée en filigrane)

- **Cross-Entropy Loss** : loss par défaut pour les 27 classes (`nn.CrossEntropyLoss`) — attend des logits bruts, pas un Softmax.
- **Boucle d'entraînement PyTorch** : `zero_grad() → forward → loss → backward() → step()` — utilisée dans chaque modèle.
- **Diagnostic overfitting** : courbes train/val loss à chaque époque → détecter quand val_loss remonte = early stopping.
- **Mini-batch 32/64/128** : taille recommandée pour l'entraînement (`DataLoader(shuffle=True)`).

### Partie II — RNN/LSTM/Transformer (cœur du projet)

**Ce qu'il faut implémenter explicitement :**

```
RNN vanille :  Embedding → nn.RNN → hidden[-1] → Linear(hidden_dim, 27)
LSTM :         Embedding → nn.LSTM(num_layers=2, dropout=0.3) → hidden[-1] → Linear
BiLSTM :       nn.LSTM(..., bidirectional=True) → cat(hidden[-2], hidden[-1]) → Linear(2*hidden, 27)
Transformer :  Embedding×√d + PositionalEncoding → N×TransformerBlock → mean_pool → Linear
```

**Points critiques à ne pas rater :**
- `batch_first=True` dans tous les modules séquentiels.
- Masque de padding : `padding_mask = (x == 0)` → passer au Transformer pour ne pas attendre les tokens PAD.
- `nn.Embedding(vocab_size, embed_dim, padding_idx=0)` : les PAD ne contribuent pas au gradient.
- BiLSTM : `hidden` a shape `(num_layers×2, batch, hidden_dim)` → `cat([hidden[-2], hidden[-1]], dim=1)`.
- Transformer : mise à l'échelle `embedding × math.sqrt(embed_dim)` (papier original).

**Vanishing gradient — argument pour la progression :**
- RNN : `h(t) = tanh(W^T × h)` → multiplication répétée → `0.95^50 ≈ 0.005` → mémoire courte.
- LSTM : mise à jour additive `c(t) = f⊙c(t-1) + i⊙c̃(t)` → addition ≠ multiplication → gradient préservé (*constant error carousel*).
- Transformer : chemin de longueur 1 entre deux tokens quelconques → plus de vanishing sur longues dépendances.

**Pré-entraîné (DistilBERT) :**
- Principe identique au ResNet dans TP8 : geler le backbone, entraîner seulement la tête de classification.
- Différence-clé : BERT utilise un tokenizer spécifique (WordPiece, via HuggingFace `AutoTokenizer`), pas notre vocab custom.
- Fine-tuning complet possible avec LR différencié (backbone lr_small, head lr_large) — comme TP8 partie 5.

### Partie III — Optimisation & régularisation (à appliquer sur le meilleur modèle)

**Ordre recommandé :**
1. Trouver le bon `lr` : démarrer à `1e-3` avec Adam, observer la loss à la 1ère époque.
2. Ajouter un scheduler : `ReduceLROnPlateau(patience=2, factor=0.5)` si le nombre d'epochs n'est pas fixé à l'avance.
3. Implémenter early stopping avec `copy.deepcopy(model.state_dict())`.
4. Régulariser : Dropout sur les couches fc (0.3–0.5) + `weight_decay=1e-4` dans l'optimiseur.
5. Hyperparameter search : grid search 2D (lr × dropout) ou random search log-uniforme.

**Adam vs AdamW :**
- AdamW = Adam + weight decay découplé (L2 appliqué aux poids, pas aux moments) → souvent meilleur en NLP.
- SGD avec momentum et `weight_decay` reste compétitif pour le fine-tuning de BERT.

**Initialisation :**
- Les couches `nn.Linear` et `nn.Embedding` utilisent par défaut Kaiming/He → ne pas toucher sauf besoin spécifique.
- Pour LSTM/GRU : `orthogonal_init` sur les poids récurrents peut améliorer la stabilité.

### Partie IV — Déploiement (étape finale)

**Pipeline à suivre :**
```
Notebook entraînement → state_dict sauvegardé → TorchScript (script pour LSTM/Transformer)
→ FastAPI /predict (texte brut → tokenisation → inference → intention + score)
→ Dockerfile → image Docker → démo capture écran
```

**Choix TorchScript trace vs script :**
- `torch.jit.trace` : CNN, MLP — flux fixe. **Ne pas utiliser** pour les modèles avec `if`/boucles dynamiques.
- `torch.jit.script` : **obligatoire** pour RNN, LSTM, Transformer (boucles, `if training`, masques conditionnels).

**Quantization (bonus) :**
- `quantize_dynamic` cible les `nn.Linear` → passe float32→int8 → ~4× moins de RAM.
- Les `nn.Embedding` et couches récurrentes restent en float32.

---

## 4. Code des TP directement réutilisable

### TP5 → Baseline RNN

```python
# Preprocessing (à adapter vocab_size=10000 → vocab_size du dataset)
VOCAB_SIZE = 10000
MAX_LEN    = 128  # les messages clients sont courts

# Architecture
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, output_dim)  # output_dim=27

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])  # hidden : (1, batch, hidden_dim)

EMBED_DIM  = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 27  # ← adapter au nombre d'intentions
```

### TP6 → LSTM et BiLSTM

```python
# LSTM 2 couches avec dropout (quasi-identique, adapter output_dim=27)
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(self.dropout(hidden[-1]))

# BiLSTM (ajouter bidirectional=True + adapter fc)
# self.lstm = nn.LSTM(..., bidirectional=True)
# last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 2*hidden_dim)
# self.fc = nn.Linear(hidden_dim * 2, output_dim)
```

### TP7 → Transformer from scratch

```python
# PositionalEncoding : code fourni dans le TP, réutiliser tel quel

# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, key_padding_mask=padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# Masque de padding + pooling moyen masqué
# padding_mask = (x == 0)  # (batch, seq_len) — True là où PAD
# mask_float = (~padding_mask).unsqueeze(-1).float()
# pooled = (output * mask_float).sum(1) / mask_float.sum(1)  # moyenne sur tokens réels
```

### TP8 → Optimisation du meilleur modèle

```python
import copy

# Early Stopping
best_val_loss    = float('inf')
best_weights     = None
patience_counter = 0
PATIENCE = 3

for epoch in range(MAX_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss   = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)                      # ReduceLROnPlateau

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_weights     = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break

model.load_state_dict(best_weights)   # restaurer le meilleur modèle

# Comparaison optimiseurs
optimizers = {
    'SGD':   optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4),
    'Adam':  optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
    'AdamW': optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2),
}
```

### TP10 → Export et API

```python
# TorchScript (script pour tous les modèles séquentiels)
scripted = torch.jit.script(model)
scripted.save('model_lstm_scripted.pt')

# ONNX
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    opset_version=17,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# FastAPI /predict
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Intent Classifier")
model = torch.jit.load("model_lstm_scripted.pt")
model.eval()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    intent: str
    confidence: float
    top3: list

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    tokens = preprocess(req.text)         # tokenisation + padding
    tensor = torch.tensor([tokens])
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top_val, top_idx = torch.topk(probs, k=3)
    return PredictResponse(
        intent=INTENT_LABELS[top_idx[0].item()],
        confidence=top_val[0].item(),
        top3=[(INTENT_LABELS[i.item()], v.item()) for i, v in zip(top_idx, top_val)]
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": "LSTM Intent Classifier"}
```

---

## 5. Architecture recommandée (résumé implémentation)

### Ordre de développement conseillé

```
Étape 1 — Pipeline de bout en bout (RNN simple)
  Chargement dataset → nettoyage → tokenisation → vocab → DataLoader
  → RNN (TP5) → CrossEntropyLoss → Adam → 5 epochs → accuracy sur val
  → API simpliste /predict (texte brut → intention)   ← faire tourner AVANT tout le reste

Étape 2 — LSTM + BiLSTM (TP6)
  Copier le preprocessing (même vocab, même DataLoader)
  → LSTM 2 couches (output_dim=27) → comparer accuracy avec RNN

Étape 3 — Transformer from scratch (TP7)
  PositionalEncoding → TransformerBlock → masque padding → mean pooling

Étape 4 — Optimisation du meilleur modèle (TP8)
  ReduceLROnPlateau + Early Stopping + Dropout + AdamW
  → Comparer SGD / Adam / AdamW (courbes d'apprentissage)
  → Recherche random search sur lr × dropout

Étape 5 — DistilBERT fine-tuné (HuggingFace)
  AutoTokenizer + AutoModelForSequenceClassification
  → Geler backbone → entraîner tête → fine-tuning progressif si temps

Étape 6 — Évaluation comparative
  Accuracy + F1 macro/weighted + matrice de confusion + latence
  → Tableau comparatif des 4+ modèles

Étape 7 — Déploiement (TP10)
  TorchScript (script) du meilleur modèle → FastAPI /predict + /health
  → Dockerfile → démo capture écran → quantization (bonus)
```

### Hyperparamètres de départ suggérés

| Paramètre | RNN | LSTM | Transformer | DistilBERT |
|---|---|---|---|---|
| `embed_dim` | 64 | 128 | 128 | — (768 gelé) |
| `hidden_dim` | 128 | 256 | — | — |
| `num_layers` | 1 | 2 | 2 blocs | — |
| `dropout` | — | 0.3 | 0.1 | 0.1 |
| `lr` | 1e-3 | 5e-4 | 1e-4 | 2e-5 |
| `optimizer` | Adam | AdamW | AdamW | AdamW |
| `batch_size` | 64 | 64 | 32 | 16–32 |
| `MAX_LEN` | 128 | 128 | 128 | 128 |

---

## 6. Points de vigilance et erreurs classiques

### Preprocessing

| Erreur | Solution |
|---|---|
| Leak train→test sur le vocabulaire | Construire le vocab **uniquement** sur le train set ; tokens inconnus → `<UNK>` |
| Padding non masqué dans le Transformer | Toujours passer `key_padding_mask=(x == 0)` à `MultiheadAttention` |
| Oublier `stratify=y` dans le split | Crucial pour 27 classes potentiellement déséquilibrées |
| `MAX_LEN` différent entre modèles | Garder le même pour permettre la comparaison directe |

### Modélisation

| Erreur | Solution |
|---|---|
| Softmax avant CrossEntropyLoss | **Ne jamais** mettre `nn.Softmax` avant `nn.CrossEntropyLoss` — double softmax |
| `batch_first=False` (défaut PyTorch) | Toujours spécifier `batch_first=True` dans `nn.RNN/LSTM/GRU` |
| BiLSTM : mauvaise dim de la tête | `hidden[-1]` pour LSTM, `cat([hidden[-2], hidden[-1]], dim=1)` pour BiLSTM |
| Expériences non reproductibles | Fixer `torch.manual_seed(42)` + `numpy.random.seed(42)` au début de chaque run |

### Optimisation

| Erreur | Solution |
|---|---|
| Oublier `model.train()` / `model.eval()` | `train()` active le Dropout, `eval()` le désactive — critique pour les métriques |
| Early stopping sur la loss train | Toujours surveiller la loss **validation** |
| Scheduler avant la loss (ReduceLROnPlateau) | `scheduler.step(val_loss)` **après** l'évaluation |
| Gradient exploding sur RNN | `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |

### Déploiement

| Erreur | Solution |
|---|---|
| `torch.jit.trace` sur LSTM/Transformer | **Utiliser `script`** — `trace` fige les branches conditionnelles |
| Oublier `model.eval()` avant l'export | `eval()` avant `torch.jit.script(model)` et avant l'inférence FastAPI |
| Preprocessing différent en API vs entraînement | Même tokenisation, même `MAX_LEN`, même padding — encapsuler dans `preprocess(text)` |
| ONNX sans `dynamic_axes` | En production la taille de batch varie → `dynamic_axes={'input': {0: 'batch_size'}}` |

---

## 7. Contenu attendu du rapport (10–15 pages)

| Section rapport | Points | Ce qu'il faut montrer |
|---|---|---|
| Introduction + choix dataset | 10 pts | Justifier Bitext vs alternatives (taille, équilibre, langue) ; stats EDA (27 intentions, longueur moyenne des messages) |
| Préparation des données | inclus | Pipeline nettoyage → tokenisation → vocab → padding → split stratifié ; graphiques distribution classes + longueurs |
| Description architectures | 15 pts | Schéma de chaque architecture, nb de paramètres, choix justifiés (embed_dim, num_layers, dropout) |
| Protocole expérimental | inclus | Hyperparamètres fixes, seed, device, métriques de comparaison, mêmes données pour tous |
| Résultats commentés | 15 pts | Tableau comparatif (4+ modèles × accuracy/F1/latence), courbes d'apprentissage commentées (overfitting détecté ? résolu ?) |
| Analyse d'erreurs | 10 pts | Matrice de confusion du meilleur modèle, intentions les plus confondues, exemples concrets mal classifiés + hypothèse |
| Déploiement | inclus | Capture Swagger /predict, format d'export choisi + justification, mesure latence |
| Conclusion | inclus | Quel modèle déployer ? Pourquoi ? Pistes d'amélioration (BERT multilingue, données supplémentaires, distillation) |

---

## 8. Synthèse : ce que le projet évalue vraiment

Le projet est la **mise en pratique intégrale du cours** sur un problème réel de NLP industriel :

```
Partie I   → boucle d'entraînement PyTorch, CrossEntropyLoss, diagnostic overfitting
Partie II  → 4 architectures séquentielles (RNN → LSTM → BiLSTM → Transformer → BERT)
Partie III → optimisation du meilleur modèle (optimiseurs, schedulers, régularisation, search)
Partie IV  → pipeline MLOps complet (TorchScript, FastAPI, Docker, benchmark latence)
```

**La valeur ajoutée** n'est pas dans le score brut mais dans :
- La **comparaison argumentée** : pourquoi le LSTM bat le RNN sur des intentions longues, pourquoi le Transformer gère mieux les dépendances globales.
- L'**analyse d'erreurs** : quelles intentions sont systématiquement confondues (ex. "remboursement" vs "annulation de commande") et pourquoi.
- Le **travail d'optimisation documenté** : montrer les courbes avant/après dropout, avant/après scheduler.
- La **cohérence du pipeline** : le même texte brut entré dans l'API doit traverser exactement le même preprocessing que pendant l'entraînement.
