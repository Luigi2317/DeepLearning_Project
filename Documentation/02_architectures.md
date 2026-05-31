# 02 — Architectures de modèles

---

## Objectif global

Implémenter et comparer **6 architectures de classification d'intentions** sur le flux client (10 classes) :

| Architecture | Notebook | Données | Test Accuracy |
|---|---|---|---|
| RNN simple | `02_model_rnn.ipynb` | `client_data.pkl` | 57.6% |
| LSTM | `03_model_lstm.ipynb` | `client_data.pkl` | 59.5% |
| BiLSTM | `03_model_lstm.ipynb` | `client_data.pkl` | 59.5% |
| Transformer from scratch | `04_model_transformer.ipynb` | `client_data.pkl` | 57.3% |
| CNN + BiLSTM (hybride) | `07_model_cnn_lstm.ipynb` | `client_data.pkl` | 59.6% |
| **DistilBERT fine-tuné** | `05_model_distilbert.ipynb` | `client_text_data.pkl` | **63.2%** |

Les 3 premiers modèles utilisent le vocabulaire custom (15 000 tokens, indices entiers). DistilBERT utilise son propre tokenizer WordPiece — il ne partage pas les mêmes données d'entrée. La comparaison reste équitable sur le **test set** (même split, même labels).

---

## Composants partagés

### Données d'entrée

Artefacts produits par `01b_data_preparation_v8.ipynb` :

| Artefact | Format | Utilisé par |
|---|---|---|
| `vocab.pkl` | vocabulaire 15 000 tokens | RNN, LSTM, BiLSTM, Transformer, CNN+BiLSTM |
| `label_maps.json` | **10 classes client** | tous les modèles |
| `client_data.pkl` | `[(indices_entiers, label)]` | RNN, LSTM, BiLSTM, Transformer, CNN+BiLSTM |
| `client_text_data.pkl` | `[(text_str, label)]` | **DistilBERT uniquement** |

> `client_data.pkl` et `client_text_data.pkl` couvrent le même split train/val/test avec les mêmes labels — seul le **format du texte** diffère (indices vocab custom vs strings nettoyées brutes).

### Gestion du déséquilibre de classes

Le flux client (v8) contient **10 classes** avec `N_MAX=100 000` par classe. Les classes `short_ack` (courts remerciements) et `other` ont été supprimées car elles n'ont pas de signal sémantique propre et dégradaient les métriques globales.

Les class weights ont été testés en v1 avec DistilBERT mais ont été supprimés : sur un modèle pré-entraîné sur 11B tokens, les poids causaient une sur-prédiction massive de `non_english` (recall=0.94, precision=0.31). Pour les modèles from scratch (RNN, LSTM, Transformer), `CrossEntropyLoss()` standard est utilisé avec le dataset équilibré à N_MAX.

### Boucle d'entraînement commune

Même structure pour les 3 architectures, inspirée des TP5/TP6 :

```
Pour chaque époque :
  1. train_epoch() → loss + accuracy sur train
  2. eval_epoch()  → loss + accuracy sur val
  3. Early stopping si val_loss ne s'améliore pas pendant patience=3 époques
  4. Sauvegarder le meilleur état (best_val_loss)

Après entraînement :
  Restaurer le meilleur état → evaluate_model() sur test
```

L'**early stopping** (Cours III) empêche l'overfitting sans fixer le nombre d'époques a priori. La restauration du meilleur état garantit que l'évaluation finale utilise le modèle qui généralise le mieux, pas forcément le dernier.

### Évaluation commune

- `accuracy_score` globale
- `classification_report` par classe (precision, recall, F1)
- `confusion_matrix` avec les noms de classes
- Courbes train/val loss et accuracy

---

## Architecture 1 — RNN (baseline)

### Justification théorique

Le RNN (Cours II) traite une séquence token par token. À chaque pas de temps `t`, il met à jour un **état caché** `h_t` qui résume tout ce qui a été lu jusqu'ici :

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

**Avantages** : simple, léger, converge vite.

**Limite principale : le vanishing gradient.** Lors de la rétropropagation, les gradients sont multipliés à travers tous les pas de temps. Sur des séquences longues, ce produit tend vers 0 — le modèle n'apprend plus les dépendances du début de séquence. Sur nos tweets (MAX_LEN=64, moyenne ~17 mots), ce problème est atténué mais présent.

**Rôle dans le projet** : baseline de référence. Toute amélioration des architectures suivantes doit se mesurer par rapport au RNN.

### Structure

```
x (batch, seq_len)
    ↓
Embedding(vocab_size, embed_dim, padding_idx=0)  → (batch, seq_len, embed_dim)
    ↓
RNN(embed_dim, hidden_dim, batch_first=True)     → hidden : (1, batch, hidden_dim)
    ↓
hidden[-1]                                        → (batch, hidden_dim)
    ↓
Dropout(dropout)
    ↓
Linear(hidden_dim, output_dim)                    → (batch, 12)
```

### Pattern TP5 réutilisé

```python
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded       = self.embedding(x)
        output, hidden = self.rnn(embedded)          # output : (batch, seq, hidden_dim)
        # Mean pooling : plus robuste que hidden[-1] pour 10 classes sur tweets courts
        # hidden[-1] souffre du vanishing gradient sur 64 tokens → perd les mots du début
        return self.fc(self.dropout(output.mean(dim=1)))
```

Différences avec TP5 : `output_dim=10` (vs 2), dropout ajouté, **mean pooling** au lieu de `hidden[-1]` (correction du vanishing gradient — cf. fix v2 qui a fait passer de 12.7% → 57.6%).

---

## Architecture 2 — BiLSTM

### Justification théorique

Le LSTM (Cours II) résout le vanishing gradient en introduisant un **cell state** `c_t` (mémoire long terme) distinct du hidden state `h_t` (résumé court terme). Trois portes régulent le flux d'information :

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate  — quoi oublier de c_{t-1}
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate   — quoi ajouter à c_t
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) # Candidate cell
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        # Mise à jour — addition, pas multiplication
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
h_t = o_t ⊙ tanh(c_t)
```

La mise à jour de `c_t` par **addition** est la clé : le gradient remonte sans s'atténuer (*constant error carousel*).

**Bidirectionnel** : deux LSTM traitent la séquence en sens opposés. Le modèle dispose ainsi du contexte passé ET futur pour chaque token. Sur Twitter, "@AmericanAir my flight got cancelled" est mieux compris si le modèle sait que "cancelled" vient après la mention.

Les deux derniers états cachés (forward et backward) sont concaténés :
```
last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 2*hidden_dim)
```

### Structure

```
x (batch, seq_len)
    ↓
Embedding(vocab_size, embed_dim, padding_idx=0)      → (batch, seq_len, embed_dim)
    ↓
LSTM(embed_dim, hidden_dim, bidirectional=True)      → hidden : (2, batch, hidden_dim)
    ↓
cat([hidden[-2], hidden[-1]])                         → (batch, 2*hidden_dim)
    ↓
Dropout(dropout)
    ↓
Linear(2*hidden_dim, output_dim)                      → (batch, 12)
```

### Pattern TP6 réutilisé

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                  num_layers=1, batch_first=True, bidirectional=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)                        # hidden : (2, batch, hidden_dim)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)    # (batch, 2*hidden_dim)
        y_pred = self.fc(self.dropout(last_hidden))
        return y_pred
```

Différences avec TP6 : `bidirectional=True` depuis le départ, `output_dim=12`.

---

## Architecture 3 — Transformer from scratch

### Justification théorique

#### Encodage positionnel

Le Transformer (Cours II, Vaswani et al. 2017) traite tous les tokens **en parallèle** — il n'a pas de notion d'ordre intrinsèque. L'encodage positionnel injecte cette information en ajoutant un vecteur sinusoïdal à chaque embedding :

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Les fréquences différentes selon la dimension permettent à chaque position d'avoir une signature unique dans l'espace d'embedding.

#### Multi-Head Self-Attention

Pour chaque token, on calcule à quel point il doit « prêter attention » à chaque autre token :

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- **Q** (Queries) : ce que le token cherche
- **K** (Keys) : ce que chaque token contient
- **V** (Values) : ce que chaque token transmet

Le facteur `1/√d_k` évite que les scores soient trop grands (saturation du softmax). `MultiHead` répète ce mécanisme `num_heads` fois en parallèle — chaque tête apprend à regarder des relations différentes — puis concatène les sorties.

Un **masque de padding** (`key_padding_mask = (x == 0)`) est passé à l'attention pour que les tokens `<PAD>` n'influencent pas les scores.

#### TransformerBlock

Chaque bloc enchaîne deux sous-couches avec **connexions résiduelles** et **Layer Normalization** (Pre-LN) :

```
x → LayerNorm → MultiHeadSelfAttention → Dropout → (+x)  → résidu 1
  → LayerNorm → FFN(Linear→ReLU→Linear) → Dropout → (+x) → résidu 2
```

Le **FFN** est appliqué indépendamment sur chaque position : `embed_dim → ff_dim → embed_dim`.

#### Pooling

Le Transformer n'a pas d'état final naturel. On utilise une **moyenne pondérée masquée** sur tous les tokens non-PAD :

```
pooled = Σ(h_t · mask_t) / Σ(mask_t)
```

### Structure

```
x (batch, seq_len)
    ↓
Embedding × √embed_dim                             → (batch, seq_len, embed_dim)
    ↓
PositionalEncoding                                 → même shape
    ↓
Dropout
    ↓
N × TransformerBlock(embed_dim, num_heads, ff_dim) → même shape
    ↓
Masked mean pooling                                → (batch, embed_dim)
    ↓
Linear(embed_dim, output_dim)                      → (batch, 12)
```

### Pattern TP7 réutilisé

- `PositionalEncoding` : `register_buffer('pe', ...)` — vecteur non appris, déplacé avec le modèle sur GPU
- `TransformerBlock` : Pre-LN, `nn.MultiheadAttention(batch_first=True)`
- `padding_mask = (x == 0)` passé comme `key_padding_mask`
- Mise à l'échelle des embeddings : `embedding(x) * math.sqrt(embed_dim)` (papier original)

---

## Architecture 4 — DistilBERT fine-tuné

### Justification théorique

#### Pourquoi un modèle pré-entraîné ?

RNN, BiLSTM et Transformer from scratch apprennent les représentations **à partir de zéro** sur nos données d'entraînement (~134 000 tweets). Leurs embeddings (128 dims, vocabulaire 15 000 tokens) ne capturent que les patterns présents dans notre dataset.

DistilBERT a été pré-entraîné sur **~11 milliards de tokens** (Wikipedia + BooksCorpus) avec deux objectifs non-supervisés : **Masked Language Modeling** (prédire des tokens masqués au milieu d'une phrase) et **Next Sentence Prediction**. Ces objectifs forcent le modèle à apprendre des représentations contextuelles riches — chaque token est représenté différemment selon son contexte (contrairement à Word2Vec qui donne le même vecteur à "bank" dans "river bank" et "bank account").

Le **fine-tuning** consiste à continuer l'entraînement de toutes les couches sur notre tâche spécifique avec un LR très faible (2e-5) pour ne pas détruire les représentations pré-apprises. Seule la tête de classification linéaire (`768 → 12`) est initialisée aléatoirement.

```
Pré-entraînement (11B tokens) → Représentations génériques riches
         +
Fine-tuning (134K tweets, LR=2e-5) → Spécialisation classification d'intentions
```

#### Architecture DistilBERT

DistilBERT est une version distillée de BERT : 6 couches Transformer (vs 12 pour BERT-base), ~66M paramètres (vs 110M). La distillation de connaissances préserve 97% des capacités de BERT avec 40% moins de paramètres et 60% plus rapide à l'inférence.

Chaque couche est un Transformer encoder standard (identique à notre Architecture 3, mais avec `embed_dim=768`, `num_heads=12`, `ff_dim=3072`). Le token spécial `[CLS]` est ajouté en début de séquence — sa représentation finale agrège l'information de toute la séquence et sert d'entrée à la tête de classification.

```
Texte brut
    ↓
WordPiece Tokenizer (~30 000 sous-mots) + [CLS] / [SEP] + position IDs
    ↓
DistilBERT encoder × 6 blocs Transformer (embed_dim=768, heads=12)
    ↓
h_[CLS] — représentation du token [CLS]   → (batch, 768)
    ↓
pre_classifier : Linear(768, 768) → ReLU → Dropout
    ↓
classifier : Linear(768, 12)              → (batch, 12)
```

#### Tokenizer WordPiece vs vocabulaire custom

Notre vocabulaire custom (15 000 tokens) traite chaque mot comme une unité : "cancelled" → un indice. Le tokenizer WordPiece de DistilBERT décompose les mots rares en sous-mots : "cancellation" → `["cancel", "##lation"]`. Cela permet de gérer les mots hors-vocabulaire sans recourir à `<UNK>` — un avantage sur les tweets qui contiennent beaucoup de noms de marques et d'abréviations.

`MAX_LEN=128` : les tweets font en moyenne 17 tokens WordPiece mais certains atteignent 60-80 sous-mots après décomposition WordPiece. 128 couvre >99.9% des tweets sans coût mémoire significatif avec BATCH_SIZE=32 (DistilBERT gère jusqu'à 512).

#### Warmup linéaire

Le fine-tuning de BERT est sensible au LR initial. Un LR soudain de 2e-5 dès la première étape peut détruire les poids pré-entraînés avant que le signal de la nouvelle tâche ne s'installe. Le **warmup linéaire** sur les 10% premiers steps monte le LR de 0 → 2e-5, puis le scheduler le redescend linéairement vers 0 sur le reste de l'entraînement. Ce profil est la pratique standard pour le fine-tuning BERT (Devlin et al. 2018).

### Structure

```python
DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=12)
```

Tous les paramètres sont entraînables (`requires_grad=True`). La tête de classification (initialisée aléatoirement) reçoit un gradient fort dès le départ ; les couches profondes du Transformer, déjà bien initialisées, reçoivent un gradient faible grâce au LR réduit.

### Pattern TP7 réutilisé

La structure de `train_epoch_bert` / `eval_epoch_bert` / `train_model_bert` est identique au TP7 avec deux adaptations :
- `scheduler.step()` appelé après chaque **batch** (pas après chaque époque) — le scheduler de warmup est batch-level
- `outputs.logits` au lieu de `outputs` directement — convention HuggingFace : les modèles retournent un objet `BaseModelOutput`, les logits sont dans `.logits`

---

## Paramètres techniques

### RNN / BiLSTM / Transformer (02_models_v1.ipynb)

| Paramètre | Valeur | Justification |
|---|---|---|
| `EMBED_DIM` | 128 | Tweets courts mais 12 classes — 64 insuffisant |
| `HIDDEN_DIM` | 256 | Capacité suffisante pour 12 intentions |
| `DROPOUT` | 0.3 | Standard TP6 — prévient l'overfitting |
| `NUM_EPOCHS` | 15 / 30 (Transformer) | Avec early stopping |
| `LR_RNN` | 5e-4 | Réduit : 1e-3 trop agressif avec weighted loss + 12 classes |
| `LR_BILSTM` | 5e-4 | Standard TP6 |
| `LR_TRANSF` | 1e-4 | Réduit : 5e-4 causait divergence sur Transformer from scratch |
| `NUM_HEADS` | 4 | 128 / 4 = 32 dims par tête |
| `FF_DIM` | 256 | 2× embed_dim |
| `NUM_BLOCKS` | 2 | Compromis capacité/vitesse |
| `DROPOUT_TRANSF` | 0.2 | Plus faible (connexions résiduelles aident déjà) |
| `PATIENCE` | 5 | Augmenté : 3 trop court, early stop prématuré |
| `BATCH_SIZE` | 64 | Standard TP5/TP6 |

### DistilBERT (04_model_distilbert.ipynb)

| Paramètre | v1 (initial) | v2 (optimisé) | Justification du changement |
|---|---|---|---|
| `MODEL_NAME` | `distilbert-base-uncased` | = | — |
| `MAX_LEN` | 64 | **128** | Certains tweets atteignent 60-80 tokens WordPiece — 64 tronquait de l'information |
| `BATCH_SIZE` | 32 | = | — |
| `LR` | 2e-5 | = | — |
| `WEIGHT_DECAY` | 0.01 | = | — |
| `NUM_EPOCHS` | 8 | **12** | Warmup sur 10% = ~2 époques avant apprentissage réel — 8 trop court |
| `PATIENCE` | 3 | **4** | Évite early stopping prématuré pendant la phase post-warmup |
| `class_weights` | oui | **non** | Supprimés — voir section Optimisations |
| `N_CLASSES` | 12 | **11** | Classe `other` supprimée — voir section Optimisations |
| warmup | 10% des steps | = | — |

---

---

## Architecture 5 — CNN + BiLSTM (hybride extracteur + classifieur)

### Pourquoi cette architecture ?

Les 4 architectures précédentes (RNN, LSTM, BiLSTM, Transformer) traitent directement les embeddings token par token. Elles ont deux limites distinctes :

- **RNN / LSTM / BiLSTM** : bons sur les dépendances séquentielles, mais leur premier traitement est un embedding brut — aucune notion de **groupe de mots** (n-grammes). Le mot "cancelled" seul dit moins que "flight got cancelled".
- **Transformer from scratch** : voit tous les tokens en parallèle via l'attention, mais avec seulement 2 blocs et des embeddings 128d appris de zéro sur ~700K tweets, sa capacité est limitée.

L'architecture CNN + BiLSTM **combine les avantages** des deux familles :
1. **CNN** (Cours I) : extrait des patterns locaux sur des fenêtres de k tokens — équivalent à un détecteur de n-grammes appris automatiquement
2. **BiLSTM** (Cours II) : modélise les dépendances séquentielles entre ces patterns locaux

C'est exactement l'hybridation mentionnée dans le sujet : **"encodeur CNN 1D + LSTM sur les embeddings"**.

---

### Rôle 1 : CNN comme extracteur de features

#### Principe

Un filtre Conv1D de taille `kernel_size=3` glisse sur la séquence et examine 3 tokens consécutifs à chaque position. Il produit un scalaire : son score de détection pour ce pattern à cet endroit.

```
Embeddings : [e_1, e_2, e_3, e_4, e_5, ..., e_64]   (chaque e_i ∈ R^128)
             └────────┘         (fenêtre kernel=3, position 1)
                  └────────┘    (fenêtre kernel=3, position 2)
                       └────────┘ ...
```

Avec `num_filters=256` filtres, on apprend 256 détecteurs de patterns différents. Certains détecteront "flight cancelled", d'autres "not working", d'autres "please refund".

#### Pourquoi Conv1D et pas Conv2D ?

Conv2D est utilisé pour les images (deux dimensions spatiales). Un texte est une séquence 1D — la convolution glisse dans une seule direction (le temps/position). Conv1D est la bonne opération.

#### Réduction de séquence — avantage sur le BiLSTM

Avec `kernel_size=3` et `padding=0` : séquence de longueur 64 → longueur 62. Le BiLSTM traite donc une séquence 3% plus courte. L'effet est plus significatif pour le vanishing gradient : la rétropropagation traverse 62 pas au lieu de 64 — légère amélioration, mais surtout les features sont déjà des abstractions de 3 tokens, donc **chaque pas du BiLSTM représente plus d'information**.

#### Implémentation — points techniques

```python
self.conv = nn.Conv1d(
    in_channels=embed_dim,   # le CNN lit la dimension des embeddings
    out_channels=num_filters, # produit num_filters features par position
    kernel_size=kernel_size,
    padding=0
)
```

`Conv1d` attend `(batch, channels, length)` mais nos embeddings sont `(batch, length, embed_dim)` → **permute obligatoire** avant et après le Conv1d :

```python
cnn_in  = embedded.permute(0, 2, 1)    # (batch, embed_dim, seq_len)
cnn_out = relu(conv(cnn_in))            # (batch, num_filters, seq_len')
cnn_out = cnn_out.permute(0, 2, 1)     # (batch, seq_len', num_filters)
```

---

### Rôle 2 : BiLSTM comme classifieur

Le BiLSTM reçoit non plus des embeddings bruts mais des **features CNN** — chaque position représente déjà un contexte local enrichi. Son rôle est de modéliser les **relations entre ces features** le long de la séquence.

Exemple pour le tweet `"my flight got cancelled and no one answered"` :
- Feature CNN position 1 : détecteur "flight got" → activé
- Feature CNN position 3 : détecteur "no one answered" → activé
- **BiLSTM** : apprend que "flight got [cancelled]" + "no one [answered]" ensemble → classe `transport_complaint` avec forte probabilité

Bidirectionnel : le BiLSTM lit la séquence de features dans les deux sens, ce qui permet de contextualiser chaque feature par rapport aux features qui précèdent ET qui suivent.

---

### Paramètres techniques — justifications détaillées

| Paramètre | Valeur | Justification |
|---|---|---|
| `EMBED_DIM` | 128 | Idem autres modèles — comparaison équitable |
| `NUM_FILTERS` | 256 | Nombre de détecteurs de patterns CNN. Trop petit (64) → sous-représentation. Trop grand (512) → overfitting + lent |
| `KERNEL_SIZE` | 3 | Trigrammes : fenêtre optimale pour tweets courts (~17 tokens). Bigrammes (2) : trop peu de contexte. Quadrigrammes (4) : rare dans tweets |
| `HIDDEN_DIM` | 256 | Capacité BiLSTM. Sortie = 2×256=512 (bidirectionnel). Cohérent avec NUM_FILTERS |
| `NUM_LAYERS` | 2 | Couche 1 : patterns de bas niveau sur features CNN. Couche 2 : abstractions de haut niveau |
| `DROPOUT` | 0.3 | Standard TP6. Appliqué entre couches LSTM ET avant la couche Dense |
| `LR` | 5e-4 | Légèrement plus faible que 1e-3 : les features CNN amplifient les gradients |
| `N_EPOCHS` | 15 | Avec early stopping (patience=4) : arrêt automatique si pas d'amélioration |
| `PATIENCE` | 4 | Plus conservateur que patience=3 : les CNN+LSTM peuvent stagner 2-3 epochs avant de progresser |
| `scheduler` | `ReduceLROnPlateau(factor=0.5, patience=2)` | Divise LR par 2 si val_loss stagne → affine la convergence sans fixer un schedule |
| `BATCH_SIZE` | 64 | Standard TP5/TP6 |

---

### Comparaison avec les autres architectures

| Architecture | Voit les n-grammes | Voit les dépendances longues | Paramètres | Complexité |
|---|---|---|---|---|
| RNN | ❌ (token par token) | ❌ (vanishing gradient) | ~2M | Faible |
| BiLSTM | ❌ | ✅ (cell state) | ~3M | Moyenne |
| Transformer | ✅ (attention globale) | ✅ | ~3M | Moyenne |
| **CNN + BiLSTM** | **✅ (Conv1D)** | **✅ (BiLSTM sur features)** | ~4M | Moyenne+ |
| DistilBERT | ✅✅ (pré-entraîné) | ✅✅ | 66M | Élevée |

CNN + BiLSTM est la seule architecture entièrement apprise from scratch qui combine explicitement les deux capacités.

---

### Lien avec les TP

- **TP5** : `nn.Embedding(padding_idx=0)`, `pad_sequence`, `collate_fn`, boucle train/eval
- **TP6** : BiLSTM `torch.cat([hidden[-2], hidden[-1]])`, dropout entre couches LSTM
- **TP7** : inspiration de la modularité extracteur → classifieur (TransformerBlock + tête MLP)
- **Cours I** : Conv1D, ReLU, detection de features locales dans des séquences
- **Cours II** : BiLSTM, cell state, bidirectionalité

---

## Plan d'implémentation

**Fichier cible** : `notebooks/02_models_v1.ipynb`

**Blocs prévus** :

| # | Bloc | Ce qu'il fait | Input → Output |
|---|---|---|---|
| 1 | Config & imports | Constantes, seed, device | — → paramètres globaux |
| 2 | Chargement artefacts | vocab, label_maps, client_data | → données encodées |
| 3 | DataLoaders & poids de classe | IntentDataset, collate_fn, weighted criterion | → loaders + criterion |
| 4 | Fonctions communes | train_epoch, eval_epoch, train_model, plot_history, evaluate_model | → fonctions réutilisables |
| 5 | Architecture RNN | Classe RNNClassifier | → modèle défini |
| 6 | Entraînement RNN | train_model() + courbes | → history_rnn |
| 7 | Évaluation RNN | evaluate_model() + sauvegarde | → acc_rnn, rnn_client.pt |
| 8 | Architecture BiLSTM | Classe BiLSTMClassifier | → modèle défini |
| 9 | Entraînement BiLSTM | train_model() + courbes | → history_bilstm |
| 10 | Évaluation BiLSTM | evaluate_model() + sauvegarde | → acc_bilstm, bilstm_client.pt |
| 11 | Architecture Transformer | PositionalEncoding + TransformerBlock + TransformerClassifier | → modèle défini |
| 12 | Entraînement Transformer | train_model() + courbes | → history_transf |
| 13 | Évaluation Transformer | evaluate_model() + sauvegarde | → acc_transf, transformer_client.pt |
| 14 | Comparaison | Tableau récapitulatif + graphique | → comparaison finale |

**Patterns TP utilisés** :
- TP5 : `super(RNN, self).__init__()`, `hidden[-1]`, `y_pred`, boucle train/eval
- TP6 : `torch.cat([hidden[-2], hidden[-1]], dim=1)`, dropout sur last_hidden
- TP7 : `PositionalEncoding` avec `register_buffer`, TransformerBlock Pre-LN, `padding_mask = (x == 0)`, masked mean pooling

---

## Implémentation réalisée

### 02_models_v1.ipynb — RNN / BiLSTM / Transformer

**Blocs écrits** : 1–14 (Config, DataLoaders, Fonctions communes, RNN×3 blocs, BiLSTM×3 blocs, Transformer×3 blocs, Comparaison)

**Écarts par rapport au plan** : Aucun

### 04_model_distilbert.ipynb — DistilBERT fine-tuné

**Blocs écrits** : 1–7 (Config, Chargement données, Tokenizer+Dataset+DataLoaders, Fonctions entraînement, Modèle, Entraînement, Évaluation+sauvegarde)

**Écarts par rapport au plan** : voir section Optimisations ci-dessous

---

### Optimisations DistilBERT — itérations successives

#### Problème v1 (accuracy 58%)

L'exécution initiale avec `CrossEntropyLoss(weight=class_weights)` et `MAX_LEN=64` produisait :

```
non_english  precision=0.31  recall=0.94
```

Recall 0.94 = le modèle trouve presque tous les vrais `non_english`.
Precision 0.31 = sur 3 prédictions `non_english`, 2 sont fausses.

**Cause** : `non_english` ne contient que 596 échantillons (le minimum du dataset) → son poids dans `CrossEntropyLoss` est très élevé → DistilBERT est pénalisé fortement pour chaque `non_english` raté → il apprend à prédire `non_english` partout. Un modèle pré-entraîné sur 11B tokens n'a pas besoin de cette correction — elle devient contre-productive.

#### Optimisation 1 — Supprimer les class weights (→ 62%)

```python
# Bloc 6 — remplace criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
```

Résultat : `non_english` passe à precision=0.91 / recall=0.68. Accuracy globale +4%.

#### Optimisation 2 — MAX_LEN 64 → 128 et plus d'époques

```python
MAX_LEN    = 128  # Bloc 1
NUM_EPOCHS = 12   # Bloc 1
PATIENCE   = 4    # Bloc 1
```

Justification MAX_LEN : à 64 tokens WordPiece, certains tweets étaient tronqués (la décomposition WordPiece allonge les tokens par rapport au custom vocab). 128 couvre >99.9% des tweets.

Justification époques : avec warmup sur 10% des steps, le modèle ne commence à vraiment apprendre qu'à l'époque 2. PATIENCE=3 entraînait un early stopping prématuré.

#### Optimisation 3 — Suppression de la classe `other`

La matrice de confusion après optimisation 1+2 révélait que `other` (support=4500, precision=0.44) agissait comme un "trou noir" : `no_response` → `other` (1212 erreurs), `internet_outage` → `other` (609), `ride_issue` → `other` (723). Le modèle prédit `other` par défaut quand il est incertain.

**Pourquoi `other` est irréductible** : cette classe a été créée par KMeans pour tous les tweets qui ne rentraient dans aucune catégorie propre — tweets courts, réactifs, sans signal sémantique fort. Aucun modèle ne peut apprendre une frontière qui n'existe pas dans le texte. Re-labelliser au sein de `other` produirait des sous-clusters encore plus arbitraires.

**Solution** : supprimer `other` de l'entraînement dans `01b_data_preparation_v6.ipynb` :

```python
# Bloc 6 de 01b — après balance_dataset
df_bal_client  = df_bal_client[df_bal_client['label'] != 2].reset_index(drop=True)
label_remap    = {0:0, 1:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10}
df_bal_client['label'] = df_bal_client['label'].map(label_remap)
```

Et `CLIENT_LABELS` mis à jour avec 11 classes (indices 0–10, `other` retiré, indices 3–11 décalés de -1).

Dans `04_model_distilbert.ipynb` : `N_CLASSES = 11`.

---

### Résultats / Observations

| Architecture | Notebook | Test Accuracy | Époques | Remarques |
|---|---|---|---|---|
| RNN (v2 — mean pooling) | `02_model_rnn` | **57.6%** | 10 | Fix mean pooling : 12.7% → 57.6%. Overfitting léger après epoch 4 |
| LSTM | `03_model_lstm` | **59.5%** | 15 | Cell state résout le vanishing gradient |
| BiLSTM | `03_model_lstm` | **59.5%** | 15 | Bidirectionnel — même résultat que LSTM sur tweets courts |
| Transformer from scratch | `04_model_transformer` | **57.3%** | 10 | Comparable RNN/LSTM sans pré-entraînement |
| CNN + BiLSTM (hybride) | `07_model_cnn_lstm` | **59.6%** | 9 | Meilleure architecture from scratch — CNN extracteur + BiLSTM classifieur |
| DistilBERT (10 classes, 30K exemples) | `05_model_distilbert` | **63.2%** | 3 | Meilleur modèle — pré-entraîné sur 11B tokens |

> **Note sur le Transformer** : résultat comparable au BiLSTM `from scratch`. L'avantage de l'attention globale se manifeste davantage avec des poids pré-entraînés (DistilBERT) — sur des tweets courts (~17 tokens), le BiLSTM bidirectionnel capture déjà bien les dépendances.
