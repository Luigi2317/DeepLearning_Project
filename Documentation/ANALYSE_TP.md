# Analyse des Travaux Pratiques — Deep Learning

---

## Vue d'ensemble

| TP | Fichier | Thème | Dataset | Modèle | Statut |
|---|---|---|---|---|---|
| TP1 | `1_les_fleurs_CORRIGE.ipynb` | Neurone artificiel from scratch | Iris (2 classes) | Neurone custom | Corrigé |
| TP2 | `2_dermatologie_CORRIGEE.ipynb` | MLP multiclasse | Dermatologie (6 classes) | MLP 2 couches | Corrigé |
| TP3 | `3_Lab_chiffres_MLP_CORRIGE.ipynb` | MLP sur images | MNIST | MLP 3 couches | Corrigé |
| TP4 | `4_Lab_chiffres_CNN_CORRIGE.ipynb` | CNN sur images | MNIST | CNN 3 conv | Corrigé |
| TP5 | `5_Lab_RNN.ipynb` | RNN sentiment | IMDB | RNN vanille | À compléter |
| TP6 | `6_Lab_LSTM_CORRIGEv2.ipynb` | LSTM + comparaison | IMDB | LSTM / BiLSTM | Corrigé |
| TP7 | `7_Lab_Transformer.ipynb` | Transformer from scratch | IMDB | TransformerClassifier | À compléter |
| TP8 | `8_Lab_OptimRegu.ipynb` | Optimisation & régularisation | CIFAR-10 | ResNet18 / EfficientNet-B0 | À compléter |
| TP10 | `10_Lab_ExInfDepl.ipynb` | Export & déploiement | CIFAR-10 | ResNet18 / EfficientNet-B0 | À compléter |

---

## TP1 — Neurone artificiel from scratch (Les Fleurs)

> `TP/1_les_fleurs_CORRIGE.ipynb`

### Objectif
Implémenter un neurone artificiel complet **sans utiliser `nn.Linear`**, pour comprendre le fonctionnement interne de PyTorch (poids, biais, forward, backprop).

### Dataset
- **Iris** : 100 exemples, 2 classes (setosa vs versicolor), 2 features (longueur et largeur des pétales)
- Split : 67% train / 33% test (`train_test_split`)
- Données converties en tenseurs float32

### Architecture

```python
class Model(nn.Module):
    def __init__(self, in_features=2, out_features=1):
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.bias    = nn.Parameter(torch.randn(out_features))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = torch.mm(x, self.weights) + self.bias
        return self.sigmoid(z)
```

- Multiplication matricielle manuelle : `torch.mm`
- Paramètres déclarés avec `nn.Parameter` pour que l'autograd les suive
- Sortie dans `]0, 1[` via Sigmoid → interprétée comme probabilité

### Entraînement
| Hyperparamètre | Valeur |
|---|---|
| Loss | MSELoss |
| Optimiseur | SGD |
| Learning rate | 0.1 |
| Epochs | 100 |
| Batch | Plein dataset |

### Résultats
- Classification binaire quasi-parfaite sur données linéairement séparables
- Visualisation de la **frontière de décision** : droite `y = (0.5 - a*x - c) / b` tracée analytiquement depuis les poids appris
- Matrice de confusion affichée avec `ConfusionMatrixDisplay`

### Points pédagogiques
- Distinguer `nn.Parameter` (paramètre apprenable) d'un simple tenseur
- Comprendre que `torch.mm` + biais est exactement ce que fait `nn.Linear` en interne
- La frontière de décision est une droite car le modèle est linéaire (un seul neurone)

---

## TP2 — MLP Multiclasse (Dermatologie)

> `TP/2_dermatologie_CORRIGEE.ipynb`

### Objectif
Construire un MLP complet avec DataLoader pour un problème de classification multiclasse sur données tabulaires médicales.

### Dataset
- **Dermatologie** : 366 patients, **6 classes** de pathologies cutanées, 34 features catégorielles + 1 feature numérique (âge)
- Répartition des classes : 112 / 72 / 61 / 52 / 49 / 20 (déséquilibrée)

### Preprocessing
```python
# Variables catégorielles (0-3) → OneHotEncoder → 129 features binaires
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[columns_to_encode])

# Âge → StandardScaler → 1 feature normalisée (moyenne=0, var=1)
scaler = StandardScaler()
encoded_df['Age_normalized'] = scaler.fit_transform(df[['age']])
# Résultat : 130 features au total
```

### Architecture
```
Input(130) → Linear(130,32) → Sigmoid → Linear(32,6) → Sigmoid → Softmax → Output(6)
```

### Entraînement
| Hyperparamètre | Valeur |
|---|---|
| Loss | CrossEntropyLoss |
| Optimiseur | Adam (lr par défaut) |
| Batch size | 64 |
| Epochs | 200 |
| Split | 80% train / 20% test (stratifié) |

### Infrastructure DataLoader
```python
dataset     = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```
- `shuffle=True` sur le train, `shuffle=False` sur le test
- Calcul de la perte séparément sur train et test à chaque époque → courbes de loss

### Points pédagogiques
- OneHotEncoding obligatoire pour les variables ordinales catégorielles (évite d'induire un ordre artificiel)
- La Sigmoid en couche cachée est sous-optimale (vanishing gradient) — choix pédagogique pour introduire le problème
- Matrice de confusion avec seaborn heatmap pour visualiser les confusions entre les 6 pathologies

---

## TP3 — MLP sur Images (MNIST)

> `TP/3_Lab_chiffres_MLP_CORRIGE.ipynb`

### Objectif
Classifier des chiffres manuscrits avec un MLP et observer sa limite : l'absence d'invariance aux translations.

### Dataset
- **MNIST** : 60 000 images train, 10 000 images test, 28×28 pixels niveaux de gris, 10 classes

### Architecture
```
flatten(28×28→784) → Linear(784,120) → ReLU → Linear(120,84) → ReLU → Linear(84,10) → Softmax
```
- **105 214 paramètres** au total
- Flatten : `torch.flatten(x, start_dim=1)` pour conserver la dimension batch

### Entraînement
| Hyperparamètre | Valeur |
|---|---|
| Loss | CrossEntropyLoss |
| Optimiseur | Adam |
| Learning rate | 0.001 |
| Batch size | 100 (train) / 500 (test) |
| Epochs | 10 |

### Résultats
| Données | Accuracy |
|---|---|
| MNIST standard | **97%** |
| MNIST décalé de 3px | **18%** (≈ hasard = 10%) |

### Expérience clé : robustesse aux translations
Les images du test set sont décalées de 3 pixels (crop + padding avec zéros) :
```python
test_data.data = test_data.data[:, 3:, 3:]  # crop haut+gauche
padding = (0, 3, 0, 3)                       # pad bas+droite
test_data.data = F.pad(test_data.data, padding)
```
→ **Chute de 97% à 18%** : le MLP n'a aucune invariance aux translations car chaque pixel est traité comme une feature indépendante. **Motivation directe pour les CNN.**

### Points pédagogiques
- Courbe de perte : l'écart train/test à la 1ère époque s'explique par le fait que la perte train est calculée *au fil* de l'entraînement, la perte test *après* l'époque
- Confusion la plus fréquente : **5 confondu avec 3** (29 fois)
- La précision se calcule depuis la matrice de confusion : `VP_diagonale / total`

---

## TP4 — CNN sur Images (MNIST)

> `TP/4_Lab_chiffres_CNN_CORRIGE.ipynb`

### Objectif
Remplacer le MLP par un CNN et démontrer l'invariance aux translations.

### Architecture détaillée
```
Input(1,28,28)
  → Conv2d(1,32,3, padding=1) → ReLU → MaxPool(2) → (32,14,14)
  → Conv2d(32,64,3, padding=1) → ReLU → MaxPool(2) → (64,7,7)
  → Conv2d(64,128,3, padding=0) → ReLU → MaxPool(2) → (128,2,2)
  → flatten → Linear(512,10)
```

| Étape | Canaux | Taille spatiale |
|---|---|---|
| Entrée | 1 | 28×28 |
| cnn1 + pool | 32 | 14×14 |
| cnn2 + pool | 64 | 7×7 |
| cnn3 + pool | 128 | 2×2 |
| flatten | — | 512 |

- **97 802 paramètres** (moins que le MLP, 105 214 !)
- Logits bruts en sortie (sans Softmax) → `CrossEntropyLoss` applique le log_softmax en interne

### Entraînement
| Hyperparamètre | Valeur |
|---|---|
| Loss | CrossEntropyLoss (logits) |
| Optimiseur | Adam |
| Learning rate | 1e-5 |
| Batch size | 128 (train) / 500 (test) |
| Epochs | 10 |

### Résultats
| Données | MLP | CNN |
|---|---|---|
| MNIST standard | 97% | ~94% |
| MNIST décalé 3px | **18%** | **29%** |

Le CNN est plus robuste aux translations grâce au partage de poids et au pooling, même si l'invariance n'est pas totale (amélioration 18% → 29%).

### Points pédagogiques
- Ne pas mettre de Softmax avant CrossEntropyLoss → double softmax = bug
- `padding=1` préserve la taille spatiale ; `padding=0` la réduit de 2
- Le CNN a *moins* de paramètres que le MLP malgré 3 couches convolutives : grâce au partage de poids

---

## TP5 — RNN pour l'Analyse de Sentiment

> `TP/5_Lab_RNN.ipynb` *(à compléter)*

### Objectif
Implémenter un RNN pour la classification de sentiment, puis analyser ses limites sur les longues séquences.

### Dataset
- **IMDB** : 25 000 critiques de films train, 25 000 test, classes parfaitement équilibrées (12 500 positif / 12 500 négatif)

### Preprocessing
```python
VOCAB_SIZE = 10000   # 10K mots les plus fréquents
MAX_LEN    = 200     # troncature des séquences longues

# Tokenisation : mot → indice (vocab.get(w, vocab['<UNK>']))
# Padding : pad_sequence(..., padding_value=0) pour uniformiser les batchs
# collate_fn : gère le padding dynamique dans le DataLoader
```

### Architecture (à compléter)
```python
class RNN(nn.Module):
    # 1) nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    # 2) nn.RNN(embed_dim, hidden_dim, batch_first=True)
    # 3) nn.Linear(hidden_dim, output_dim)
    # forward : embedding → RNN → hidden[-1] → Linear

EMBED_DIM  = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 2
```

### Entraînement (à compléter)
| Hyperparamètre | Valeur suggérée |
|---|---|
| Loss | CrossEntropyLoss |
| Optimiseur | Adam lr=1e-3 |
| Epochs | 5 |
| Batch size | 64 (train) / 256 (test) |

### Questions à traiter
- Pourquoi afficher les deux courbes train/test ?
- Que se passe-t-il si la perte test augmente alors que la perte train diminue ? (overfitting)
- Limites du RNN sur les longues séquences : vanishing gradient, mémoire bornée
- Bonus : `predict_sentiment(critique, model, vocab)` sur une critique inventée

---

## TP6 — LSTM et Comparaison RNN vs LSTM vs BiLSTM

> `TP/6_Lab_LSTM_CORRIGEv2.ipynb`

### Objectif
Maîtriser l'architecture LSTM, comprendre ses avantages sur le RNN vanille et implémenter un BiLSTM.

### Dataset
Identique au TP5 (IMDB, même preprocessing) pour permettre la **comparaison directe**.

### Architecture LSTM
```python
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)                    # (batch, seq, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)   # hidden : (num_layers, batch, hidden_dim)
        last_hidden = hidden[-1]                        # (batch, hidden_dim)
        return self.fc(self.dropout(last_hidden))
```

**871 682 paramètres** (vs ~400K pour le RNN équivalent — ×4 dû aux 4 portes).

### Architecture BiLSTM (bonus)
```python
self.lstm = nn.LSTM(..., bidirectional=True)
# hidden : (num_layers*2, batch, hidden_dim)
last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 2*hidden_dim)
self.fc = nn.Linear(hidden_dim * 2, output_dim)
```

### Entraînement
| Modèle | lr | Epochs | Device |
|---|---|---|---|
| LSTM | 5e-4 | 20 | GPU (cuda) |
| RNN (comparaison) | 1e-3 | 20 | GPU |
| BiLSTM | 1e-3 | 20 | GPU |

### Résultats comparatifs
| Modèle | Accuracy typique |
|---|---|
| RNN vanille | 75–82% |
| LSTM (2 couches) | 87–90% |
| BiLSTM | 88–91% |

### Points pédagogiques clés
- **Pourquoi LSTM > RNN** : mise à jour additive `c(t) = f⊙c(t-1) + i⊙c̃(t)` (addition ≠ multiplication) → *constant error carousel* → gradient préservé
- Le LSTM a 4× plus de paramètres **mais** la performance supérieure n'est pas due aux paramètres seuls — un RNN avec 4× plus de hidden_dim ne résoudrait pas le vanishing gradient
- **Quand préférer RNN** : séquences courtes (<50 tokens), contraintes de ressources, prototypage rapide
- BiLSTM inutilisable pour la génération temps réel (nécessite séquence complète)

---

## TP7 — Transformer from Scratch

> `TP/7_Lab_Transformer.ipynb` *(à compléter)*

### Objectif
Implémenter un Transformer complet depuis zéro : encodage positionnel, multi-head attention, bloc Transformer, classifieur. Comparer avec le RNN du TP5.

### Dataset
IMDB (même preprocessing — comparaison directe RNN/Transformer).

### Architecture cible
```
Input(indices) → Embedding × √d_model → PositionalEncoding → N × TransformerBlock
              → Pooling(moyenne masquée) → Linear → Output(2)
```

#### PositionalEncoding (fourni)
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# Ajouté aux embeddings, non substituable
```

#### TransformerBlock (à compléter)
```python
# Sous-couche 1 : Self-Attention
x = x + Dropout(MultiHeadAttention(LayerNorm(x)))
# Sous-couche 2 : Feed-Forward
x = x + Dropout(FFN(LayerNorm(x)))
# FFN : Linear(embed_dim, ff_dim) → ReLU → Linear(ff_dim, embed_dim)
```

#### Hyperparamètres
| Paramètre | Valeur |
|---|---|
| embed_dim | 64 |
| num_heads | 4 |
| ff_dim | 128 |
| num_blocks | 2 |
| dropout | 0.1 |
| max_len | 200 |

### Concepts à implémenter
- **Masque de padding** : `padding_mask = (x == 0)` — les tokens PAD n'influencent pas l'attention
- **Pooling masqué** : moyenne sur les tokens non-PAD uniquement
- **Mise à l'échelle** : `embedding × math.sqrt(embed_dim)` (papier original)
- **SingleHeadAttention (bonus)** : `Attention(Q,K,V) = softmax(QK⊤/√dk) · V` implémentée à la main sans `nn.MultiheadAttention`

### Visualisation de l'attention
```python
# Récupère les poids d'attention du premier bloc pour une phrase donnée
# Visualisation heatmap : tokens en ligne (Query) × tokens en colonne (Key)
_, attn_weights = model.blocks[0].attention(x_norm, x_norm, x_norm)
# → identifie quels mots influencent la décision ("wonderful", "loved"...)
```

### Tableau comparatif à compléter
| Critère | RNN | Transformer |
|---|---|---|
| Traitement | Séquentiel (token/token) | **Parallèle** (tous tokens) |
| Vanishing gradient | Problème majeur | **Résolu** (chemin longueur 1) |
| Complexité | O(seq_len) par layer | O(seq_len²) — attention globale |
| Dépendances longues | Difficile | **Facile** |
| Interprétabilité | Faible | **Partielle** (poids attention) |

---

## TP8 — Optimisation & Régularisation (Transfer Learning CIFAR-10)

> `TP/8_Lab_OptimRegu.ipynb` *(à compléter)*

### Objectif
Appliquer toutes les techniques d'optimisation et régularisation vues en cours sur un vrai problème de vision, avec transfer learning depuis ImageNet.

### Dataset
- **CIFAR-10** : sous-échantillon stratifié → **5 000 images train** (500/classe) + **1 000 images test** (100/classe)
- Normalisation ImageNet : `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- Resize 32px → 224px (format ResNet)

### Data Augmentation (train uniquement)
```python
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224, padding=8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])
```

### Parties et progression

**Partie 1 — Baseline (feature extraction)**
```python
# Geler tout ResNet18 sauf la tête fc
model.fc = nn.Linear(512, 10)
for p in model.parameters(): p.requires_grad = False
for p in model.fc.parameters(): p.requires_grad = True
# → ~5 130 paramètres entraînables sur 11 177 514 (0.05%)
# SGD(lr=1e-2, momentum=0.9), 8 epochs
```

**Partie 2 — Schedulers de Learning Rate**
- `ReduceLROnPlateau(patience=2, factor=0.5)` → profil en escalier, déclenché sur plateau
- `CosineAnnealingLR(T_max=8)` → profil continu cosinusoïdal
- Stratégie 1 préférée quand le nombre d'epochs n'est pas connu à l'avance

**Partie 3 — Early Stopping**
```python
# Conserver les poids du meilleur modèle
best_weights = copy.deepcopy(model.state_dict())
# Arrêter si val_acc ne s'améliore pas pendant patience=3 epochs
# Restaurer best_weights à la fin (pas le dernier epoch)
```

**Partie 4 — Régularisation**
```python
# Tête avec Dropout
model.fc = nn.Sequential(
    nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(),
    nn.Dropout(0.5), nn.Linear(256, 10)
)
# Weight Decay dans l'optimiseur
optimizer = optim.SGD(..., weight_decay=1e-4)
# ResNet18 contient déjà 16 couches BatchNorm2d (régularisation implicite)
```

**Partie 5 — Fine-tuning complet avec LR différencié**
```python
# Dégeler tout + LR différencié par groupe
optimizer = optim.SGD([
    {'params': backbone_params, 'lr': 1e-4},  # faible : ne pas écraser ImageNet
    {'params': head_params,     'lr': 1e-3}   # plus grand : tête aléatoire
], momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=12)
```

**Partie 6 — ResNet18 vs EfficientNet-B0**
```python
# EfficientNet-B0 : remplacer model.classifier
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 10))
```

| Modèle | Paramètres | Top-1 ImageNet |
|---|---|---|
| ResNet18 | ~11.2M | 69.8% |
| EfficientNet-B0 | **~5.3M** | **77.7%** |

**Bonus — Grid Search vs Random Search**
```python
# Grid Search : 3 lr × 3 dropout = 9 combinaisons (produit cartésien)
lr_values = [1e-1, 1e-2, 1e-3]
dropout_values = [0.2, 0.4, 0.6]

# Random Search : 9 trials, distributions continues
lr = 10 ** np.random.uniform(-3, -1)   # log-uniforme
dropout = np.random.uniform(0.1, 0.7)
```
- Random Search : log-uniforme pour lr car les ordres de grandeur comptent (1e-3 vs 1e-2 ≠ 0.001 vs 0.002)
- Visualisation : heatmap (grid) + scatter log-scale (random)

### Fonctions réutilisables à connaître
- `train_one_epoch` / `evaluate` : briques de base
- `run_training_basic` → `run_training_with_scheduler` → `run_training_full` : évolution progressive
- `plot_history(histories, labels)` : comparaison côte-à-côte de plusieurs entraînements

---

## TP10 — Export, Inférence et Déploiement

> `TP/10_Lab_ExInfDepl.ipynb` *(à compléter)*

### Objectif
Préparer les modèles TP8 (ResNet18 + EfficientNet-B0) pour la production : formats portables, optimisation mémoire, benchmark, API REST.

### Prérequis
Fichiers issus du TP8 :
- `models/resnet18_cifar10.pth`
- `models/efficientnet_b0_cifar10.pth`

### Partie 1 — Rechargement et sérialisation
```python
# Charger les poids (map_location='cpu' : indépendant du matériel d'entraînement)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
# map_location='cpu' essentiel si le modèle a été entraîné sur GPU
```

### Partie 2 — TorchScript
```python
# trace : adapté CNN/MLP (flux fixe, pas de branches)
traced = torch.jit.trace(model, dummy)   # dummy = torch.randn(1,3,224,224)
traced.save('model_traced.pt')

# script : nécessaire si forward() contient if/for (RNN, LSTM, Transformer)
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')
```

**Démonstration trace vs script** :
```python
class ModelAvecBranche(nn.Module):
    def forward(self, x, use_relu: bool = True):
        if use_relu: return F.relu(self.fc(x))
        else:        return self.fc(x)
# trace fige la branche observée → impossible de passer use_relu=False
# script capture les deux chemins
```
→ Dans ce cours : utiliser `script` pour les RNN et LSTM (TP5/TP6), `trace` pour CNN (TP4).

### Partie 3 — Export ONNX
```python
torch.onnx.export(
    model, dummy, 'model.onnx',
    export_params=True,
    opset_version=17,
    input_names=['input'], output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},   # batch variable en production
        'output': {0: 'batch_size'}
    }
)
# Vérification : onnx.checker.check_model(onnx.load('model.onnx'))
# Inférence : onnxruntime.InferenceSession (sans PyTorch)
```

### Partie 4 — Quantization dynamique
```python
model_q = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},       # cibler les couches fc (Conv2d non supporté en dynamique)
    dtype=torch.qint8  # float32 → int8 : 4× moins de mémoire
)
# Perte d'accuracy généralement < 1%
```

| Format | Taille | Comportement |
|---|---|---|
| Base (.pth) | 100% | Référence |
| TorchScript (.pt) | ~100% | Même poids, graphe compilé |
| Quantizé (.pth) | **~25–40%** | int8, ratio ≠ 4× exact (couches non-Linear conservées en float32) |

### Partie 5 — Benchmark 6 versions
**2 modèles × 3 formats** (base / TorchScript / quantizé) comparés sur :
- Taille (Mo)
- Latence (ms, batch=1, 100 répétitions, warmup 20)
- Accuracy sur test CIFAR-10

```python
def mesure_latence(model, input_tensor, n=200):
    # Warmup (20 passes) puis mesure sur n passes
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n): _ = model(input_tensor)
    return (time.time() - t0) / n * 1000  # ms
```

### Partie 6 — Inférence sur images externes
```python
def preprocess_image(image_path=None, pil_image=None):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),    # recadrage carré
        transforms.ToTensor(),          # [0,255] → [0,1]
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    return transform(img.convert('RGB')).unsqueeze(0)  # (1,3,224,224)

def predict(model, tensor, top_k=3):
    probs = torch.softmax(model(tensor), dim=1)[0]
    top_val, top_idx = torch.topk(probs, k=top_k)
    return [(classes[i], v.item()) for i, v in zip(top_idx, top_val)]
```

### Bonus — Mini API FastAPI
```python
app = FastAPI(title="CIFAR-10 Classifier")
model = torch.jit.load("resnet18_traced.pt")  # chargé une seule fois au démarrage
model.eval()

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # 1) Validation MIME (image/*)
    # 2) PIL.Image.open + convert('RGB')
    # 3) preprocess → tensor (1,3,224,224)
    # 4) inference torch.no_grad()
    # 5) softmax → top-3
    return PredictionResponse(classe=..., confiance=..., top3=[...])

@app.get("/health")
def health():
    return {"status": "ok", "modele": "ResNet18 CIFAR-10"}
# Lancement : uvicorn api:app --reload --port 8000
# Docs      : http://localhost:8000/docs
```

### Choix de format selon le scénario (Q6)
| Scénario | Format recommandé |
|---|---|
| Serveur Linux Python, faible latence | Base PyTorch ou TorchScript |
| Application mobile Android | TorchScript (LibTorch) |
| Service C++ haute performance | TorchScript `.pt` |
| Raspberry Pi (faible RAM) | Quantizé int8 |
| Interopérabilité multi-frameworks | ONNX |

---

## Fil conducteur des TP

```
TP1  Neurone from scratch (1 neurone, 2 features, Iris)
  ↓ ajouter des couches
TP2  MLP multiclasse (données tabulaires, OneHot, DataLoader)
  ↓ passer aux images
TP3  MLP sur images (MNIST, flatten, 97%) → limite : pas d'invariance
  ↓ résoudre avec CNN
TP4  CNN sur images (convolutions, pooling, plus robuste aux translations)
  ↓ passer aux séquences
TP5  RNN (IMDB, embeddings, seq2seq) → limite : vanishing gradient
  ↓ résoudre avec LSTM
TP6  LSTM + BiLSTM (portes, mémoire long terme, +10pp vs RNN)
  ↓ éliminer la récurrence
TP7  Transformer (attention parallèle, positional encoding, visualisation)
  ↓ bien entraîner le modèle
TP8  Optimisation & régularisation (schedulers, early stopping, dropout, fine-tuning)
  ↓ déployer le modèle
TP10 Export & déploiement (TorchScript, ONNX, quantization, FastAPI)
```
