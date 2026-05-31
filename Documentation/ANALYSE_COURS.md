# Analyse du Cours — Deep Learning

---

## Partie I — Fondations et CNN

> Source : `Cours/partie1_fondations-et-cnn.pdf` — 110 pages

### Structure du cours

| Section | Pages | Thème |
|---|---|---|
| Introduction au DL | 3–15 | Définition, historique, causes de la renaissance |
| Types d'apprentissage | 16–34 | Supervisé/non-supervisé/RL, boucle d'entraînement, métriques |
| Neurones artificiels | 35–66 | Perceptron, MLP, fonctions d'activation |
| PyTorch | 67–91 | Tenseurs, Autograd, boucle d'entraînement, cas pratiques |
| CNN | 92–110 | Convolution, pooling, architectures historiques |

---

### Concepts clés couverts

#### Fondations

- Théorème d'approximation universelle
- ML classique vs. Deep Learning (feature engineering vs. représentations apprises)
- Les 3 causes de la renaissance DL :
  - **Calcul** (GPU/TPU)
  - **Big Data** (ImageNet, Common Crawl)
  - **Algorithmes** (ReLU, BatchNorm, Dropout, Adam)

#### Neurone & MLP

- Structure : somme pondérée `z = w⊤x + b` → activation `ŷ = f(z)`
- Problème XOR : limite du perceptron, résolu par couche cachée + ReLU
- Fonctions d'activation :
  - **Sigmoid** — sortie binaire uniquement ; vanishing gradient en couches cachées
  - **Tanh** — sortie centrée en 0 ; utilisée dans les LSTM
  - **ReLU** — couches cachées par défaut ; gradient = 1 dans la partie active
  - **Leaky ReLU** — variante évitant le dying ReLU (pente `α` dans la partie négative)
  - **Softmax** — sortie multiclasse ; avec température pour distillation/génération
- Propagation avant, fonction de coût (MSE pour régression, Cross-Entropy pour classification)
- Rétropropagation par règle de la chaîne : `∂L/∂W[l] = δ[l] · (a[l-1])⊤`
- Descente de gradient : Batch GD, SGD, **Mini-batch** (recommandé : taille 32/64/128)
- Learning rate : démarrer à `η = 0.001` avec Adam, utiliser un scheduler

#### CNN

- Motivation : explosion des paramètres du MLP sur images, perte de structure spatiale, absence d'invariance aux translations
- Opération de convolution : filtre glissant (kernel), feature map, stride, padding
  - Taille de sortie : `out = (W - K + 2P) / S + 1`
- Max Pooling / Average Pooling : réduction de résolution, invariance aux petites translations
- Extraction hiérarchique des features : bords → textures → formes → concepts
- Architectures historiques :

| Modèle | Année | Innovation |
|---|---|---|
| **LeNet-5** | 1998 | Premier CNN moderne, MNIST |
| **AlexNet** | 2012 | ReLU + Dropout + entraînement GPU |
| **VGG** | 2014 | Convolutions 3×3 empilées, transfert learning |
| **ResNet** | 2015 | Skip connections, réseaux de 100+ couches |

#### Pratique PyTorch

- Tenseurs et Autograd : calcul automatique de gradients via `loss.backward()`
- API clés : `nn.Linear`, `nn.Conv2d`, `nn.MaxPool2d`, `nn.Softmax`, `nn.CrossEntropyLoss`
- Boucle d'entraînement : `zero_grad()` → forward → loss → `backward()` → `step()`
- Cas pratiques : dataset dermatologique (5 classes), MNIST (10 classes, 70 000 images)

---

### Points pédagogiques notables

- Approche **théorie → code → expérimentation** : chaque concept est ancré dans un notebook PyTorch
- Vanishing gradient expliqué dès l'introduction (Sigmoid vs. ReLU)
- Softmax avec température : distillation de connaissance (Hinton et al., 2015), génération de texte créative
- ResNet skip connections présentées comme solution directe au vanishing gradient sur réseaux profonds
- Distinction claire train / validation / test et diagnostic overfitting/underfitting via les courbes de loss

---

## Partie II — RNN, LSTM et Transformeurs

> Source : `Cours/partie2_rnn-lstm-transformeurs.pdf` — 78 pages

### Structure du cours

| Section | Pages | Thème |
|---|---|---|
| RNN vanille | 2–33 | Modélisation séquentielle, cellule RNN, BPTT, topologies, word embeddings |
| LSTM & GRU | 34–46 | Mécanismes de portes, comparaison, RNN profonds |
| Transformeurs & Attention | 47–78 | Self-attention, multi-head, encodeur-décodeur, BERT/GPT |

---

### Concepts clés couverts

#### Réseaux de neurones récurrents (RNN)

- **Motivation** : MLP et CNN incapables de traiter des séquences de longueur variable, sans mémoire temporelle ni notion d'ordre
- **4 propriétés idéales** d'un modèle séquentiel : longueur variable, dépendances longues, conservation de l'ordre, partage des paramètres (weight sharing)
- **Cellule RNN vanille** : `h(t) = tanh(W·h(t-1) + U·x(t) + b)` — les poids W, U, V sont partagés à chaque pas
  - Lien formel avec les **systèmes dynamiques** : règle d'évolution stationnaire
  - Limite : traitement fondamentalement séquentiel, pas de parallélisation
- **Pourquoi tanh** : sortie dans [-1, 1], symétrique, sans biais positif systématique (contrairement à Sigmoid)
- **Graphe de calcul déroulé** : le RNN vu comme un réseau profond sur T pas avec poids partagés
- **Topologies** :
  - One-to-many : génération de texte, image captioning
  - Many-to-one : classification de sentiment
  - Many-to-many synchrone : classification de trames vidéo
  - Many-to-many asynchrone (seq2seq) : traduction, QA
- **Architecture seq2seq** : encodeur produit un résumé sémantique C → décodeur génère la sortie ; C utilisable comme état initial, entrée à chaque pas, ou les deux
- **RNN bidirectionnel** : traitement dans les deux sens ; contexte gauche + droit pour chaque token ; inutilisable pour la génération temps réel (nécessite la séquence complète)
- **Word embeddings** : vecteur dense appris par mot, mots proches en sens → vecteurs proches ; `nn.Embedding(vocab_size, embed_dim)`

#### Vanishing & Exploding Gradient

- **Origine** : multiplication de W par lui-même T fois → valeurs propres élevées à la puissance t
  - `|λ| < 1` → disparition (0.95⁵⁰ ≈ 0.005)
  - `|λ| > 1` → explosion (1.15⁵⁰ ≈ 117)
- **Solution à l'explosion** : Gradient Clipping — norme globale (`nn.utils.clip_grad_norm_`) ou clipping par composante
- **Solution au vanishing** : LSTM et GRU (addition au lieu de multiplication → gradient préservé)
- **BPTT** (Backpropagation Through Time) : déroulage complet sur T pas ; **Truncated BPTT** : segmentation en blocs de taille k, compromis mémoire/qualité du gradient

#### LSTM et GRU

- **LSTM** (Hochreiter & Schmidhuber, 1997) : deux états `h(t)` (court terme) et `c(t)` (long terme), trois portes :
  - **Forget gate** `ft = σ(...)` : ce qu'on efface de c(t-1)
  - **Input gate** `it = σ(...)` + candidat `c̃(t) = tanh(...)` : nouvelle information à stocker
  - **Output gate** `ot = σ(...)` : quelle partie de c(t) exposer dans h(t)
  - Mise à jour : `c(t) = ft ⊙ c(t-1) + it ⊙ c̃(t)` — addition ≠ multiplication → gradient préservé
- **GRU** (Cho et al., 2014) : un seul état h(t), deux portes seulement
  - **Reset gate** `rt` : quel passé ignorer pour le candidat
  - **Update gate** `zt` : dosage entre mémoire passée et nouvelle info — `h(t) = (1-zt)⊙h(t-1) + zt⊙h̃(t)`

| Modèle | États | Portes | Paramètres | Dép. longues | Vitesse |
|---|---|---|---|---|---|
| **RNN** | h | 0 | peu | faible | rapide |
| **LSTM** | h, c | 3 | beaucoup | forte | lente |
| **GRU** | h | 2 | moyen | forte | moyenne |

- **RNN profonds** : empilement de couches LSTM/GRU, sortie d'une couche → entrée de la suivante ; `nn.LSTM(..., num_layers=3, dropout=0.3)`
- Tendance : LSTM et GRU largement supplantés par les Transformeurs en NLP, mais restent pertinents pour les **séries temporelles**

#### Transformeurs et mécanisme d'attention

- **Motivation** — "Attention is All You Need" (Vaswani et al., NIPS 2017) :
  - Avant : attention input-output uniquement, toujours combinée à une récurrence
  - Transformer : attention input-input, input-output et output-output simultanément, **aucune récurrence**, parallélisation totale
- **Scaled Dot-Product Attention** :
  - Triplet (Q, K, V) : Query = ce qu'on cherche, Key = ce que chaque token offre, Value = contenu à récupérer
  - Formule : `Attention(Q,K,V) = Softmax(QK⊤ / √dk) · V`
  - Analogie : moteur de recherche soft vs. dictionnaire Python hard
- **Multi-Head Attention** : h têtes en parallèle, chacune avec ses propres W_Q, W_K, W_V ; sorties concaténées puis projetées par W_O
  - Chaque tête se spécialise sur une relation différente (sujet/verbe, nom/adjectif, coréférence…)
  - Coût identique à une seule tête grâce à la réduction de dimension : `dk = dmodel / h = 512 / 8 = 64`
- **Encodage positionnel** : vecteurs sin/cos ajoutés aux embeddings pour réintroduire l'ordre (sans quoi la self-attention est un sac de mots)
- **Architecture encodeur-décodeur** :
  - Encodeur (×6) : Multi-Head Self-Attention + FFN (fc + ReLU + fc), connexions résiduelles + LayerNorm
  - Décodeur (×6) : Masked Self-Attention + Cross-Attention (Q du décodeur, K/V de l'encodeur) + FFN
  - Génération **autorégressive** : token produit → réinjecté comme entrée suivante, jusqu'au token `<EOS>`
- **Trois types d'attention** :
  - Input-Input (encodeur) : représentation contextuelle bidirectionnelle
  - Output-Output masquée (décodeur) : prédiction causale, ne voit pas les tokens futurs
  - Input-Output / cross-attention (décodeur) : consultation de toute la séquence source
- **Avantages sur le RNN** : parallélisation totale sur GPU + chemin de gradient de longueur 1 entre deux tokens quelconques (fin du vanishing gradient sur longues dépendances)

#### BERT vs GPT et panorama

| Modèle | Architecture | Direction | Usage principal |
|---|---|---|---|
| **BERT** (Google, 2018) | Encodeur seul | Bidirectionnelle | Compréhension (classification, NER, QA) |
| **GPT** (OpenAI, 2018) | Décodeur seul | Causale (gauche→droite) | Génération (texte, code, dialogue) |
| **T5** (Google, 2019) | Enc. + Déc. | Les deux | Universel |

- Applications modernes : NLP (ChatGPT, Claude, DeepL), Vision (ViT, DALL-E, DETR), Audio (Whisper, MusicGen), Sciences (AlphaFold 2, Codex)

#### Pratique PyTorch

- `nn.RNN`, `nn.LSTM`, `nn.GRU` avec `batch_first=True`
- `nn.Embedding(vocab_size, embed_dim, padding_idx=0)`
- `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)` + `nn.TransformerEncoder`
- Gradient clipping : `nn.utils.clip_grad_norm_(params, max_norm=1.0)`
- Cas pratique : **analyse de sentiment IMDB** (25 000 critiques, classification binaire positif/négatif)

---

### Points pédagogiques notables

- Progression logique : RNN vanille → limites (vanishing, mémoire bornée) → LSTM/GRU → limites (séquentialité) → Transformeur
- La limite de capacité du vecteur `h` (taille fixe) est expliquée comme la motivation directe de l'attention (pas de goulot d'étranglement)
- L'analogie dictionnaire hard vs. soft (mémoire associative) rend le mécanisme Q/K/V très intuitif
- La spécialisation des têtes d'attention est visualisée sur un exemple concret ("Le chat... dort")
- BERT et GPT présentés comme deux visions complémentaires du même Transformer (compréhension vs. génération)

---

## Partie III — Optimisation et Régularisation

> Source : `Cours/partie3_optimisation-regularisationv2.pdf` — 84 pages

### Structure du cours

| Section | Pages | Thème |
|---|---|---|
| Algorithmes d'optimisation | 8–11 | Variantes de la descente de gradient |
| Stratégies d'optimisation | 12–56 | Hyperparamètres, initialisation, normalisation |
| Régularisation | 57–83 | Data augmentation, finetuning, early stopping, dropout |

---

### Concepts clés couverts

#### Algorithmes d'optimisation

Sept variantes de la descente de gradient sont présentées, du plus simple au plus avancé :

1. **Batch GD** — gradient sur tout le dataset ; stable mais très lent
2. **SGD** (stochastique) — gradient sur un seul exemple ; rapide mais bruité
3. **SGD + Momentum** — accumule un vecteur de vitesse pour amortir les oscillations : `v ← αv - ε∇J` ; `θ ← θ + v`
4. **Nesterov Accelerated Gradient (NAG)** — calcule le gradient *après* avoir appliqué le momentum (look-ahead) ; converge plus vite que le momentum classique
5. **Adagrad** — adapte le taux d'apprentissage par paramètre selon l'historique des gradients ; bon pour données creuses, mais taux décroît trop vite
6. **RMSprop** — corrige Adagrad avec une moyenne glissante des gradients² ; stabilise l'apprentissage
7. **Adam** — combine momentum + RMSprop : estimations du 1er et 2e moment des gradients ; **algorithme par défaut en pratique**

#### Stratégies d'optimisation

**1. Choix des hyperparamètres**

Trois méthodes pour fixer les hyperparamètres statiques (architecture, dropout, filtres…) :

- **Horaire d'entraînement** (training schedule) : valeurs prédéfinies par époque, surtout pour le learning rate
  - *Stratégie 1 (par paliers)* : réduire `ε ← ε · β` quand `J(θ)` atteint un plateau → courbe en escalier
  - *Stratégie 2 (continue)* : décroissance à chaque époque selon `εt = ε0 / (1 + β·t)` ou `ε0 · exp(-β·t)`
  - Stratégie 1 préférée car elle approxime mieux la relation non-linéaire entre décroissance de `J` et de `ε`
  - Règle pratique : `ε0` = plus grande valeur ne faisant pas diverger `J(θ)` ; `β ∈ {0.1, 0.2, 0.5}`

- **Recherche en grille** (grid search) : produit cartésien de valeurs sur échelle logarithmique ; exhaustif mais redondant et coûteux
- **Recherche aléatoire** (random search) : échantillonnage de distributions par hyperparamètre ; moins redondant, arrêtable à tout moment (*anytime*), reproductibilité via seed

**2. Initialisation des paramètres**

- Biais : initialisés à **0** par convention
- Poids — deux familles selon la fonction d'activation :
  - **Glorot / Xavier** : pour Sigmoid et Tanh — normalise la variance selon `ni` (entrées) et `no` (sorties)
  - **Kaiming / He** : pour ReLU — compense le facteur ½ introduit par la rectification
- **Initialisation constante interdite** : tous les poids identiques → mises à jour identiques → brisure de symétrie impossible, le réseau reste bloqué
- **Pré-entraînement** (fine-tuning) : initialiser les poids cibles depuis un modèle source entraîné ; fonctionne si `Dom(source) ≈ Dom(cible)` ; devient un fine-tuning avec peu de données cibles

**3. Normalisation**

- **Min-Max** : ramène `x` dans `[0, 1]` ; sensible aux outliers ; uniquement si `x` est borné (ex. images `[0, 255] → [0, 1]`)
- **Z-score** : moyenne = 0, variance = 1 ; calculé sur le train set, appliqué uniformément ; le plus courant
- **Batch Normalization (BN)** : normalise les activations *à l'intérieur* du réseau, par mini-batch ; réduit la dépendance à l'initialisation, permet des learning rates plus élevés, agit comme régulariseur léger

#### Régularisation

Objectif : réduire l'erreur de test, quitte à augmenter légèrement l'erreur d'entraînement. Conseil général : **mieux vaut un réseau plus grand bien régularisé qu'un petit réseau non régularisé**.

**1. Data Augmentation**

- Meilleur régularisateur : plus de données — la data augmentation en crée artificiellement
- Transformations courantes pour la vision : rotation, translation, flip horizontal/vertical, recadrage, changement d'échelle, intensité lumineuse
- Précautions : éviter de changer le sens (ex. 6 → 9 par rotation), estimer les plages valides (ex. angle ±10°)
- Effectué sur CPU en parallèle avec la lecture disque ; pour les rotations coûteuses : `crop → rotation → crop`
- API PyTorch (`torchvision.transforms`) : `ColorJitter`, `RandomCrop`, `RandomHorizontalFlip`, `RandomResizedCrop`, `RandomRotation`, `RandomErasing` (résistance aux occlusions)

**2. Fine-tuning**

- Réutiliser un modèle pré-entraîné sur une tâche source pour initialiser un modèle cible
- Permet de transférer des représentations apprises sur grandes quantités de données vers des domaines avec peu de données

**3. Early Stopping**

- Conserver les paramètres `θ*` qui minimisent la perte de **validation** pendant l'entraînement
- Après toutes les epochs, retourner `θ*`
- Peut se voir comme la sélection de l'hyperparamètre "nombre d'epochs"
- Contrôle la capacité effective du réseau sans modifier l'architecture
- Deux stratégies post-entraînement : (A) ré-entraîner depuis zéro avec le nombre d'epochs identifié, (B) poursuivre depuis `θ*` un nombre fixe d'epochs supplémentaires
- Très utilisé : transparent, aucun changement d'architecture ni de fonction objective

**4. Dropout**

- Désactive aléatoirement et temporairement des neurones à chaque exemple (multiplication par 0)
- Peut se voir comme un **ensemble exponentiel de modèles** avec weight sharing partiel
- Surtout appliqué aux couches **fully connected** (moins aux couches CNN d'extraction de features)
- À l'inférence, deux stratégies : (A) moyenne sur 10–20 masques, (B) conserver tous les neurones pondérés par `p` (stratégie B utilisée par PyTorch, qui pondère par `1/(1-p)` à l'entraînement)
- Variante : **DropConnect** — supprime des connexions (poids) plutôt que des neurones

---

### Points pédagogiques notables

- La progression **algorithme → stratégie → régularisation** suit l'ordre naturel du diagnostic d'un modèle sous-performant
- L'initialisation est présentée comme souvent négligée mais critique : un mauvais init peut bloquer complètement l'optimisation (exemple : sigmoid saturée dès le départ)
- La recherche aléatoire est présentée comme supérieure à la grille car elle explore mieux l'espace (chaque essai change *tous* les hyperparamètres simultanément)
- Dropout présenté comme un peu moins populaire qu'avant, supplanté par d'autres formes de régularisation (BatchNorm, data augmentation) dans les architectures modernes
- Combinaison recommandée en pratique : plusieurs régularisateurs simultanément (data aug + early stopping + dropout)

---

## Partie IV — Déploiement, Éthique et Tendances

> Source : `Cours/partie4_deploiement-ethique-projetv2.pdf` — 48 pages

### Structure du cours

| Section | Pages | Thème |
|---|---|---|
| Du lab à la production | 3–11 | Le fossé notebook/prod, MLOps, versioning |
| Portabilité et export | 12–16 | Sérialisation, TorchScript, ONNX |
| Optimisation de l'inférence | 17–19 | Quantization, pruning, distillation |
| API et conteneurisation | 20–27 | FastAPI, Docker, Docker Compose |
| CI/CD et monitoring | 28–35 | Pipeline ML, MLflow, data drift, plateformes |
| Éthique et biais | 37–48 | Biais, fairness, explicabilité, EU AI Act |

---

### Concepts clés couverts

#### Du laboratoire à la production

- **Le fossé notebook / production** : dans un notebook les données sont fixes, l'environnement contrôlé, les erreurs sans conséquence ; en production les données arrivent en continu, la réponse doit être rapide et les erreurs ont un coût métier
- **Les 3 environnements** :
  - **Dev** — expérimentation locale, notebooks
  - **Staging** — copie contrôlée de la production pour tester ; tout déploiement en prod doit y être validé
  - **Production** — utilisateurs réels, zone critique ; le notebook n'est **jamais** une source directe de déploiement
- **MLOps** (Machine Learning + DevOps, ~2018) : ensemble des pratiques pour déployer, surveiller et maintenir des modèles ML en production
  - Quatre objectifs : **Reproductibilité**, **Automatisation**, **Traçabilité**, **Surveillance**
  - Cycle de vie : Données → Train → Évaluer → Déployer → Monitorer → Réentraîner
- **Versioning ML** : le code seul (Git) ne suffit pas — un modèle dépend aussi du dataset, des hyperparamètres, de la graine aléatoire et des versions de bibliothèques

| Artefact | Outil |
|---|---|
| Code | Git |
| Données & pipelines | DVC |
| Expériences & modèles | MLflow |

#### Portabilité et export

- **Sérialisation** : sauvegarder uniquement les poids (`state_dict`) est plus robuste que sauvegarder le modèle complet (dépendance à la classe Python et aux chemins de fichiers)
  ```python
  torch.save(model.state_dict(), "model.pth")
  model.load_state_dict(torch.load("model.pth", map_location="cpu"))
  model.eval()
  ```
- **TorchScript** : compile un modèle PyTorch en format exécutable sans Python (C++, mobile, embarqué)
  - `trace` : enregistre les opérations observées sur un exemple d'entrée — idéal CNN/MLP, flux fixe
  - `script` : analyse le code Python directement — nécessaire si le modèle contient des `if`/`for` (RNN, LSTM)
- **ONNX** : format ouvert interopérable entre frameworks (PyTorch, TensorFlow, JAX…) et moteurs d'inférence (ONNX Runtime, TensorRT, CoreML) ; export via `torch.onnx.export` avec `dynamic_axes` pour accepter des batchs de tailles variables en production

#### Optimisation de l'inférence

Compromis inévitable entre performance statistique et contraintes système (latence, mémoire, coût GPU, consommation énergétique) :

- **Quantization** : réduire la précision des poids de `float32` à `int8` → 4× moins de mémoire, inférence CPU plus rapide, perte de précision contrôlée
  ```python
  model_q = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
  ```
- **Pruning** : supprimer les poids peu utiles → réseau plus sparse, moins de paramètres
- **Knowledge Distillation** : entraîner un petit modèle (*student*) à imiter les *soft labels* d'un grand modèle (*teacher*) → modèle léger avec performances proches du teacher

#### API REST et conteneurisation

- **API REST avec FastAPI** : interface standardisée entre le modèle et les clients (web, mobile)
  - Routes types : `POST /predict`, `GET /health`, `GET /model/info`
  - Validation automatique des entrées avec **Pydantic** (`BaseModel`, `@validator`)
  - Documentation Swagger auto-générée sur `localhost:8000/docs`
  - Gestion des erreurs HTTP : 200 (succès), 400 (requête invalide), 422 (validation échouée), 500 (erreur serveur)
  - Règle : ne jamais faire confiance aux données reçues — valider la longueur, le format, les valeurs vides
- **Docker** : emballe code + dépendances + configuration dans une image portable
  - Vocabulaire : **Image** (modèle figé), **Conteneur** (instance en cours), **Dockerfile** (recette de build)
  - Même image → même comportement en local, staging et production
- **Docker Compose** : orchestre plusieurs services simultanément (API + Redis + PostgreSQL + logs) via un seul fichier `docker-compose.yml`

#### CI/CD et monitoring

- **Pipeline CI/CD pour le ML** (exemple GitHub Actions) : push → tests unitaires (`pytest`) → entraînement → évaluation → validation du seuil de performance → build Docker → deploy staging → deploy prod
  - Spécificité ML : on ne déploie pas si le code tourne — il faut aussi que les **métriques** passent le seuil
- **MLflow tracking** : enregistre paramètres, métriques par époque, artefacts et fichiers de poids ; **Model Registry** pour promouvoir une version vers la production ; `mlflow ui` pour comparer visuellement les runs
- **Monitoring en production** — deux types de dérive :
  - **Data drift** : la distribution des données d'entrée change (nouveaux comportements, nouveaux mots…)
  - **Model drift** : la relation entre entrées et cible évolue (nouvelles fraudes, nouveau contexte économique)
  - Détection avec **Evidently** : compare `train_df` et `production_df`, génère un rapport HTML de drift
- **Plateformes de déploiement** :
  - Prototype rapide : Hugging Face Spaces, Streamlit Cloud, Gradio
  - Production managée : AWS SageMaker, Google Vertex AI, Azure ML
  - Production contrôlée (DIY) : VPS + Docker + Kubernetes

- **Cas fil rouge** : LSTM entraîné → sérialisation → TorchScript → API FastAPI → conteneurisation Docker → tests → monitoring

#### Éthique, biais et explicabilité

- **Responsabilité** : un modèle déployé influence des décisions réelles (crédit, recrutement, diagnostic médical, liberté conditionnelle) — le data scientist est responsable de l'impact, pas seulement de la précision
- **Exemples documentés** :
  - COMPAS (2016) : algorithme de récidive biaisé contre les personnes noires
  - Amazon Hiring (2018) : outil de recrutement discriminant les femmes, abandonné en interne
  - GPT-3 (2020) : textes stéréotypés selon le groupe démographique
- **5 piliers de l'IA éthique** (Commission Européenne, OCDE, Microsoft) : Transparence, Équité, Responsabilité, Vie privée, Robustesse
- **Types de biais** :
  - **Représentation** : données non uniformes entre groupes (ImageNet : 78 % de personnes blanches)
  - **Mesure** : variable proxy imparfaite (prédire la récidive via le code postal = encoder l'origine ethnique)
  - **Agrégation** : modèle entraîné sur population mixte, peu précis sur sous-groupes (diabète : moins précis sur les femmes)
  - **Confirmation** : le modèle renforce les décisions passées déjà biaisées
- **Features proxy** : supprimer une variable sensible ne suffit pas — code postal → revenu, prénom → genre/origine, université → niveau social
- **Métriques de fairness** :
  - *Demographic parity* : taux de prédictions positives identique entre groupes
  - *Equal opportunity* : taux de vrais positifs identique
  - *Calibration* : scores de confiance reflètent les vraies probabilités par groupe
  - **Théorème d'impossibilité** (Chouldechova, 2017) : impossible de satisfaire simultanément les 3 définitions sauf si les groupes ont les mêmes taux de base — tout choix de métrique d'équité est un **choix politique**
- **Outils** : **Fairlearn** (mesure des disparités par groupe, réduction via `ExponentiatedGradient`) ; **Evidently** (drift) ; **XAI** (explicabilité — boîtes noires vs. droit à l'explication du RGPD Art. 22)
- **EU AI Act (2024)** — première loi mondiale encadrant l'IA par niveau de risque :

| Niveau | Exemples | Obligations |
|---|---|---|
| Risque inacceptable | Notation sociale, manipulation subliminale | **Interdit** |
| Haut risque | Justice, RH, médecine, infrastructures critiques | Conformité, traçabilité, supervision humaine |
| Risque limité | Chatbots | Obligation de transparence |
| Risque minimal | Jeux vidéo, filtres spam | Aucune |

  - Sanctions : jusqu'à **35 M€ ou 7 % du CA mondial**

- **Checklist éthique** (4 phases) : avant entraînement (datasheet, sous-représentation, features proxy, consentement), pendant (métriques par sous-groupe), avant déploiement (Model Card, cas déconseillés, mécanisme de recours, supervision humaine), en production (monitoring des biais, processus de réentraînement)

---

### Points pédagogiques notables

- Le fossé notebook → production est présenté dès le départ comme le vrai défi industriel, changeant complètement la perspective du cours
- TorchScript `trace` vs `script` : distinction pratique claire — utiliser `script` dès qu'il y a une logique conditionnelle (RNN, LSTM)
- Le pipeline CI/CD ML intègre explicitement une **validation de seuil de performance** avant tout déploiement, pas seulement des tests de code
- Le théorème d'impossibilité de fairness est présenté honnêtement : il n'existe pas de solution technique parfaite, le choix de métrique est un choix éthique et politique
- La checklist éthique est opérationnelle et structurée en 4 phases, directement applicable à un projet réel
