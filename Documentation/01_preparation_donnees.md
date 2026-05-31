# 01 — Préparation des données

---

## Objectif global et logique du pipeline

### Ce que l'on cherche à construire

L'objectif final de ce projet est de construire un **modèle de classification d'intentions** capable de lire n'importe quel tweet du dataset (ou tout nouveau tweet entrant) et de lui assigner automatiquement une classe d'intention parmi **12 possibles (flux client)** ou **9 possibles (flux entreprise)**. Ce modèle, une fois entraîné, sera la brique centrale d'un système de support client automatisé.

Le problème de départ est le suivant : le dataset Twitter contient environ 3 millions de tweets, et **aucun n'est labellisé**. Il n'existe pas de colonne "intention" dans le CSV. Pour entraîner un modèle supervisé, il faut obligatoirement des exemples labellisés — des paires (tweet, intention). Le pipeline de préparation des données a pour seul but de produire ces exemples.

## Choix du dataset

### Raison 1 — Projet personnel

Le dataset Twitter s'intègre dans un projet de génération de tweets à partir d'autres tweets. La classification d'intentions est la brique amont obligatoire de ce pipeline :

```
Tweet client entrant
        ↓
Classification d'intention    ← ce que ce projet construit
        ↓
Génération de réponse conditionnée à l'intention
        ↓
Tweet de support généré
```

Sans un classifieur fiable en entrée, le générateur produit des réponses génériques non pertinentes. Ce projet académique construit donc directement une composante réutilisable du projet personnel.

### Raison 2 — Réalisme industriel

Le cours (Partie IV) insiste sur "le fossé notebook/production" — en production les données arrivent bruitées, sans labels, en continu. Le dataset Twitter illustre exactement ce contexte : 3M tweets, aucun label d'intention, mentions parasites, emojis, sarcasme, abréviations. Choisir Bitext (propre, pré-labellisé) aurait évité précisément les difficultés que l'on rencontre en situation réelle.

---

### Pourquoi on n'entraîne pas sur les 3 millions de tweets

Entraîner directement sur 3 millions de tweets sans labels est impossible pour un modèle supervisé. Il faudrait labelliser manuellement les 3 millions de tweets. L'approche retenue est différente : on labellise **automatiquement** un sous-ensemble représentatif via KMeans + Label Propagation FAISS (aucune annotation manuelle), on s'assure que ces labels sont de bonne qualité, et on entraîne le modèle sur ce sous-ensemble.

La qualité des labels est plus importante que la quantité. Après équilibrage (N_MAX=30 000 par classe), le dataset réel produit par `01b_data_preparation_v5.ipynb` contient **~192 000 tweets client** (12 classes) et **~223 000 tweets entreprise** (9 classes). Ces chiffres sont inférieurs au maximum théorique (360K / 270K) car plusieurs classes sont naturellement sous-représentées dans le dataset original.

### Pourquoi environ 360 000 tweets labellisés (flux client)

Le chiffre n'est pas arbitraire — il résulte de deux décisions techniques :

Premièrement, la **Label Propagation avec seuil cosine = 0.85** filtre les tweets ambigus. Sur les ~1.2 millions de tweets non échantillonnés par flux, seuls ceux dont l'embedding est suffisamment proche (cosine ≥ 0.85) d'un tweet déjà labellisé reçoivent un label. Les tweets qui se trouvent à la frontière entre deux classes, ou qui n'appartiennent clairement à aucune, sont rejetés. Ce filtrage garantit que chaque label propagé est fiable.

Deuxièmement, l'**équilibrage à N_MAX = 30 000 par classe** évite que le modèle soit biaisé vers les classes les plus fréquentes. Sans équilibrage, une classe représentant 60% du dataset dominerait l'entraînement et le modèle apprendrait à prédire cette classe par défaut. Avec **12 classes client** et 30 000 tweets maximum par classe, le dataset client contient au plus 12 × 30 000 = **360 000 tweets**. Le flux entreprise (9 classes) contient au plus 9 × 30 000 = **270 000 tweets**. Certaines classes naturellement moins représentées dans le dataset original restent en dessous de ce plafond — ce qui signifie que les chiffres réels sont inférieurs à ces maximums théoriques.

### Ce que le modèle fait après l'entraînement

Une fois entraîné sur les tweets labellisés (jusqu'à ~360 000 pour le flux client, ~270 000 pour le flux entreprise), le modèle a appris à reconnaître les patterns linguistiques associés à chaque intention. Il peut alors être appliqué à **n'importe quel tweet**, qu'il fasse partie des 3 millions du dataset original ou qu'il soit un nouveau tweet entrant en production.

Le modèle n'a pas besoin d'avoir vu un tweet spécifique pendant l'entraînement pour le classifier correctement. Il généralise à partir des patterns appris : si un tweet contient des formulations similaires à celles des tweets d'entraînement d'une classe donnée, le modèle l'assignera à cette classe. C'est le principe fondamental de la généralisation en apprentissage automatique.

En pratique, après entraînement, on peut passer les 3 millions de tweets du dataset dans le modèle pour obtenir une classification complète du corpus. On peut aussi déployer le modèle en production pour classifier des tweets en temps réel, ce qui constitue l'objectif final décrit dans la Partie IV du cours.

---

## Pipeline de préparation

Le principe fondamental : **data-driven avant tout**. Les classes sont découvertes dans les données via clustering, pas définies a priori. Sur un dataset pré-labellisé comme Bitext, les classes sont connues → on peut écrire les LFs directement. Sur 3M tweets bruts sans labels, les classes réelles sont inconnues → le clustering est obligatoire avant toute décision de labellisation.

Deux flux traités en parallèle : **df_client** (`inbound=True`) et **df_company** (`inbound=False`).

```
CSV twcs.csv (3M tweets)
        ↓
0. Nettoyage structurel (× 2 flux)
   → supprimer NaN sur text/tweet_id
   → supprimer doublons tweet_id
   → séparer inbound=True (df_client) | inbound=False (df_company)
        ↓
1. EDA + Clustering (× 2 flux — AVANT de définir les classes)
   sentence-transformers → embeddings 384d
   MiniBatchKMeans + elbow method (k=4..15)
   → inspecter centroïdes → nommer les classes
        ↓
2. Labellisation directe par KMeans (× 2 flux)
   Même échantillon 50K + même seed qu'en 01a
   → cluster IDs cohérents avec CLIENT_LABELS / COMPANY_LABELS
        ↓
3. Label Propagation KNN (× 2 flux)
   sentence-transformers + FAISS IndexFlatIP
   K=5, seuil cosine=0.85
   → propager labels vers les tweets non échantillonnés
        ↓
4. Équilibrage + Split stratifié 70/15/15 (× 2 flux)
        ↓
5. Nettoyage textuel + Vocabulaire + Tokenisation + Padding
        ↓
6. Jointure conversationnelle
   df_full_company.in_response_to_tweet_id ↔ df_full_client.tweet_id
   → paired_data.pkl : (tokens_client, tokens_company, label_client, label_company)
   → split 70/15/15 stratifié sur label_client
        ↓
6 artefacts sauvegardés dans data/ :
  vocab.pkl              → vocabulaire 15 000 tokens (commun aux 2 flux)
  label_maps.json        → CLIENT_LABELS (10 classes) + COMPANY_LABELS (9 classes)
  client_data.pkl        → modèles client seul  (OUTPUT_DIM=10) — indices vocab custom
  company_data.pkl       → modèles company seul (OUTPUT_DIM=9)
  paired_data.pkl        → modèle multi-task    (OUTPUT_DIM=12 + 9)
  client_text_data.pkl   → textes nettoyés (strings brutes) pour DistilBERT — le tokenizer
                           HuggingFace ne peut pas utiliser notre vocab custom
```

---

## Étape 0 — Nettoyage structurel

### Justification théorique

Avant tout traitement, vérifier l'intégrité du CSV. Des NaN dans `text` ou des doublons sur `tweet_id` corrompent silencieusement le pipeline sans erreur explicite.

Les tweets sont séparés en deux flux : `inbound=True` (messages clients) et `inbound=False` (réponses entreprises). Les deux sont traités car les tweets entreprise servent à la fois comme **signal de labellisation** (enrichissement contextuel des tweets client ambigus via le join `tweet_id`) et comme **dataset de classification indépendant**.

### Opérations (× 2 flux)
- Supprimer les lignes où `text` est NaN
- Supprimer les doublons sur `tweet_id`
- Séparer `inbound=True` → `df_client` | `inbound=False` → `df_company`

---

## Étape 1 — EDA + Clustering (data-driven)

### Justification théorique

Le cours (Partie I) distingue apprentissage supervisé et **non-supervisé**. Le clustering est la brique non-supervisée qui révèle la structure naturelle des données — utilisée ici comme **étape exploratoire obligatoire** avant toute décision de labellisation.

**Pourquoi clustering avant de définir les classes ?**

Sur Twitter non labellisé, écrire des LFs sans avoir regardé les données produit des règles arbitraires qui ne capturent pas ce qui existe vraiment. Le Snorkel ne peut pas converger si les LFs ne couvrent pas les patterns réels. Le clustering permet d'ancrer les classes sur la réalité du dataset.

### Méthode

`sentence-transformers` (`all-MiniLM-L6-v2`) encode chaque tweet en vecteur 384d — représentation sémantique dense, pas lexicale. `MiniBatchKMeans` regroupe les vecteurs. L'**elbow method** (k=4..12) détermine le nombre optimal de clusters. On inspecte les 5 tweets les plus proches du centroïde de chaque cluster pour nommer les classes réelles.

Appliqué sur les **deux flux** : client ET entreprise.

### Limites de l'elbow method et tentative avec le Silhouette Score

L'elbow method présente une limite importante sur des embeddings de texte : la courbe d'inertie descend de façon quasi-linéaire sans coude franc. Les tweets étant représentés dans un espace sémantique dense à 384 dimensions, les clusters se chevauchent naturellement et il n'existe pas de séparation géométrique nette comme sur des données tabulaires. Sur nos deux courbes (client et entreprise), aucun coude clair n'apparaît entre K=4 et K=12, ce qui rend le choix de K par elbow subjectif.

Le **Silhouette Score** a été calculé pour tenter d'objectiver ce choix. Il mesure pour chaque point à quel point il est proche de son propre cluster (cohésion) et éloigné des clusters voisins (séparation). Un score proche de 1 indique des clusters bien séparés. Un score proche de 0 indique des points à la frontière entre clusters.

**Résultats obtenus :**

| K | Silhouette Client | Silhouette Entreprise |
|---|---|---|
| 4  | 0.0174 | **0.0410** |
| 5  | 0.0257 | 0.0408 |
| 6  | 0.0243 | 0.0336 |
| 7  | 0.0274 | 0.0324 |
| 8  | 0.0290 | 0.0235 |
| 9  | 0.0330 | 0.0244 |
| 10 | 0.0358 | 0.0177 |
| 11 | 0.0358 | 0.0160 |
| 12 | **0.0369** | 0.0157 |

**Interprétation :** tous les scores se situent entre 0.016 et 0.041 — quasiment nuls. Un clustering exploitable donnerait des scores supérieurs à 0.5. De plus, les deux flux donnent des signaux contradictoires : pour les tweets client, le score monte jusqu'à K=12 sans se stabiliser ; pour les tweets entreprise, il descend dès K=4. Ces comportements opposés confirment que le silhouette score ne peut pas identifier un K optimal fiable sur ce dataset.

**Pourquoi les tweets de support client ne se clustérisent pas bien :**

`all-MiniLM-L6-v2` encode la sémantique générale d'une phrase — il capture le fait qu'un tweet exprime une plainte, mais pas le type précis de plainte. "My phone isn't working" et "my delivery is late" sont deux intentions distinctes, mais leurs embeddings sont proches car les deux expriment frustration et demande d'aide. Les intentions de support client partagent un registre émotionnel commun qui efface les frontières sémantiques dans l'espace d'embedding.

**Décision finale : exploration manuelle des K par inspection des centroïdes.**

En l'absence d'un critère automatique fiable, plusieurs valeurs de K ont été testées manuellement pour chaque flux. Pour chaque K, les 5 tweets les plus proches du centroïde de chaque cluster ont été lus et évalués. La règle de sélection était simple : un K est retenu si et seulement si **tous** ses clusters présentent des thèmes distincts, nommables et non redondants — c'est-à-dire qu'un humain peut donner un nom différent à chaque cluster sans hésiter.

### Méthode complémentaire — Évaluation par mots-clés dominants par cluster

**Contexte** : l'inspection des 5 tweets centroïdes donne un aperçu local mais peut être biaisée par des tweets atypiques. Une vue plus robuste consiste à extraire les **mots les plus fréquents dans l'ensemble du cluster** (après suppression des stopwords), ce qui révèle le vocabulaire caractéristique de chaque groupe.

**Principe** : pour chaque cluster, on concatène tous les tweets du cluster, on tokenise, on filtre les stopwords, puis on affiche le top-N des mots les plus fréquents. Un cluster avec un vocabulaire cohérent et thématiquement homogène (ex: `flight`, `delay`, `cancelled`, `airline`) est un bon cluster. Un cluster avec un vocabulaire dispersé (mélange de `refund`, `order`, `flight`, `wifi`) signale une classe fourre-tout à subdiviser.

**Pourquoi c'est plus fiable que le silhouette score** :
- Le silhouette score mesure la séparation géométrique dans l'espace d'embedding — peu interprétable et faible par construction sur des tweets courts
- Les mots-clés dominants mesurent la **cohérence sémantique humaine** du cluster — c'est exactement ce qui compte pour nommer les classes et les labelliser correctement

**Implémentation (à ajouter dans 01a)** :
```python
from collections import Counter
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

for c in range(K_CLIENT):
    mask = df_sample_client['cluster'] == c
    tweets = df_sample_client[mask]['text'].str.lower().str.cat(sep=' ')
    words = [w for w in tweets.split() if w not in stop_words and len(w) > 2]
    top_words = Counter(words).most_common(10)
    print(f'\n--- Cluster {c} ({mask.sum():,} tweets) — Top mots : {top_words}')
```

**Règle de décision** : si les top-10 mots d'un cluster partagent un thème clair → cluster valide. Si les mots-clés mélangent 2 thèmes distincts → K trop petit, augmenter. Si deux clusters ont des top mots très similaires → K trop grand, réduire.

---

### Exploration flux client — de K=8 à K=14

**K=8 (point de départ)**

Premier test raisonnable. Les 8 clusters produits étaient : `flight_complaint`, `tech_service_issue`, `other`, `no_response`, `ride_issue`, `general_complaint`, `delivery_complaint`, `device_software`. Tous avaient un thème lisible, mais plusieurs clusters semblaient mélanger des intentions distinctes (notamment `tech_service_issue` qui regroupait à la fois des plaintes téléphoniques et des problèmes de connexion internet).

**K=10 (amélioration nette)**

Deux clusters supplémentaires apparurent clairement :
- Un cluster de plaintes liées aux trains (`VirginTrains`, `LondonMidland`, `SouthernRailUK`) séparé des plaintes aériennes — deux intentions distinctes que K=8 fusionnait
- Un cluster de tweets en langue étrangère (espagnol, italien, portugais) — `non_english` — très homogène et nettement séparable du reste

Ces deux séparations étaient sémantiquement justifiées. K=10 constituait une amélioration réelle par rapport à K=8.

**K=12 (nouvelles intentions identifiées — retenu)**

Deux nouveaux clusters pertinents émergèrent :
- `billing_refund` : tweets concernant des remboursements, des frais incorrects, des demandes de crédit — une intention financière distincte absente de K=10
- `internet_outage` : plaintes spécifiques sur des coupures de connexion internet ou de réseau mobile — intention infrastructure distincte des autres `tech_service_issue`

Ces deux nouvelles classes correspondent à des **intentions client genuinement différentes** qui justifient un traitement différent en support client. K=12 fut retenu pour le flux client.

**K=14 (trop granulaire — rejeté)**

Deux nouveaux clusters apparurent : `food_quality` (plaintes sur la qualité de la nourriture) et `spotify_streaming` (problèmes de lecture audio Spotify). Les raisons de rejet diffèrent selon le cluster :

- `spotify_streaming` : trop spécifique à une marque. "Mon application Spotify ne charge pas" est une instance de `tech_service_issue` déjà existante — ce n'est pas une nouvelle intention client, c'est la même intention appliquée à un produit particulier.
- `food_quality` : rejeté pour une raison différente — trop proche sémantiquement de `food_delivery_complaint`. "Ma livraison est en retard" et "ma nourriture était froide" sont deux facettes de la même expérience client food, difficiles à distinguer en pratique dans des tweets courts. La frontière entre les deux serait trop floue pour être apprise de façon fiable.

K=14 fut rejeté.

**Conclusion flux client : K = 12**

Les 12 classes retenues pour `CLIENT_LABELS` :
`train_complaint`, `tech_general`, `other`, `no_response`, `ride_issue`, `general_complaint`, `food_delivery_complaint`, `ios_device_issue`, `flight_complaint`, `non_english`, `billing_refund`, `internet_outage`

---

### Exploration flux entreprise — de K=7 à K=11

**K=7 (insuffisant — rejeté)**

K=7 résolvait l'ambiguïté entre deux clusters de redirection vers DM qui se ressemblaient à K=8, mais au prix d'une perte : le cluster `non_english` (réponses aux tweets en langue étrangère) avait disparu, absorbé dans `general_ack`. Or `non_english` était l'un des clusters les plus homogènes et distinctifs observés. Perdre une information claire pour en résoudre une floue n'est pas un bon compromis. K=7 fut rejeté.

**K=8 (point de référence)**

Les 8 clusters produits étaient : `general_ack`, `order_apology_dm`, `redirect_dm`, `flight_apology`, `non_english`, `redirect_dm_help`, `tech_support_dm`, `product_info`. Le problème identifié : `redirect_dm` et `redirect_dm_help` avaient des centroïdes aux contenus très proches — les deux invitaient le client à envoyer un DM, avec une légère variation de formulation. Cette ambiguïté rendait le frontière entre ces deux classes peu fiable.

**K=9 (amélioration nette — retenu)**

Un nouveau cluster `ios_dm_support` apparut, séparant les réponses spécifiques aux problèmes iOS/iPhone (instructions de mise à jour, réinitialisation, paramètres) des réponses `tech_support_dm` génériques. Cette séparation était sémantiquement justifiée. Surtout, K=9 résolut l'ambiguïté du flux `redirect_dm` : la subdivision supplémentaire permit à KMeans de distinguer les redirections vers DM avec un contexte de commande (`redirect_dm_order`) des redirections génériques (`redirect_dm`). Les trois clusters de redirection à K=9 avaient des centroïdes clairement distinguables. K=9 fut retenu pour le flux entreprise.

**K=11 (trop redondant — rejeté)**

Les deux clusters supplémentaires à K=11 étaient des variations de **style rédactionnel** au sein d'une même intention (ton formel vs. informel pour les accusations, longueur courte vs. longue pour les redirections) plutôt que des intentions distinctes. Des variations de style ne constituent pas des classes d'intention — elles constituent du bruit pour le classifieur. K=11 fut rejeté.

**Conclusion flux entreprise : K = 9**

Les 9 classes retenues pour `COMPANY_LABELS` :
`general_ack`, `order_apology_dm`, `redirect_dm`, `flight_apology`, `non_english`, `ios_dm_support`, `tech_support_dm`, `store_feedback_or_product_info`, `redirect_dm_help`

---

### Historique des itérations — versions v3 à v7 (`01a_clustering_v7.ipynb`)

Cette section documente toutes les décisions techniques testées et les raisons de chaque choix ou abandon, dans l'ordre chronologique.

#### 1. Labeling Functions (LFs) + Snorkel — abandonné

Première approche tentée : des règles par mots-clés (`if 'flight' in tweet → flight_complaint`) agrégées par Snorkel LabelModel. Abandonné car la confiance moyenne obtenue sur les tweets entreprise n'atteignait que **0.210** (à peine au-dessus du hasard 1/8 = 0.125). Les LFs avaient une couverture trop faible et des conflits entre elles. Snorkel ne peut pas converger sans signal fiable.

**Décision : remplacé par labellisation KMeans directe** (même seed → mêmes IDs de clusters → labels cohérents).

#### 2. Modèle d'embedding — évolution vers twitter-roberta-base

Trois modèles testés dans l'ordre :

| Modèle | Résultat |
|---|---|
| `all-MiniLM-L6-v2` | Premier modèle testé. Encode la sémantique générale (384d). Silhouette scores faibles (~0.02-0.04). Capture le ton (frustration) mais pas le domaine (flight vs. delivery). |
| `BAAI/bge-base-en-v1.5` | Testé pour améliorer la séparation. Erreur 401 sur le serveur (accès restreint). Abandonné. |
| `cardiffnlp/twitter-roberta-base` | **Retenu.** Pré-entraîné sur 58M tweets — mieux adapté au registre Twitter. Nécessite un wrapping manuel (`models.Transformer` + `models.Pooling`) car pas natif SentenceTransformer. Paramètre `model_kwargs={'use_safetensors': True}` ajouté pour contourner CVE-2025-32434 sur torch < 2.6. |

#### 3. BERTopic — testé, abandonné au profit de KMeans

BERTopic (UMAP + HDBSCAN) a été implémenté dans `01a_clustering_v5.ipynb`. Avantages : pas besoin de fixer K, marque les tweets ambigus `-1`, extrait automatiquement les mots-clés par topic. Finalement abandonné au profit de KMeans pour une raison pratique : **KMeans est plus interprétable et contrôlable** — on comprend directement pourquoi un tweet appartient à un cluster (distance au centroïde), ce qui facilite la validation manuelle des classes.

#### 4. Taille de l'échantillon — SAMPLE_CLUSTER

- `50 000` testé en premier : rapide, mais les clusters sont moins stables et certaines classes rares (train_complaint, non_english) sont sous-représentées.
- `100 000` retenu : meilleure représentativité des classes rares, clusters plus stables. Compromis acceptable avec le temps de calcul GPU.

#### 5. Taille de batch embedding — BATCH_SIZE_EMBED = 256

Valeur conservatrice pour éviter les OOM (Out Of Memory) GPU. Une valeur plus grande (512) serait possible mais 256 a été retenu pour la stabilité sur le serveur Scaleway.

#### 6. Filtre langue — langdetect, anglais uniquement

Ajouté en v4 après avoir constaté que des tweets japonais, coréens, arabes formaient des clusters parasites basés uniquement sur la langue (pas l'intention). Implémentation :
```python
from langdetect import detect, LangDetectException
df_sample_client = df_sample_client[df_sample_client['language'] == 'en'].reset_index(drop=True)
emb_client = emb_client[keep_idx]  # réalignement obligatoire
```
Appliqué aux deux flux (client ET company). La classe `non_english` dans les labels 01b reste valide — elle est définie par le contenu linguistique, pas par le clustering.

#### 7. Nettoyage textuel du clustering (Bloc 3)

Opérations appliquées avant embedding :
- Minuscules
- Suppression `@mentions` (bruit identitaire, pas sémantique)
- Suppression URLs
- Suppression ponctuation (sauf `#hashtags`)
- Suppression stopwords anglais (sauf liste blanche : `no`, `yes`, `ok`, `why`, `how`...)

#### 8. n_init — stabilité KMeans

`n_init=5` utilisé en standard. Testé à `n_init=15` sur la dernière itération — **résultats identiques** (même distribution de tweets par cluster, mêmes mots-clés). Confirmé : le problème n'est pas la convergence de KMeans mais la structure du dataset. `n_init=5` retenu pour limiter le temps de calcul.

#### 9. Itérations K supplémentaires (v6-v7) — K=11, K=15

Après les résultats K=12, des tests supplémentaires ont été menés pour tenter d'obtenir des clusters plus nets :

- **K=11** : fusionne flight + train en un cluster transport. Toujours 5-6 clusters propres, le reste est bruit général.
- **K=15** : éclate les clusters propres en doublons (deux clusters transport, deux clusters iOS). Aucun nouveau cluster sémantiquement justifié.
- **K=12 retenu** : les 4-5 clusters propres sont stables (transport, iOS, thanks, account). Le reste (`general_complaint`) est un artefact de la densité sémantique du dataset.

#### 10. Conclusion définitive — limites du clustering sur ce dataset

Après v3→v7 et K=8→15, la structure réelle du dataset présente **4-5 intentions géométriquement séparables** par KMeans :
- `transport_complaint` (flight + train)
- `ios_device_issue`
- `customer_thanks`
- `account_service`
- (partiellement) `general_complaint`

Les autres classes (`billing_refund`, `internet_outage`, `ride_issue`, `food_delivery_complaint`) existent avec un signal lexical réel mais ne forment pas de clusters géométriquement nets. Elles sont valides comme classes d'entraînement pour DistilBERT — qui lit chaque mot et non des distances cosines. Le clustering a servi à **découvrir et nommer** les classes. L'évaluation finale de leur apprenabilité appartient au modèle de classification (`04_model_distilbert.ipynb`).

---

### Conséquence sur OUTPUT_DIM

La décision K=12 / K=9 entraîne une modification directe du paramètre `OUTPUT_DIM` dans tous les modèles :
- Flux client → `OUTPUT_DIM = 10` (après suppression de `short_ack` en Bloc 6 de 01b_v8)
- Flux entreprise → `OUTPUT_DIM = 9`


### Output attendu
- K=12 retenu pour le flux client après exploration itérative (K=8→10→12→14)
- K=9 retenu pour le flux entreprise après exploration itérative (K=7→8→9→11)
- Dictionnaires `CLIENT_LABELS` (12 classes) / `COMPANY_LABELS` (9 classes) nommés sur la base des patterns observés dans chaque cluster à K final
- `OUTPUT_DIM = 12` pour les modèles entraînés sur le flux client

---

## Étape 2 — Labellisation directe par KMeans

### Pourquoi on a abandonné les LFs et Snorkel

Une première approche avait été tentée avec des **Labeling Functions (LFs)** — des règles par mots-clés (`if 'flight' in tweet → flight_complaint`) agrégées par Snorkel LabelModel. Cette approche a été abandonnée pour une raison mesurée :

- Avec 8 classes, la **confiance aléatoire de base** = 1/8 = **0.125**
- La confiance moyenne obtenue par Snorkel sur les tweets entreprise = **0.210**
- Soit à peine +0.085 au-dessus du hasard — le LabelModel n'avait quasiment aucun signal utile

**Cause** : les LFs par mots-clés avaient une couverture trop faible (la plupart des tweets ne contenaient pas les phrases exactes cherchées) et des conflits entre LFs (un même tweet matchant plusieurs classes différentes). Snorkel ne peut pas converger sans signal fiable.

### Solution retenue : assignation KMeans directe

Les clusters KMeans produits à l'étape 1 constituent déjà une labellisation sémantique fiable. Plutôt que de retranscrire ces clusters en mots-clés, on utilise **directement les assignations KMeans** comme labels.

En reprenant le **même échantillon 50K** et le **même seed** qu'en 01a, les IDs de clusters correspondent exactement à ceux observés lors de l'inspection des centroïdes, et donc aux dictionnaires `CLIENT_LABELS` / `COMPANY_LABELS`.

```
KMeans.fit_predict(emb_50K, seed=42) → labels directs, confiance = 1.0
```

Les 50K tweets labellisés servent d'ancres pour la Label Propagation qui étend la couverture au reste du dataset.

---

## Étape 3 — Label Propagation (sentence-transformers + FAISS)

### Justification théorique

Les tweets non échantillonnés (~1M+ par flux) ne peuvent pas être annotés manuellement. On propage les labels des 50K ancres KMeans vers ces tweets incertains par **similarité sémantique**.

Ce principe est directement inspiré du mécanisme de **multi-head self-attention** (Cours II) :

```
Attention(Q, K, V) = Softmax(QK⊤ / √dk) · V
```

- **Query** = tweet incertain à labelliser
- **Keys** = tweets labellisés dans l'index FAISS
- **Values** = leurs labels
- **Similarité cosine** = équivalent du score d'attention

La différence : l'attention opère sur des tokens dans une séquence, la label propagation opère sur des tweets dans un espace d'embedding.

### Paramètres
| Paramètre | Valeur | Justification |
|---|---|---|
| `K` | 5 | Vote majoritaire robuste |
| `similarity_threshold` | 0.85 | N'accepter que les voisins vraiment proches |
| `batch_size_embed` | 512 | Optimal GPU sentence-transformers |
| `faiss_index` | `IndexFlatIP` | Produit scalaire sur vecteurs normalisés = cosine |

### Lien avec les TP
TP7 utilise `key_padding_mask` pour ignorer les tokens non pertinents — même logique que `similarity_threshold` : on n'accepte que les connexions fortes.

---

## Étape 4 — Pipeline standard (patterns TP5/TP6)

### Justification théorique

#### Nettoyage textuel
Supprimer le bruit non sémantique avant la tokenisation :
- `@mentions` → pas d'information d'intention
- URLs → bruit pur
- Emojis → convertir en texte descriptif (`emoji.demojize`)
- Caractères spéciaux, casse → normaliser en minuscules

#### Équilibrage et split

Sous-échantillonnage `N_MAX` par classe pour éviter que `CrossEntropyLoss` favorise les classes majoritaires. Split stratifié 70/15/15 avec `stratify=y` — le cours (Partie I) rappelle que la courbe de validation est la seule information fiable pour détecter l'overfitting.

#### Vocabulaire — construit uniquement sur le train set

Le cours (Partie II) est explicite : le vocabulaire se construit exclusivement sur le train set pour éviter le **data leakage**. Tokens inconnus du val/test → `<UNK>`.

```
<PAD> → indice 0  (padding_idx=0 → l'Embedding ne l'apprend pas)
<UNK> → indice 1  (tokens hors vocabulaire)
```

#### Padding et collate_fn

`pad_sequence` uniformise dynamiquement chaque batch à la longueur maximale du batch (pas du dataset). `MAX_LEN=64` : tweets font en moyenne 17 mots, P95 ≈ 40 mots → 64 couvre 99%+.

### Lien avec les TP

| Concept | TP | Pattern |
|---|---|---|
| `Counter` + `vocab.most_common(VOCAB_SIZE-2)` | TP5 | Vocabulaire sur train uniquement |
| `{'<PAD>': 0, '<UNK>': 1}` | TP5 | Tokens spéciaux |
| `tokenize(text, vocab, max_len)` | TP5 | Troncature + UNK |
| `collate_fn` + `pad_sequence(batch_first=True)` | TP5 | Padding dynamique |
| `DataLoader(shuffle=True/False)` | TP5/TP6 | Train shufflé, val/test non |
| `train_test_split(stratify=y, random_state=42)` | TP2 | Split reproductible |
| `nn.Embedding(padding_idx=0)` | TP6 | PAD non appris |
| `padding_mask = (x == 0)` | TP7 | Masque Transformer |

---

## Paramètres techniques

| Paramètre | Valeur | Justification |
|---|---|---|
| `SAMPLE_CLUSTER` | **200 000** (v8) | 4× plus grand qu'en v7 — meilleure représentativité des classes rares |
| `VOCAB_SIZE` | 15 000 | Tweets bruités, argot, abréviations |
| `MAX_LEN` | 64 | P95 ≈ 40 mots |
| `K` | **10** (v8) | Vote KNN plus robuste avec 200K anchors |
| `SIMILARITY_THRESHOLD` | 0.85 | Cosine — connexions fortes uniquement |
| `BATCH_SIZE_EMBED` | **256** (v8) | Conservateur pour roberta-base (768d) |
| `N_MAX` | **100 000** (v8) | Cap par classe — meilleure généralisation RNN/LSTM |
| `N_CLASSES` | **10** | Après suppression `short_ack` (courts acks sans signal sémantique) |
| `BATCH_SIZE` | 64 | Standard TP5/TP6 |
| `split` | 70/15/15 | Stratifié reproductible |
| `RANDOM_STATE` | 42 | Seed global unique |

---

## Étape 6 — Jointure conversationnelle (dataset paires)

### Justification

Dans le dataset Twitter, chaque tweet entreprise contient un champ `in_response_to_tweet_id` qui référence le `tweet_id` du tweet client auquel il répond. Cette structure conversationnelle est de l'information que les modèles séparés ignorent complètement.

Entraîner deux modèles indépendants revient à dire : "le type de réponse entreprise n'apprend rien sur l'intention client, et vice versa." C'est faux : une réponse `flight_apology` est un signal fort que le tweet client était un `flight_complaint`. Une réponse `ios_dm_support` indique presque certainement que le tweet client était un `ios_device_issue`. Cette co-occurrence est du signal gratuit.

L'approche visée est le **multi-task learning** : un encodeur partagé apprend une représentation commune des tweets de support, avec deux têtes de classification indépendantes (une pour le flux client, une pour le flux entreprise). La loss totale est la somme des deux losses.

`paired_data.pkl` rend cette approche possible pour n'importe quelle architecture (RNN, BiLSTM, Transformer, DistilBERT). Le choix de l'architecture concrète sera fait dans les notebooks modèles (`02`, `03`, `04`).

### Méthode de construction des paires

La jointure s'effectue sur `df_full_company.in_response_to_tweet_id == df_full_client.tweet_id` (inner join). Seules les paires où les **deux tweets sont labellisés** sont retenues. Les tweets sans réponse (client sans company correspondant) restent dans `client_data.pkl` et sont utilisés pour les modèles single-flux.

Le split est stratifié sur `label_client` (la variable la plus diverse avec 12 classes) pour garantir une représentation équilibrée dans train/val/test.

### Output
| Artefact | Contenu | Utilisé par |
|---|---|---|
| `client_data.pkl` | `(tokens_indices, label_client)` — indices vocab custom | RNN, BiLSTM, Transformer from scratch |
| `company_data.pkl` | `(tokens_indices, label_company)` | RNN, BiLSTM company |
| `paired_data.pkl` | `(tokens_client, tokens_company, label_client, label_company)` | Multi-task BiLSTM |
| `client_text_data.pkl` | `(text_str, label_client)` — textes nettoyés bruts | **DistilBERT** (tokenizer HuggingFace) |

> **Pourquoi deux formats pour le même texte ?** Les modèles `02_models_v1.ipynb` utilisent notre vocabulaire custom de 15 000 tokens (indices entiers). DistilBERT a son propre tokenizer WordPiece (~30 000 sous-mots) et ne peut pas utiliser des indices qui n'ont aucun sens pour lui. `client_text_data.pkl` sauvegarde donc les **strings nettoyées** avant tokenisation — DistilBERT fait lui-même la tokenisation dans le notebook `04`.

---

---

## Plan d'implémentation

**Fichiers cibles** : `notebooks/01a_clustering.ipynb` + `notebooks/01b_data_preparation_v2.ipynb`

**Blocs prévus** :

| # | Bloc | Ce qu'il fait | Input → Output |
|---|---|---|---|
| **01a** | | | |
| 1 | Config & imports | Constantes, seed, device, chemins | — → paramètres globaux |
| 2 | Chargement CSV | Lecture twcs.csv + séparation inbound | CSV → `df_client` + `df_company` |
| 3 | Nettoyage structurel | NaN + doublons `tweet_id` × 2 flux | df bruts → df propres |
| 4 | EDA | Distributions longueurs + mots fréquents × 2 | df → graphiques |
| 5 | Embeddings clustering | `sentence-transformers` sur 50K × 2 | df → `emb_client` + `emb_company` |
| 6 | Clustering client | Elbow + KMeans + inspection centroïdes | `emb_client` → groupes nommés |
| 7 | Clustering entreprise | Elbow + KMeans + inspection centroïdes | `emb_company` → groupes nommés |
| 8 | Sauvegarde | `df_client.pkl`, `df_company.pkl` | → `data/` |
| **01b** | | | |
| 1 | Config & imports | Constantes, seed, device | — → paramètres globaux |
| 2 | Chargement pickles | `df_client.pkl` + `df_company.pkl` | → df propres |
| 3 | Définition classes | `CLIENT_LABELS` + `COMPANY_LABELS` | → dictionnaires |
| 4 | Labels client | KMeans 50K sample (même seed) → labels directs | → `df_labeled_client` + `df_uncertain_client` |
| 5 | Labels company | KMeans 50K sample (même seed) → labels directs | → `df_labeled_company` + `df_uncertain_company` |
| 6 | Label Propagation | FAISS KNN vectorisé × 2 flux | → `df_full_client` + `df_full_company` |
| 7 | Équilibrage | Sous-échantillonnage `N_MAX` × 2 | → `df_bal_*` |
| 8 | Split stratifié | 70/15/15 `stratify=y` × 2 | → train/val/test × 2 |
| 9 | Nettoyage textuel | `@mentions`, URLs, `emoji.demojize`, casse | → textes propres |
| 10 | Vocabulaire | `Counter` sur train client → `vocab.pkl` | → `vocab` |
| 11 | Tokenisation | `tokenize()` pattern TP5 × 6 splits | → listes d'indices |
| 12 | Dataset + DataLoader | `IntentDataset` + `collate_fn` × 2 flux | → DataLoaders |
| 13 | Sauvegarde artefacts | `vocab.pkl`, `label_maps.json`, `*_data.pkl` | → `data/` |

**Patterns TP utilisés** : `Counter + vocab.most_common` (TP5), `{'<PAD>': 0, '<UNK>': 1}` (TP5), `tokenize()` (TP5), `collate_fn + pad_sequence` (TP5), `DataLoader shuffle` (TP5/TP6), `train_test_split stratify` (TP2)

---

---

### Implémentation réalisée

**Fichiers** :
- `notebooks/01a_clustering_v3.ipynb` — exploration + clustering (Blocs 1-7 + sauvegarde)
- `notebooks/01b_data_preparation_v5.ipynb` — KMeans direct + Label Propagation + pipeline + paires (Blocs 1-13)

**Blocs écrits** :
- 01a : Config, Chargement CSV, Nettoyage structurel, EDA, Embeddings, Clustering client (elbow + KMeans + inspection), Clustering entreprise (elbow + KMeans + inspection), Sauvegarde df_client/df_company
- 01b : Config, Chargement pickles, Définition classes, Labels client (KMeans direct, 50K sample, seed=42), Labels company (KMeans direct, 50K sample, seed=42), Label Propagation vectorisée (FAISS + numpy), Équilibrage N_MAX=30K, Split stratifié 70/15/15, Nettoyage textuel, Vocabulaire (TP5), Tokenisation (TP5), Dataset + DataLoader (TP5), Sauvegarde artefacts

**Écarts par rapport au plan initial** :
- Snorkel + LFs supprimés : remplacés par labellisation KMeans directe (même seed → même cluster IDs)
- Label Propagation entièrement vectorisée (numpy) : `df.iloc[valid_rows].copy()` en une seule opération
- Notebook renommé `01b_data_preparation_v2.ipynb` pour forcer le rechargement sur le serveur GPU

### Résultats / Observations

**Artefacts produits** (dans `data/`) :
- `vocab.pkl` — 15 000 tokens (construit sur train client uniquement)
- `label_maps.json` — **12 classes client + 9 classes company** (mis à jour après exploration K=12/K=9)
- `client_data.pkl` — splits train/val/test — indices vocab custom (RNN, BiLSTM, Transformer)
- `company_data.pkl` — splits train/val/test
- `client_text_data.pkl` — splits train/val/test — textes nettoyés bruts (DistilBERT)

**EDA — Longueur des tweets** :
- Client : distribution unimodale, pic autour de 20-25 mots, P95 = **39 mots**
- Entreprise : distribution plus irrégulière (double pic 15-25 mots), P95 = **36 mots**
- → `MAX_LEN=64` couvre 99%+ des deux flux avec une marge confortable

**Distribution après équilibrage (N_MAX=30 000)** :

Client (K=12 — exécution `01b_data_preparation_v5.ipynb`) :
| label | classe | count |
|---|---|---|
| 0 | train_complaint | 3 640 |
| 1 | tech_general | 23 688 |
| 2 | other | 30 000 |
| 3 | no_response | 30 000 |
| 4 | ride_issue | 20 265 |
| 5 | general_complaint | 18 217 |
| 6 | food_delivery_complaint | 11 935 |
| 7 | ios_device_issue | 9 409 |
| 8 | flight_complaint | 11 048 |
| 9 | non_english | 3 974 |
| 10 | billing_refund | 8 987 |
| 11 | internet_outage | 20 861 |

*Classes naturellement sous le cap (< 30 000) : `train_complaint` (3 640) et `non_english` (3 974) sont les moins représentées — à surveiller lors de l'entraînement (weighted loss potentiellement nécessaire).*

Entreprise (K=9 — exécution `01b_data_preparation_v5.ipynb`) :
| label | classe | count |
|---|---|---|
| 0 | general_ack | 30 000 |
| 1 | order_apology_dm | 30 000 |
| 2 | redirect_dm | 30 000 |
| 3 | flight_apology | 18 627 |
| 4 | non_english | 9 533 |
| 5 | ios_dm_support | 29 496 |
| 6 | tech_support_dm | 30 000 |
| 7 | store_feedback_or_product_info | 15 455 |
| 8 | redirect_dm_help | 30 000 |

**Dataset paires — Bloc 13 (`paired_data.pkl`)** :

70 692 paires conversationnelles trouvées (inner join sur `in_response_to_tweet_id`).
Split stratifié sur `label_client` : Train 49 484 / Val 10 604 / Test 10 604.

Distribution `label_client` dans les paires :
| label | classe | count |
|---|---|---|
| 0 | train_complaint | 465 |
| 1 | tech_general | 10 196 |
| 2 | other | 16 415 |
| 3 | no_response | 6 008 |
| 4 | ride_issue | 9 477 |
| 5 | general_complaint | 8 593 |
| 6 | food_delivery_complaint | 2 567 |
| 7 | ios_device_issue | 5 068 |
| 8 | flight_complaint | 1 414 |
| 9 | non_english | 1 147 |
| 10 | billing_refund | 2 864 |
| 11 | internet_outage | 6 478 |

Distribution `label_company` dans les paires :
| label | classe | count |
|---|---|---|
| 0 | general_ack | 6 171 |
| 1 | order_apology_dm | 5 716 |
| 2 | redirect_dm | 24 124 |
| 3 | flight_apology | 1 613 |
| 4 | non_english | 1 377 |
| 5 | ios_dm_support | 5 586 |
| 6 | tech_support_dm | 10 463 |
| 7 | store_feedback_or_product_info | 1 454 |
| 8 | redirect_dm_help | 8 188 |

*`redirect_dm` domine côté company (24 124 / 70 692 = 34%) — réponse générique la plus fréquente du support. `train_complaint` (465) et `non_english` (1 147) très sous-représentés côté client → weighted loss à prévoir pour la tête client du modèle multi-task.*

**Note** : lors de l'approche initiale avec LFs + Snorkel LabelModel, la confiance moyenne sur les tweets entreprise n'atteignait que **0.210** (à peine au-dessus du hasard 1/8 = 0.125). C'est ce résultat qui a motivé l'abandon des LFs au profit de la labellisation KMeans directe.

---

## Modification v6 — Correctif tokenisation (découvert lors de l'entraînement des architectures)

### Contexte — Ce qui a révélé le problème

Lors de l'entraînement des 3 architectures dans `02_models_v1.ipynb`, le **Transformer from scratch** produisait une accuracy de ~1.9% fixe pendant 5 époques, tandis que le BiLSTM atteignait ~54%. L'analyse des courbes d'entraînement montrait une loss à NaN (graphique vide) et une accuracy qui ne bougeait pas d'une époque à l'autre — signe que le modèle n'était jamais mis à jour.

### Cause racine — La chaîne de défaillance

Certains tweets du dataset contiennent uniquement des caractères non-latins : japonais, coréen, arabe, chinois. La fonction `clean_text` de `01b` applique le regex `[^a-zA-Z0-9\s_:]` qui supprime **tous** les caractères non-alphanumériques latins :

```
Tweet : "@AmazonHelp 出品者からの急なキャンセルで..." (japonais)
→ Après suppression @mention   : "出品者からの急なキャンセルで..."
→ Après regex                  : "                              " (espaces uniquement)
→ Après strip()                : ""  (chaîne vide)
→ tokenize("")                 : []  (liste vide)
→ collate_fn pad_sequence      : [0, 0, 0, ..., 0]  (séquence entièrement PAD)
```

Ce tweet entièrement constitué de `<PAD>` (indice 0) produit un `key_padding_mask = all True` dans le Transformer :

```python
padding_mask = (x == 0)  # → [True, True, True, ..., True] pour cette séquence
```

Dans `nn.MultiheadAttention`, un masque entièrement `True` signifie "ignore tous les tokens". Le calcul d'attention reçoit alors `-inf` partout dans les logits :

```
softmax([-inf, -inf, ..., -inf]) = exp(-inf) / Σexp(-inf) = 0 / 0 = NaN
```

Ce NaN se propage à la loss du batch entier. Avec le garde-fou `if torch.isnan(loss): continue` dans `train_epoch`, le batch est sauté — mais comme **chaque batch de 64 tweets contient statistiquement au moins un tweet non-latin**, tous les batchs sont sautés et le modèle n'est jamais entraîné.

### Pourquoi le BiLSTM n'était pas affecté

Le BiLSTM ne pose pas de masque d'attention. Une séquence all-PAD y reçoit des embeddings nuls (`padding_idx=0` → vecteur 0) en entrée. Le LSTM traite ces zéros et produit un hidden state valide (petites valeurs dues aux biais). Pas de division par zéro, pas de NaN. Le BiLSTM entraîne donc sur tous les batchs y compris ceux qui contiennent des tweets non-latins.

### Fix appliqué — `tokenize()` dans `01b_data_preparation_v6.ipynb`

Une seule ligne ajoutée dans `tokenize` :

```python
def tokenize(text, vocab, max_len=MAX_LEN):
    tokens = text.split()[:max_len]
    ids = [vocab.get(w, vocab['<UNK>']) for w in tokens]
    return ids if ids else [vocab['<UNK>']]  # tweet non-latin nettoyé à "" → un seul <UNK>
```

Un tweet japonais entièrement vidé par le nettoyage produit désormais `[1]` (indice de `<UNK>`) au lieu de `[]`. La séquence contient au moins un token non-PAD → `padding_mask` a au moins un `False` → `softmax` reçoit au moins un logit fini → plus de NaN.

**Pourquoi ne pas filtrer ces tweets** : ils appartiennent à la classe `non_english` (indice 9 pour le flux client). Les supprimer détruirait les données d'entraînement de cette classe. Avec le fallback `<UNK>`, le modèle apprend que "une séquence contenant uniquement des tokens inconnus → classe `non_english`" — ce qui est sémantiquement juste.

### Cellule de diagnostic ajoutée

Une cellule de diagnostic a été insérée dans `01b_data_preparation_v6.ipynb` juste après la tokenisation pour mesurer l'impact du problème :

```
split_name           total  | single_unk | short(2-3) | tronqués
client_train         ...    |    ?        |    ?        |    ?
...
```

Elle affiche pour chaque split : nombre de séquences avec fallback `<UNK>` unique (tweets non-latins), séquences très courtes, et séquences tronquées à `MAX_LEN=64`. À remplir après exécution de v6.

### Fichier mis à jour
`notebooks/01b_data_preparation_v6.ipynb` — à relancer depuis le **Bloc 8** (nettoyage textuel → tokenisation → artefacts). KMeans et FAISS ne changent pas.

### Cellule supplémentaire — `client_text_data.pkl`

Une cellule a également été ajoutée dans `01b_data_preparation_v6.ipynb` après le Bloc 12 (sauvegarde standard). Elle sauvegarde les textes nettoyés **avant tokenisation custom** :

```python
client_text_data = {
    'train': [(text, int(label)) for text, label in zip(client_train[0], client_train[1])],
    'val':   [(text, int(label)) for text, label in zip(client_val[0],   client_val[1])],
    'test':  [(text, int(label)) for text, label in zip(client_test[0],  client_test[1])],
}
```

`client_train[0]` contient les strings nettoyées (après `clean_text`, avant `tokenize`). Ce format est la seule entrée valide pour DistilBERT : son tokenizer WordPiece attend du texte brut, pas des indices entiers issus d'un vocabulaire arbitraire.

---

## Modification v7-v8 — Nouveau modèle d'embedding, nouveau mapping CLIENT_LABELS, paramètres augmentés

### Fichier : `01b_data_preparation_v8.ipynb`

---

### 1. Modèle d'embedding : `cardiffnlp/twitter-roberta-base`

**Pourquoi ce modèle ?**

`cardiffnlp/twitter-roberta-base` est le même modèle utilisé dans `01a_clustering_v7.ipynb` pour le clustering. Ce choix de cohérence est intentionnel : si 01a a identifié des clusters avec ce modèle, et qu'on utilise le même modèle dans 01b pour refaire KMeans, les frontières entre clusters restent dans le même espace vectoriel. Utiliser un modèle différent entre 01a et 01b reviendrait à labelliser avec une boussole différente de celle qui a servi à tracer la carte.

Ce modèle est pré-entraîné sur **58 millions de tweets** — il comprend nativement le registre Twitter : abréviations, argot, emojis transcrits, hashtags, mentions. `all-MiniLM-L6-v2` est un modèle généraliste entraîné sur du texte web standard. Il capture la sémantique générale d'une phrase (frustration, demande d'aide) mais pas les distinctions de domaine (plainte vol vs. plainte livraison). Sur des tweets de support client, twitter-roberta-base produit des embeddings plus discriminants.

**Conséquence technique** : embeddings 768d au lieu de 384d → `EMBED_DIM` est calculé dynamiquement via `word_embedding_model.get_word_embedding_dimension()` et injecté dans le reshape FAISS et l'index `IndexFlatIP`. Plus aucune valeur `384` hardcodée.

---

### 2. `SAMPLE_CLUSTER = 200 000` (était 50 000)

Avec 50K tweets échantillonnés pour KMeans, certaines classes rares (notamment `non_english` et `train_complaint`) n'avaient que quelques centaines d'exemples dans l'échantillon. KMeans instancie ses centroïdes sur ces exemples — si une classe a 200 exemples sur 50K, son centroïde est peu représentatif et sa frontière avec les classes voisines est floue.

Avec 200K tweets (soit 4× plus), chaque classe dispose d'un centroïde plus stable et les assignations label propagation sont de meilleure qualité. Le coût : temps d'embedding 4× plus long. Sur GPU Scaleway, ce surcoût est acceptable comparé au gain en qualité des labels.

---

### 3. `N_MAX = 100 000` (était 30 000)

Avec un meilleur modèle d'embedding et un meilleur clustering, la qualité des labels propagés est suffisante pour utiliser davantage de données par classe. Plus de données d'entraînement bénéficie directement aux architectures les moins puissantes (RNN, LSTM) qui ont besoin de beaucoup d'exemples pour généraliser. DistilBERT converge avec peu d'exemples, mais RNN nécessite plus de données pour apprendre des patterns stables.

Résultat attendu : jusqu'à **100 000 × 10 = 1 000 000 de tweets** client au total avant split (certaines classes resteront sous ce plafond si les données ne suffisent pas).

---

### 4. `K = 10` pour FAISS label propagation (était 5)

Le paramètre `K` contrôle le nombre de voisins consultés par FAISS pour chaque tweet incertain. Avec K=5, le vote majoritaire s'appuie sur 5 voisins — sensible si un voisin est mal labellisé. Avec K=10 et 200K anchors disponibles, il y a davantage de voisins de qualité dans le voisinage de chaque tweet → le vote est plus robuste. La combinaison K=10 + seuil cosine 0.85 rejette les voisins éloignés tout en consultant plus de voisins proches.

---

### 5. `max_seq_length=None` (était 128)

RoBERTa supporte des séquences jusqu'à 512 tokens. Les tweets font en moyenne 17-25 mots (P95 ≈ 40 tokens), bien en dessous de 128. `max_seq_length=None` laisse le modèle utiliser sa longueur native sans troncature artificielle — aucun tweet ne sera coupé, ce qui est souhaitable pour les cas limites.

---

### 6. Comment CLIENT_LABELS a été déterminé (Bloc 4b)

**Le problème** : chaque exécution de KMeans avec un modèle différent produit des IDs de clusters différents. L'ancien CLIENT_LABELS (écrit avec MiniLM) ne correspond plus aux clusters produits par twitter-roberta-base. `CLIENT_LABELS[0] = 'train_complaint'` ne veut plus rien dire si le cluster 0 du nouveau modèle contient des tweets de plainte iOS.

**La procédure** : après l'exécution du Bloc 4 (KMeans → `df_labeled_client`), le Bloc 4b exécute un diagnostic sur chaque cluster :

```python
for c in range(N_CLIENT):
    mask  = df_labeled_client['label'] == c
    textes = df_labeled_client[mask]['text']
    # top mots-clés (hors stopwords)
    # 3 tweets exemples
```

Ce diagnostic affiche pour chaque cluster (0-10) : les 8 mots-clés dominants et 3 tweets représentatifs. L'humain lit l'output et détermine le nom sémantique de chaque cluster.

**Résultats obtenus sur 01b_v7 avec roberta-base + K=11** :

| Cluster | Mots-clés / exemples observés | Nom assigné |
|---|---|---|
| C0 | phone, fix, service — O2, AsurionCares | `general_complaint` |
| C1 | delivery, sainsbury's, order not delivered | `food_delivery_complaint` |
| C2 | **que, por, pas, ich, pour** — tweets non-anglais | `non_english` |
| C3 | account, email, call, issue — Spectrum, Chase | `account_issue` |
| C4 | **iphone, ios, app, update, fix** | `ios_device_issue` |
| C5 | order, amazon, customer, billing | `billing_refund` |
| C6 | **"Thanks!", "done", "Yep."** — courts remerciements | `short_ack` → **supprimé** |
| C7 | fix, help, service, phone, AmericanAir | `service_complaint` |
| C8 | **flight, train, delayed**, British Airways | `transport_complaint` |
| C9 | phone, Apple, card, iOS | `tech_general` |
| C10 | customer service, account, OneDrive, Dropbox | `no_response` |

Le dictionnaire `CLIENT_LABELS` dans Bloc 3 reflète ce mapping. Il est mis à jour dans Bloc 6 (après suppression de C6) pour avoir les IDs finals continus 0-9.

---

### 7. Suppression de `short_ack` (cluster 6)

**Ce qu'est `short_ack`** : tweets très courts de type remerciement ou accusé de réception — "Thank you very much!", "Done.", "Yep. Full power cycle.", "Let me DM it over." Ces tweets n'expriment aucune **intention client** : ils ne demandent rien, ne se plaignent de rien, n'ont aucun contenu sémantique propre. Ils constituent du bruit conversationnel.

**Pourquoi les supprimer** : lors de l'entraînement DistilBERT en v6 (avec l'ancienne classe `other` qui jouait le même rôle), la précision de cette classe était de **0.44** (la pire du dataset, et inférieure au hasard à 1/11 ≈ 0.09 non, à 1/11 ≈ 0.09 non, à 1/11 → near-random). La matrice de confusion montrait que `other` attiraient des prédictions de toutes les autres classes — le modèle ne savait pas comment distinguer un remerciement d'une plainte iOS ou d'un problème de vol. Cette confusion dégradait les métriques globales.

**Mécanisme** : KMeans est lancé avec `n_clusters=11` (N_CLIENT=11 avant suppression) pour que `short_ack` soit bien isolé dans un cluster propre. Après `balance_dataset`, les tweets `label==6` sont filtrés, et les labels 7→10 sont remappés en 6→9 pour conserver des IDs continus. `N_CLIENT` est mis à jour à 10. Les modèles d'entraînement utilisent `OUTPUT_DIM=10`.

---

### 8. Split stratifié 70/15/15

**Pourquoi ces proportions ?**

- **70% train** : avec N_MAX=100K par classe et 10 classes, le train contient jusqu'à 700K tweets. Suffisant pour toutes les architectures, y compris RNN qui nécessite beaucoup d'exemples pour généraliser.
- **15% validation** : utilisé pendant l'entraînement pour surveiller l'overfitting et sauvegarder le meilleur modèle (`best_val_acc`). Le cours (Partie III) insiste : la courbe de validation est la seule information fiable pour détecter l'overfitting. Ne jamais utiliser le test pour choisir les hyperparamètres.
- **15% test** : évalué **une seule fois**, à la fin, après sélection du meilleur modèle sur la validation. C'est l'estimation honnête de la généralisation. Si on évaluait sur le test à chaque epoch, on introduirait du data leakage indirect (les hyperparamètres seraient implicitement optimisés pour le test).

**Pourquoi stratifié** (`stratify=y`) : garantit que chaque split contient la même proportion de chaque classe. Sans stratification, le split aléatoire peut laisser une classe rare (ex: `non_english`) absente du test set → métriques non représentatives.

---

### Résumé des paramètres v8

| Paramètre | v6/v7 | v8 | Raison du changement |
|---|---|---|---|
| `SAMPLE_CLUSTER` | 50 000 | **200 000** | Meilleure représentativité des classes rares |
| `N_MAX` | 30 000 | **100 000** | Plus de données → meilleure généralisation |
| `K` (FAISS) | 5 | **10** | Vote majoritaire plus robuste avec plus d'anchors |
| `n_init` (KMeans) | 5 | **15** | Centroïdes initiaux de meilleure qualité |
| `max_seq_length` | 128 | **None** | Aucune troncature (tweets < 128 tokens) |
| Modèle embedding | `all-MiniLM-L6-v2` | **`twitter-roberta-base`** | Cohérence avec 01a + adapté Twitter |
| `N_CLIENT` (final) | 11 | **10** | Suppression `short_ack` (cluster 6) |
| `OUTPUT_DIM` models | 11 | **10** | Conséquence directe |
