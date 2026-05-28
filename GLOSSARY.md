# Glossary

Définitions de tous les termes techniques. Niveau bac/licence. Mises à jour itératives.

## Systèmes dynamiques

### Système dynamique
Ensemble d'états qui évoluent selon une règle déterministe ou stochastique :

```
x_{t+1} = F(x_t)
```

où `x_t` est l'état au temps `t`, et `F` la fonction de transition.

**Exemple** : pendule, marché financier, activité cérébrale, **réseau neuronal pendant training**.

### Espace des phases
Espace mathématique abstrait où **chaque point représente un état complet du système**. Trajectoire = série de points visités au fil du temps.

**En NN** : l'espace des phases peut être l'espace des poids (très haut-dimensionnel) ou un espace réduit (PCA, t-SNE, UMAP) sur les activations latentes.

### Attracteur
Configuration vers laquelle le système converge depuis un voisinage. Types :
- **Point fixe** : un seul état stable (ex. : un NN convergé vers une solution)
- **Cycle limite** : oscillation périodique stable
- **Attracteur étrange** : structure fractale (ex. : Lorenz attractor)

**En NN** : différents seeds convergent souvent vers le même attracteur en fonction → suggère que loss landscape a peu de minima profonds (Frankle 2020).

### Bassin d'attraction
Ensemble des conditions initiales qui aboutissent au même attracteur. Si bassin large → robustesse. Bassin étroit → sensibilité aux conditions initiales.

### Exposant de Lyapunov λ
Mesure quantitative de **divergence de trajectoires proches** :

```
δ(t) ≈ δ_0 · e^{λt}
```

où `δ_0` = écart initial entre 2 trajectoires, `δ(t)` = écart au temps `t`.

- **λ > 0** → trajectoires s'écartent exponentiellement = **chaos**
- **λ < 0** → trajectoires convergent = **stabilité**
- **λ ≈ 0** → frontière critique (souvent optimum pour calcul, hypothèse de Langton 1990)

**Application** : si on perturbe légèrement les poids d'un NN, est-ce que sa trajectoire de training diverge ou converge ? Si λ ≈ 0 → architecture critique.

## Auto-organisation et criticalité

### Self-Organized Criticality (SOC)
Bak 1987 (papier original : "Self-organized criticality: an explanation of 1/f noise"). Idée : certains systèmes naturels s'auto-régulent à un **point critique** entre ordre et chaos, **sans qu'on doive régler de paramètre extérieur**.

**Exemples** : pile de sable, tremblements de terre, activité neuronale (Beggs & Plenz 2003 — avalanches neuronales).

**Signature** : distributions en loi de puissance (power-law), notamment **spectre de fréquences en 1/f**.

### Spectral slope 1/f^β
Forme du **spectre de puissance** d'un signal : si on prend la FFT (transformée de Fourier) et qu'on regarde la puissance par fréquence, on obtient :

```
P(f) ∝ 1 / f^β
```

- **β ≈ 0** → bruit blanc (toutes fréquences pareilles)
- **β ≈ 1** → bruit rose (1/f, courant en biologie, musique)
- **β ≈ 2** → bruit brownien (random walk integration), **signature SOC**
- **β > 2** → corrélations long-terme fortes

**En NN architecte** : si l'évolution des poids d'un NN suit une loi 1/f², ça suggère que l'architecture s'auto-organise de façon critique.

### Phase transition (transition de phase)
Changement qualitatif du comportement d'un système quand un paramètre dépasse un seuil. Exemple physique : eau → glace à 0°C.

**En NN** : transition entre régime "sur-paramétré" (loss → 0) et "sous-paramétré" (loss > 0). Frankle & Carbin 2020 — *Lottery Ticket Hypothesis*.

## Mesures internes (auto-observation)

### Effective rank
Rang effectif de la matrice des activations. Indique combien de **directions indépendantes** le réseau utilise.

```
eff_rank = exp(- Σ p_i log p_i)
où p_i = singular values normalisées
```

- eff_rank petit → représentation **collapsée** (peu de richesse)
- eff_rank ≈ dim → représentation **étalée** (utilise toute la capacité)

### Intrinsic dimension
Dimension "vraie" du manifold sur lequel les activations vivent. Souvent << dimension nominale.

**Méthodes** :
- **TwoNN** (Facco 2017) — ratio distances 2 plus proches voisins
- **MLE Levina-Bickel** (2004) — max likelihood
- **Local boundary** estimators

### φ_id (phi-ID) ou Φ d'IIT
Tononi (2008) — *Integrated Information Theory*. Quantité d'information **intégrée** dans un système : ce qu'il "sait" en plus de la somme de ses parties.

```
Φ = info_intégrée - max(info_partition)
```

Si Φ > 0 → système intégré, candidat à conscience subjective (controversée).

**En NN** : φ_id peut être estimée sur la matrice de connectivité fonctionnelle des couches.

### RQA (Recurrence Quantification Analysis)
Analyse de **patterns récurrents** dans une trajectoire. Détecte cycles, déterminisme, laminarité (états quasi-stationnaires).

Métriques principales :
- **DET** (déterminisme) — fraction de points qui forment des diagonales (récurrence corrélée)
- **LAM** (laminarité) — fraction de points qui forment des lignes verticales (quasi-immobiles)

### SINDy (Sparse Identification of Nonlinear Dynamics)
Brunton 2016. À partir de données temporelles, **découvrir l'équation différentielle** sous-jacente :

```
dx/dt = Σ ξ_i · θ_i(x)
```

où `θ_i` sont des fonctions candidates (polynomiales, trigonométriques, etc.) et `ξ_i` sont des coefficients trouvés par régression sparse (Lasso).

### DMD (Dynamic Mode Decomposition)
Schmid 2010. Décompose une trajectoire en **modes spatiotemporels** (combinaison de pattern spatial × croissance/décroissance temporelle exponentielle).

## Active inference et self-modeling

### Active inference
Friston 2010. Agent intelligent = agent qui minimise sa **free energy variationnelle** (proxy de surprise prédite). Implique :
- Predicting next state
- Predicting consequences of own actions
- Minimizing prediction error

**Hypothèse** : la conscience pourrait émerger naturellement d'un système qui modélise ses propres états.

### Strange loop
Hofstadter (1979 *Gödel Escher Bach*, 2007 *I Am a Strange Loop*). Système qui **se référence à travers les niveaux d'abstraction** :
- Niveau 1 : représente un concept X
- Niveau 2 : représente comment Niveau 1 représente X
- Niveau 3 : représente comment Niveau 2 représente le Niveau 1...
- Boucle : Niveau N revient à Niveau 1, fermant la boucle

**Conjecture Hofstadter** : conscience subjective = strange loop dans le système cognitif.

### Self-modeling
Architecture qui maintient une **représentation interne de son propre état** :
- Forme la plus simple : copie des activations d'une couche
- Forme avancée : prédiction explicite de l'état au temps t+1

**En NN** : Sutton & Barto (RL), Schmidhuber (forward models), récemment LeCun (V-JEPA, JEPA-2).

## Inductive bias

Forme implicite que prend la solution apprise du fait de l'architecture, indépendamment des données. Exemples :
- **CNN** → invariance à translation
- **Transformer** → contexte long via attention
- **ComplexMLP** → traitement phase via algèbre complexe (CLAIM RÉCEMMENT RÉFUTÉ — voir M4 findings)

## Bitter Lesson (Sutton 2019)

Observation empirique : à long terme, **calcul brute force + données** bat les méthodes spécifiques.

**Corollaire** : architectures avec inductive bias fort gagnent en regime low-data, mais sont rattrapées avec assez de data.

**Confirmé sur nos données M4** (cross-dataset chirps/ESC50/UrbanSound : Δ inverse à n_train).

## NTK (Neural Tangent Kernel)

Jacot 2018. Dans la limite infinite-width, un NN se comporte comme une **régression linéaire dans un espace de features fixe** (le NTK).

**Implication** : large NN ≈ kernel method. Permet analyse théorique de la dynamique d'apprentissage.

**Pour metamorph** : dériver NTK de VGCNN ou ComplexMLP donnerait base théorique formelle.

## Géométrie non-commutative

Connes (1985). Géométrie où xy ≠ yx (ordre des opérations compte). Outils :
- C*-algèbras
- Espaces non-séparables
- Spectral triples

**Utilité ML** : peu explorée. Potentiellement pertinent pour systèmes avec **ordre temporel intrinsèque** ou rotations non-abéliennes (quaternions, octonions).

## À compléter

Ce fichier est itératif. Chaque nouveau concept rencontré dans le travail = entrée à ajouter.
