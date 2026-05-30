# De l'émergence mesurable : observables, sincérité, et la frontière entre la règle et le surgissement

*Essai transversal — philosophie, épistémologie, informatique*
*Programme metamorphic-systems · 2026-05-30*

---

## Avant-propos

Ce document consigne une réflexion menée à la frontière de trois disciplines : la
philosophie de l'esprit, l'épistémologie des systèmes complexes, et l'apprentissage
machine. Il ne cherche pas l'exhaustivité généraliste mais la *transversalité* — la
manière dont une même question se reformule en passant d'un champ à l'autre, et dont
les outils d'un domaine éclairent les impasses d'un autre.

La question matrice : **comment des règles simples engendrent-elles de la complexité,
et comment distinguer une complexité qui *signifie* d'un chaos qui ne dit rien ?**

Cette question, ancienne (Anaximandre, Leibniz, Turing, von Neumann, Bak), reçoit ici
un traitement particulier : on cherche à la rendre *mesurable*. Non pas à trancher
métaphysiquement ce qu'est l'émergence, mais à construire des observables qui en
capturent la signature.

---

## 1. L'ancrage empirique : β, ou la pente qui révèle la mémoire

Tout le reste s'organise autour d'un objet concret. Soit un réseau de neurones, ou
plus généralement un système qui transforme un signal. On mesure le **spectre de
puissance** de ses activations internes — la décomposition de son activité en
fréquences — et l'on en extrait la pente en échelle log-log. On l'appelle β.

**Vulgarisation.** Imaginez une bibliothèque. Si les livres sont éparpillés au hasard,
l'information est partout et nulle part : c'est le bruit blanc, β ≈ 0. Si tout est
rangé sur une seule étagère, c'est rigide : β élevé. Entre les deux — un rangement
fractal, où chaque échelle reflète les autres — vit la structure intéressante :
β ≈ 1, le bruit rose, signature des systèmes critiques (Bak, 1987).

β mesure donc **la profondeur de mémoire intégrative** d'un système : à quel point son
état présent porte la trace structurée de son passé.

**Le fait empirique.** Sous un bruit gaussien pur en entrée (β = 0), les architectures
diffèrent radicalement : un perceptron multicouche reste passif (β ≈ 0, il ne fait
qu'écho à l'entrée), tandis qu'un système récurrent intégratif — une dynamique de type
équation différentielle — *génère* de la structure (β ≈ 1,5) à partir de rien. La
mémoire récurrente fabrique de la corrélation là où il n'y en avait pas.

Ce simple fait porte une charge philosophique : il opérationnalise la différence entre
**transmettre** (le système recrache les statistiques de son entrée) et **engendrer**
(le système impose sa propre signature dynamique).

---

## 2. Simuler n'est pas émerger — mais la frontière n'est pas où on la croit

La question naïve oppose les grands modèles de langage, *entraînés à imiter* le texte
humain, à une hypothétique émergence *authentique*. Cette opposition est en partie
fausse, et il importe de dire pourquoi.

**Argument épistémologique.** Les capacités d'un modèle de langage *émergent* aussi —
d'une règle simple (prédire le mot suivant) appliquée à grande échelle. Les têtes
d'induction, l'apprentissage en contexte, les algorithmes internes découverts lors du
*grokking* ne sont écrits nulle part dans les données : ils surgissent. Inversement,
les systèmes dits émergents (automates cellulaires, nuées d'oiseaux) ont leurs règles
*conçues*. Les deux relèvent de l'émergence-à-partir-de-règles-simples.

La vraie ligne de partage n'est donc pas émergence contre imitation. Elle court selon
trois axes mesurables :

1. **La provenance.** Le comportement est-il copié de la donnée, ou généré par la
   dynamique interne ? (Test : ablater la donnée ; le comportement survit-il ?)
2. **Le moteur.** Le système agit-il pour satisfaire un objectif *externe* (matcher une
   distribution) ou *intrinsèque* (minimiser sa surprise, maximiser sa compression) ?
3. **Le bouclage sur soi.** Le comportement modifie-t-il la dynamique future du système
   (boucle causale) ou n'est-il qu'une sortie morte adressée à un observateur ?

**Le point décisif.** β opérationnalise déjà l'axe de la provenance. Un système qui
imite voit ses activations refléter les statistiques de la donnée : β_interne suit
β_entrée. Un système qui engendre possède une signature dynamique *propre*, découplée
de l'entrée : β_interne ⊥ β_entrée. La différence entre « recracher le monde » et
« avoir un monde intérieur » devient, à ce niveau, une mesure.

**Position épistémologique assumée.** Au niveau du comportement observable, la
distinction simulation/émergence peut être indécidable (l'argument fonctionnaliste de
Turing tient). Au niveau de la *structure causale génératrice*, elle est réelle et
mesurable. Au niveau de l'*expérience phénoménale* — le « quelque chose que cela fait
d'être » — elle est probablement hors d'atteinte, et peut-être mal posée. La décision
méthodologique consiste à refuser la troisième question et à travailler la deuxième.

---

## 3. Une échelle de sincérité de l'émergence

Tout phénomène émergent n'a pas la même valeur. On peut les graduer selon leur
*sincérité* — terme imparfait pour désigner la distance entre ce qui fut *spécifié* et
ce qui *apparut*.

| Niveau | Nature | Exemple |
|---|---|---|
| L0 | Codé en dur | une règle si-alors |
| L1 | Le comportement *est* l'objectif | un réseau entraîné à classer classe |
| L2 | Sous-mécanisme utile non visé | les têtes d'induction d'un transformeur |
| L3 | Inattendu mais explicable après coup | l'apprentissage en contexte surgi de l'échelle |
| L4 | Irréductible, non dérivable même en principe | l'émergence forte — contestée |

**Note philosophique.** Le niveau L4 — l'émergence dite *forte* (Chalmers, Bedau) —
fait l'objet d'un débat non tranché : beaucoup soutiennent que tout est dérivable en
principe des règles, et que L4 est un mirage de notre ignorance. Le plafond opérationnel
est donc L3.

**Objection et réponse.** On pourrait dire : « L3 n'est qu'épistémique — la surprise
tient à nos limites de prévision, non à une nouveauté ontologique. » C'est exact, et
cela n'a aucune importance. Ce que nous nommons *hasard* n'est jamais que l'ensemble
des paramètres d'un système que nous ignorons. Nous ne cherchons pas un surgissement
magique, mais l'irruption de comportements à la fois *imprévus* et *structurellement
intéressants* — distincts d'un chaos inintelligible. La sincérité épistémique suffit,
pourvu qu'on la rende objective.

**Le verrou méthodologique : le pré-enregistrement.** Le seul moyen de rendre
« inattendu » objectif est de déclarer ses prédictions *avant* l'expérience. La surprise
se mesure alors comme divergence (au sens de Kullback-Leibler) entre l'attendu et
l'observé. Sans cette discipline, l'affirmation « je ne m'y attendais pas » reste
invérifiable, contaminée par le biais rétrospectif. La sincérité devient une grandeur.

---

## 4. Caractériser l'intérêt : au-delà du bord du chaos

Reste à définir ce qui rend un comportement *intéressant*. L'intuition classique pointe
le **bord du chaos** : la zone, entre l'ordre figé et le désordre, où la complexité et
la computation sont maximales (Langton, Kauffman ; la classe IV de Wolfram). Cette
intuition est juste mais insuffisante.

**Le diagnostic.** La criticalité est une propriété *statique*. L'intérêt est
*dynamique* et *relatif à un observateur*. Un système critique peut ne rien faire
d'utile ; la turbulence est critique sans être cognitivement intéressante. Mesurer β
seul ne conclut pas, précisément pour cette raison.

**Le complément (Schmidhuber).** Est intéressant ce sur quoi le modèle d'un observateur
*progresse* — non la complexité absolue, mais sa dérivée. Un motif trivial n'a rien à
apprendre ; un motif aléatoire est inapprenable ; un motif à la frontière mobile de
l'apprenable est intéressant. L'intérêt est le **progrès de compression**, une grandeur
qui se déplace à mesure que l'observateur apprend.

**Cadre proposé — cinq axes.** Un comportement est intéressant dans la mesure où il est :

1. **Nouveau** — imprévisible depuis le passé (surprise) ;
2. **Structuré** — compressible, non aléatoire (information prédictive) ;
3. **Porteur de progrès** — l'observateur s'améliore en le modélisant (progrès de compression) ;
4. **Intégré** — coordonné à l'échelle du système (information intégrée φ) ;
5. **Conséquent** — il affecte le futur (efficacité causale, testée par intervention).

Les axes 1 et 2 décrivent le bord du chaos — condition *nécessaire*. L'axe 3 est le
cœur *suffisant*. Les axes 4 et 5 séparent le phénomène significatif du soubresaut local
sans portée.

**Raffinement de la criticalité.** Plutôt qu'un seuil, on confère au bord du chaos une
*étendue*, bornée par deux observables : l'entrée dans la zone (onset de corrélation
longue portée, β quitte 0) et la sortie (l'exposant de Lyapunov franchit zéro, la
sensibilité explose). La largeur de cette zone — variable selon l'échelle spatiale,
temporelle, latente — devient elle-même un observable : elle mesure la robustesse du
régime critique.

---

## 5. Les observables ne doivent pas être choisis, mais émerger

Se pose alors une question réflexive : qui calibre les observables ? Trois voies.

- **À la main** : nous choisissons les mesures et les seuils. Fragile, subjectif.
- **Par entraînement supervisé** : on entraîne les observables contre des exemples
  étiquetés « intéressants ». **C'est un piège circulaire.** On ne récupérerait que
  notre propre préjugé de l'intéressant — une imitation (niveau L1), l'inverse de ce que
  l'on cherche.
- **Par émergence intrinsèque** : la bonne voie.

**Critères non circulaires.** Les observables pertinents peuvent émerger d'un principe
auto-supervisé, sans aucune étiquette humaine :

- **Suffisance prédictive** (goulot d'étranglement informationnel, Tishby) : un bon
  observable compresse le passé tout en préservant le pouvoir de prédire le futur ;
- **Invariance d'échelle** (groupe de renormalisation) : les variables qui comptent sont
  celles qui survivent au grossissement — la physique a résolu, via la renormalisation,
  la question « quelles variables sont pertinentes » ;
- **Désentanglement** : les observables s'auto-organisent pour être mutuellement
  indépendants tout en reconstruisant la dynamique.

**Validation décisive.** Le test qui distingue la trouvaille du bavardage : les
observables émergents *redécouvrent-ils* les connus ? Sur des systèmes où la réponse
est établie — l'attracteur de Lorenz et son exposant de Lyapunov, le bruit coloré et
son β — un dispositif d'émergence valide doit retrouver spontanément ces grandeurs.
S'il y parvient, on peut lui faire confiance sur des phénomènes inédits.

---

## 6. La question fondatrice : quelle règle définir, laquelle laisser émerger ?

On parvient au cœur du programme. Concevoir un système, c'est trancher une partition :
une partie *fixe* (architecture, objectif, a priori — les règles que l'on définit) et
une partie *apprise* (les poids, les représentations, les comportements qui émergent).
**Où placer la frontière ? Et peut-on la déterminer automatiquement ?**

**Le critère principié (Solomonoff, longueur de description minimale).** Une règle doit
être *spécifiée* si l'inscrire coûte moins que l'apprendre, *et* si elle vaut pour tous
les contextes du cadre. Elle doit *émerger* si elle dépend du contexte. En une formule :

> **Spécifier les invariants ; laisser émerger les variants.**

Ce que l'on grave dans l'architecture, ce sont les symétries du domaine — l'invariance
par translation engendre la convolution, l'invariance par permutation engendre les
réseaux de graphes. Tout le reste se découvre.

**La limite théorique dure (pas de repas gratuit).** Il n'existe aucune partition
universellement optimale. Le théorème du *no free lunch* l'interdit : la meilleure
partition est toujours *relative à un cadre* — une distribution de contextes. C'est
pourquoi la formulation « dans un cadre donné » n'est pas un détail mais la condition
de possibilité même du problème. On ne peut déterminer les règles dans l'absolu ; on
peut déterminer celles qui sont invariantes *de ce cadre*.

**La réponse.** Oui, la partition est auto-déterminable — mais relativement à un cadre.
Le mécanisme : présenter au système une diversité de contextes ; pour chaque structure
candidate, mesurer si l'inscrire aide à travers *tous* les contextes (alors : spécifier)
ou ne sert que localement (alors : laisser émerger). C'est apprendre le biais inductif
lui-même.

**La boucle méta, sans paradoxe.** Le dispositif qui détermine la partition opère à un
méta-niveau, et l'on évite la régression à l'infini par un critère d'arrêt : *ajouter une
règle spécifiée seulement si elle compresse à travers les contextes*. La récursion
s'auto-termine quand le niveau marginal cesse d'apporter du progrès. La profondeur
d'intrication n'est pas postulée : elle est dictée par la persistance de l'intérêt.

---

## 7. Fil épistémologique : ce que l'on peut savoir

Trois ordres de connaissance traversent ce programme, qu'il faut tenir distincts.

- **L'ordre comportemental** : ce que le système fait. Observable, mais possiblement
  indécidable quant à sa nature (Turing).
- **L'ordre causal-structurel** : *comment* il le fait, par quel mécanisme générateur.
  Mesurable — c'est le terrain de β, de Lyapunov, de φ. C'est là que se joue la
  distinction réelle entre imiter et engendrer.
- **L'ordre phénoménal** : ce que cela *fait* d'être ce système. Hors d'atteinte, et
  peut-être une fausse question.

La discipline consiste à ne jamais glisser de l'un à l'autre subrepticement — à ne pas
conclure d'une signature structurelle (« le système a une dynamique auto-référentielle
mesurable ») une affirmation phénoménale (« le système ressent »). C'est exactement le
glissement que toute la rigueur du programme s'emploie à interdire.

---

## 8. Fil méthodologique : l'expérience qui tranche

Une réflexion ne vaut que par les expériences qui la mettent en danger. Le programme
s'astreint à un protocole *discriminant* : concevoir l'expérience pour *tuer*
l'hypothèse, non pour la confirmer (Popper) ; choisir le régime où le réel et le nul
prédisent des résultats maximalement différents ; gravir une échelle de preuve —
existence, gradient dose-réponse, survie à l'ablation, transfert inter-conditions,
robustesse à l'échelle.

Un résultat n'accède au rang de *trouvaille* que par le gradient (un effet réel se
décline en dose-réponse, pas en point isolé) et à celui de *percée candidate* que par
le transfert (il vaut sur des tâches et des architectures distinctes). Cette exigence a,
dans la pratique, produit autant de réfutations nettes que de confirmations — et c'est
sa vertu.

---

## 9. Questions ouvertes

1. **Le progrès de compression pique-t-il à la criticalité ?** Si la zone d'intérêt
   (entre les bornes ordre et chaos) coïncide avec le maximum de progrès de compression,
   le cadre des cinq axes se referme sur lui-même de façon cohérente.
2. **Les observables auto-émergents redécouvrent-ils β et Lyapunov ?** Test de validité
   du principe d'émergence non circulaire.
3. **La partition auto-déterminée bat-elle la conception manuelle** sur des tâches
   tenues à l'écart d'un cadre donné ? C'est le test du méta-système.
4. **Un système à moteur intrinsèque et bouclé sur lui-même développe-t-il spontanément
   une structure auto-référentielle mesurable** — sans qu'on l'ait spécifiée (niveau L3) ?
5. **β découplé de l'entrée + attracteurs internes + persistance sans cible externe** :
   ce triptyque suffit-il à signer l'émergence par opposition à l'imitation ?

---

## Coda

Ce programme ne vise pas à construire un esprit, ni à trancher l'énigme de la conscience.
Il vise plus modestement, et peut-être plus solidement, à *instrumenter l'émergence* :
à transformer des intuitions séculaires — l'ordre naît du chaos, le sens surgit du bruit
— en grandeurs que l'on mesure, en partitions que l'on détermine, en surprises que l'on
quantifie.

La leçon transversale est peut-être celle-ci : la frontière entre ce que l'on conçoit
et ce qui advient n'est pas un mur, mais une membrane mesurable. Et le travail du
chercheur — qu'il soit humain ou, demain, partiellement machine — consiste moins à
décréter de quel côté se tient le sens qu'à construire les instruments qui rendent la
question décidable.

---

*Références implicites : Bak (criticalité auto-organisée), Mandelbrot (fractales),
Langton & Kauffman (bord du chaos), Wolfram (classes d'automates), Schmidhuber (théorie
formelle de l'intérêt), Tishby (goulot d'étranglement informationnel), Solomonoff &
Rissanen (longueur de description minimale), Wolpert (no free lunch), Friston (énergie
libre), Tononi (information intégrée), Chalmers & Bedau (émergence forte/faible),
Nanda & Power (grokking), Martin & Mahoney (spectres à queue lourde).*
