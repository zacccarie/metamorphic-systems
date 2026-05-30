# Protocole d'expérience discriminante

But : produire des expériences qui SÉPARENT nettement "vraie piste" de "rien".
Signal binaire, dur à truquer. Tuer vite ce qui est faux, escalader ce qui survit.

Dérivé de ce qui a marché cette session (7 falsifications nettes + β-accel grokking).

---

## Principe central : MAXIMISER LA DISCRIMINATION

Une expérience est bonne si les 2 hypothèses (réel vs null) prédisent des outcomes
MAXIMALEMENT différents. Ne JAMAIS tester là où réel et null se ressemblent.

Choisir le régime où ils DIVERGENT le plus.
- Ex : grokking = mémorise-vs-généralise = gap énorme. β shift Δ+2.0 = net.
- Mauvais : tester β-LLM difficulty (sep 0.54 = overlap = inconcluant).

Règle : si tu ne peux pas prédire un écart net AVANT, l'expérience est mal posée.

---

## L'échelle de preuve (lead → finding → breakthrough)

| Niveau | Critère | Exemple session |
|---|---|---|
| **L1 Lead** | effet existe (bat control, multi-seed) | β track généralisation r=0.6 |
| **L2 Finding** | dose-response (gradient = mécanisme, pas artefact) | β-accel : λ sweep, sweet spot 0.01 |
| **L3 Causal** | survit ablation (retire mécanisme → effet disparaît) | — |
| **L4 Général** | transfère cross-condition | β-accel : modular+parity+permutation |
| **L5 Robuste** | tient à l'échelle / vraie tâche | (pas atteint, toy) |

- **Lead** = L1. **Finding** = L2-3. **Breakthrough candidate** = L4-5.
- β-accel grokking actuel = **L4** (dose-response + cross-task). Manque L5 (scale).

---

## Template expérience discriminante (réutilisable)

```
1. HYPOTHÈSE : H + null H0, opérationnels (pas vagues)
2. PRÉ-REGISTRE : effet attendu (taille) + ce qui FALSIFIERAIT
3. MÉTRIQUE SHARP : quasi-binaire, dure à truquer (AUROC, grok_step, Δβ)
4. CONTROL matched : params/FLOPs/data identiques
5. SWEEP dose-response × seeds (PAS un point unique)
6. RÈGLE DE DÉCISION déclarée AVANT : keep / kill / nuance + seuils
7. SI keep → ablation (tue mécanisme) + cross-condition (transfère ?)
8. LOG structuré (results.tsv : commit, métrique, verdict)
```

---

## Type de claim → test sharp correspondant

| Claim | Test décisif |
|---|---|
| **Existence** ("X arrive") | bat control, multi-seed, effet > bruit |
| **Causal** ("X cause Y") | ablation + dose-response |
| **Mécanisme** ("X via Z") | intervenir sur Z directement |
| **Généralité** ("X universel") | batterie cross-condition |
| **Supériorité** ("X bat SOTA") | comparaison matched vs vrais baselines (pas petit MLP) |

Chaque claim a SON test. Ne pas confondre (ex : montrer existence ≠ prouver causalité).

---

## Discipline du kill-switch

**Chaque lead reçoit UNE expérience cheap conçue pour le TUER.**

- Survit → escalade échelle de preuve
- Meurt → enterre, log, move on

C'est ce qui a gardé la session honnête : 7 falsifications. On ne s'accroche pas.

Conçois l'expérience pour réfuter, pas confirmer (Popper). Si tu cherches à confirmer,
tu trouveras toujours un angle qui confirme (confirmation bias).

---

## Critères qu'un finding est SHARP (pas bruit)

1. **Effet >> bruit** : séparation, pas overlap (means hors std mutuels)
2. **Dose-response** : gradient monotone (ou U/inversé) = mécanisme réel
3. **Survit ablation** : causal, pas corrélation
4. **Cross-condition** : transfère (tâches/archis différentes)
5. **Pré-enregistré** : surprise objective (KL attendu vs observé)

Breakthrough candidate = grand effet + causal + général + inattendu (pré-reg).

---

## Anti-patterns (ce qui produit du faux signal)

- **Point unique** : 1 config qui marche = anecdote. Besoin sweep.
- **Baseline faible** : battre un petit MLP ≠ battre SOTA. Control fair.
- **p-hacking** : tester 20 trucs, reporter le significatif. Pré-register.
- **HARKing** : inventer l'hypothèse après avoir vu le résultat. Pré-register.
- **Métrique molle** : si "ça a l'air mieux" sans seuil → inconcluant. Métrique binaire.
- **Pas d'ablation** : corrélation prise pour causation.
- **Régime overlap** : tester où réel/null se ressemblent → toujours inconcluant.

---

## Format efficace (cheap + itérable)

- Toy testbed quand possible (grokking 75k params, ~$5, ~20min)
- Sweep paramètre clé × 3-5 seeds minimum
- Métrique unique de décision + extras loggés
- Cloud detached + volume + fetch (voir MODAL_ROBUST.md)
- Résultat en < 30min idéalement → itération rapide

---

## La question à se poser AVANT chaque expérience

1. Quel claim exact ? (existence/causal/mécanisme/généralité/supériorité)
2. Quel test sharp pour CE claim ?
3. Quel écart je prédis AVANT ? (pré-register)
4. Quelle règle de décision ? (keep/kill seuils)
5. Réel et null prédisent-ils des outcomes NETS différents ? (sinon repose l'exp)
6. C'est cheap/rapide ? (sinon réduis scope)

Si tu ne peux pas répondre aux 6 → l'expérience n'est pas prête.

---

## Application : prochaine piste

Pour toute nouvelle idée (observable self-émergent, edge-of-chaos étendue, méta-loop) :
1. Réduire à UN claim testable
2. Concevoir l'expérience qui le TUE
3. Pré-register l'écart attendu
4. Sweep + seeds, métrique sharp
5. Escalader l'échelle de preuve si survit

Le but : chaque session produit soit un finding net (L2+), soit une falsification nette.
Jamais "inconcluant" par design bâclé.
