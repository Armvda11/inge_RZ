# Résultats observés de la simulation multi-sauts améliorée

## Améliorations constatées

La version améliorée de la simulation du protocole Spray and Wait en scénario multi-sauts a permis d'observer les résultats suivants:

1. **Délai de livraison plus réaliste**:
   - L=4: Livraison à t=4
   - L=8: Livraison à t=4
   - L=16: Livraison à t=2
   - (Contre t=1 systématiquement dans la version précédente)

2. **Distribution progressive des copies**:
   - Les copies se propagent sur plusieurs pas de temps (5-10)
   - Le taux de distribution de 0.4 ralentit efficacement la diffusion
   - La distribution par 1/4 (au lieu de 1/2) limite davantage la vitesse de propagation

3. **TTL efficace**:
   - Les copies expirent après 20 pas de temps
   - On observe une réduction graduelle du nombre de copies actives
   - L'overhead est mieux contrôlé après la livraison du message

4. **Suivi des sauts précis**:
   - Pour L=4 et L=8: 2 sauts pour atteindre la destination
   - Pour L=16: 3 sauts pour atteindre la destination
   - Les sauts sont correctement comptabilisés pour tous les nœuds

5. **Visualisation améliorée**:
   - Les clusters sont clairement visibles dans les graphiques réseau
   - Les liens entre nœuds sont colorés selon leur statut
   - Le moment de livraison est marqué sur tous les graphiques temporels

## Analyse comparative

Les résultats montrent un compromis clair entre le nombre initial de copies (L) et les performances:

| Paramètre L | Délai de livraison | Sauts vers destination | Nœuds avec copies |
|-------------|-------------------|------------------------|-------------------|
| L=4         | 4                 | 2                      | 4                 |
| L=8         | 4                 | 2                      | 8                 |
| L=16        | 2                 | 3                      | 16                |

## Points forts du protocole

1. **Adaptabilité**: Le paramètre L permet d'ajuster le compromis entre délai et overhead
2. **Robustesse**: Même avec un faible nombre de copies (L=4), le message est livré
3. **Scalabilité**: L'overhead reste contrôlé grâce au TTL et au nombre limité de copies

## Perspectives

Les résultats suggèrent plusieurs pistes d'amélioration:
- Tester avec des valeurs de TTL variables pour analyser son impact
- Introduire des injections périodiques pour des statistiques plus robustes
- Analyser l'impact de différents modèles de mobilité
- Comparer avec d'autres variantes du protocole Spray and Wait
