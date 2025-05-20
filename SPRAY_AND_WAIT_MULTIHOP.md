# Simulation Multi-sauts du Protocole Spray and Wait (Améliorée)

## Description du scénario

Le protocole Spray-and-Wait est particulièrement intéressant dans des environnements où les messages doivent transiter par plusieurs nœuds intermédiaires avant d'atteindre leur destination. Ce scénario multi-sauts permet d'observer comment les copies se propagent à travers le réseau et comment le paramètre L influence la diffusion, le délai et le nombre de sauts.

## Améliorations apportées

La simulation a été significativement améliorée pour résoudre plusieurs problèmes identifiés dans la version précédente:

1. **Distribution des copies ralentie**: La distribution des copies se fait maintenant progressivement (sur plusieurs pas de temps) grâce à:
   - Un taux de distribution contrôlé (`distribution_rate = 0.4`)
   - Une division réduite des copies (1/4 au lieu de 1/2) 
   - Une topologie réseau plus restrictive avec moins de connexions

2. **Simulation complète**: La simulation ne s'arrête plus dès la première livraison mais continue jusqu'à:
   - Atteindre le nombre maximum de pas de temps
   - Ou jusqu'à expiration/consommation de toutes les copies

3. **Time-To-Live (TTL)**: Les copies ont désormais une durée de vie limitée:
   - Expire après un nombre défini de pas de temps (TTL = 20)
   - Contrôle l'overhead réseau après la livraison

4. **Comptage des sauts amélioré**:
   - Suivi explicite du nombre de sauts pour chaque nœud
   - Statistiques détaillées (min, max, moyenne, médiane)

## Architecture du réseau multi-sauts

Le réseau multi-sauts est conçu pour simuler une topologie dynamique avec les caractéristiques suivantes:

1. **Structure en chaîne de clusters** : Les nœuds sont organisés en groupes disposés en chaîne linéaire
2. **Mobilité contrôlée** : La composition des clusters et les connexions changent au fil du temps
3. **Passerelles inter-clusters limitées** : Un nombre réduit de nœuds servent de passerelles entre clusters adjacents
4. **Source et destination éloignées** : La source et la destination sont placées dans des clusters aux extrémités et ne sont jamais directement connectées
5. **Connectivité variable** : La probabilité de connexion varie dans le temps pour simuler la mobilité

## Paramètres et métriques analysés

### Paramètres
- **L** : Nombre initial de copies (valeurs testées : 4, 8, 16)
- **TTL** : Durée de vie des copies (20 pas de temps)
- **Distribution Rate** : Taux de distribution des copies (0.4)
- **Nombre de nœuds** : 20 nœuds au total
- **Durée de simulation** : Maximum 50 étapes temporelles

### Métriques évaluées
1. **Délai de livraison** : Temps nécessaire pour livrer le message à destination
2. **Nombre de sauts** : Nombre d'intermédiaires traversés par le message
3. **Distribution des copies** : Comment les copies se propagent à travers le réseau
4. **Overhead** : Surcharge réseau (nombre total de copies créées)

## Visualisations produites

Le test génère plusieurs types de visualisations pour analyser le comportement du protocole :

1. **État du réseau** : Représentation graphique du réseau à certains moments-clés, montrant :
   - Les nœuds et leurs connexions
   - La distribution des copies (taille des nœuds proportionnelle au nombre de copies)
   - Les états des nœuds (source, destination, relais avec copies, nœuds sans copie)

2. **Carte de chaleur** : Visualisation de la propagation des copies dans le temps
   - L'axe X représente le temps
   - L'axe Y représente les différents nœuds
   - L'intensité des couleurs indique le nombre de copies

3. **Évolution des sauts** : Graphique montrant comment le nombre de sauts évolue pour chaque nœud
   - Permet de visualiser la progression du message à travers le réseau
   - Met en évidence le chemin vers la destination

4. **Graphiques comparatifs** : Analyse de l'impact du paramètre L sur :
   - Le délai de livraison
   - Le nombre total de copies créées
   - Le nombre maximum de sauts
   - Le nombre de sauts pour atteindre la destination

## Analyse des résultats

### Impact du paramètre L

Le choix de L influence directement :

1. **Vitesse de diffusion** : Un L plus grand permet une propagation plus rapide dans la phase Spray
2. **Probabilité de livraison** : Un L plus grand augmente les chances de livraison, particulièrement utile dans les réseaux peu denses
3. **Overhead** : Un L plus grand génère plus de copies et donc plus de trafic réseau

### Efficacité en multi-sauts

Le protocole Spray-and-Wait présente plusieurs avantages en scénario multi-sauts :

1. **Distribution rapide et contrôlée** : Les copies se répandent rapidement dans les premiers sauts, mais le nombre total reste limité à L
2. **Compromis délai/overhead** : Ajustable selon les besoins via le paramètre L
3. **Robustesse** : Le message peut emprunter plusieurs chemins parallèles, ce qui améliore la résilience aux pannes de nœuds

### Comparaison avec d'autres approches

Dans un scénario multi-sauts, Spray-and-Wait se positionne entre deux extrêmes :

- **Plus efficace que les protocoles single-copy** : Offre une meilleure probabilité de livraison et un délai réduit par rapport aux approches qui ne créent qu'une seule copie
- **Plus économe que les protocoles épidémiques** : Limite le nombre de copies à L, contrairement aux approches épidémiques qui peuvent créer un nombre exponentiel de copies

## Conclusion

Le protocole Spray-and-Wait est particulièrement adapté aux scénarios multi-sauts car il offre un bon compromis entre la rapidité de livraison et l'utilisation des ressources. Le paramètre L permet d'ajuster ce compromis en fonction des caractéristiques du réseau et des exigences de l'application.
