# Protocole Spray-and-Wait

## Description

Le protocole Spray-and-Wait est une stratégie de routage opportuniste pour les réseaux tolérants aux délais (DTN). Il fonctionne en deux phases distinctes :

### Phase Spray
- À l'émission d'un message, le nœud source reçoit un quota initial de copies, noté L.
- Tant qu'un nœud détient plus d'une copie (> 1), dès qu'il rencontre un pair qui ne possède pas encore le message, il lui remet une partie de ses copies (généralement la moitié).
- Si un nœud A a C > 1 copies, il distribue C_to_give = ⌊C/2⌋ à son interlocuteur, et conserve C_remaining = C - C_to_give.

### Phase Wait
- Dès qu'un nœud n'a plus qu'une seule copie, il passe en phase d'attente.
- En phase "wait", il conserve cette unique copie et ne l'envoie que si le nœud rencontré est la destination finale.
- Cela évite de multiplier les copies de manière incontrôlée une fois que la diffusion initiale est faite.

## Variantes implémentées

1. **Binary Spray and Wait** (par défaut) : Chaque nœud avec plus d'une copie donne la moitié de ses copies à un nœud rencontré qui n'a pas encore le message.

2. **Source Spray and Wait** : Seul le nœud source distribue des copies (une par rencontre), les autres nœuds restent en mode "wait".

## Utilisation

### Exemple simple

```python
from protocols.spray_and_wait import SprayAndWait

# Créer une instance du protocole
num_nodes = 100  # Nombre total de nœuds dans le réseau
L = 10           # Nombre initial de copies
destination = 42 # ID du nœud destinataire
source = 0       # ID du nœud source
binary = True    # Utiliser Binary Spray and Wait

protocol = SprayAndWait(num_nodes, L, destination, source, binary)

# Simulation pas à pas
for t in range(max_time):
    adjacency = {...}  # Obtenir l'état du réseau à l'instant t
    protocol.step(t, adjacency)
    
    # Vérifier si le message est livré
    if destination in protocol.delivered_at:
        print(f"Message délivré à t={protocol.delivered_at[destination]}")
        break

# Calculer les métriques de performance
delivery_ratio = protocol.delivery_ratio()
delivery_delay = protocol.delivery_delay()
overhead = protocol.overhead_ratio()
```

## Paramètre clé : L

Le paramètre L (nombre initial de copies) doit être choisi en fonction de la densité et de la dynamique du réseau :

- **Petit L** (ex. 2-5) → overhead faible, mais risque de delivery ratio plus bas si peu de rencontres.
- **Grand L** (ex. 20-50) → beaucoup de copies disséminées, delivery ratio proche de l'épidémique, mais overhead (ressources) élevé.

## Métriques de performance

1. **Ratio de livraison** : Indique si le message a été livré à destination.
2. **Délai de livraison** : Temps nécessaire pour livrer le message.
3. **Overhead** : Nombre total de copies créées pour livrer un message.

## Avantages du protocole Spray-and-Wait

1. Limites clairement le nombre maximum de transmissions à L
2. Distribution rapide des copies dans la phase Spray
3. Garanties théoriques de performance (délai proche de l'optimal avec overhead contrôlé)
4. Meilleur compromis entre l'inondation (épidémique) et les approches single-copy

## Références

Thrasyvoulos Spyropoulos, Konstantinos Psounis, Cauligi S. Raghavendra, "Spray and Wait: An Efficient Routing Scheme for Intermittently Connected Mobile Networks", SIGCOMM 2005.
