#!/usr/bin/env python3
# protocols/spray_and_wait.py
"""
Implémentation du protocole Spray-and-Wait pour les réseaux tolérants aux délais (DTN).

Principe:
1. Phase Spray: À l'émission, la source initialise L copies.
   Lors d'une rencontre, un nœud avec >1 copies en donne la moitié à son pair.
2. Phase Wait: Dès qu'un nœud n'a plus qu'une seule copie, il attend de 
   rencontrer directement la destination pour transmettre.

Avantages:
- Plus économe en ressources que l'épidémique
- Délais de livraison contrôlables via le paramètre L
- Compromis entre overhead et fiabilité

Référence: Thrasyvoulos Spyropoulos, Konstantinos Psounis, Cauligi S. Raghavendra,
"Spray and Wait: An Efficient Routing Scheme for Intermittently Connected Mobile Networks"
"""
from protocols.base import DTNProtocol

class SprayAndWait(DTNProtocol):
    """
    Implémentation du protocole Spray-and-Wait.
    
    Ce protocole limite la diffusion des messages en distribuant un nombre limité L de copies
    dans le réseau. Chaque nœud qui possède plusieurs copies peut en transférer la moitié 
    à un nœud rencontré qui n'a pas encore reçu de copie.
    """
    
    def __init__(self, num_nodes: int, L: int, dest: int, source: int = 0, binary: bool = True, ttl: int = None, 
                 distribution_rate: float = 1.0):
        """
        Initialise le protocole Spray-and-Wait.
        
        Args:
            num_nodes (int): Nombre total de nœuds dans le réseau
            L (int): Nombre initial de copies du message
            dest (int): ID du nœud destinataire
            source (int): ID du nœud source (par défaut 0)
            binary (bool): Si True, utilise Binary Spray and Wait (division par 2).
                          Si False, utilise le Source Spray and Wait (la source garde toutes les copies)
            ttl (int): Time-To-Live pour les copies du message (en pas de temps). Si None, aucune expiration.
            distribution_rate (float): Taux de distribution des copies (1.0 = normal, < 1.0 = plus lent)
        """
        super().__init__(num_nodes, source, dest)
        self.copies = {i: 0 for i in range(num_nodes)}  # Nombre de copies par nœud
        self.copies[source] = L  # Le nœud source reçoit toutes les copies initialement
        self.t_emit[source] = 0.0  # Timestamp d'émission du message depuis la source
        self.num_hops[source] = 0  # Nombre initial de sauts (0 pour la source)
        self.binary = binary  # Mode de distribution (binary ou source)
        self.L = L  # Conserver la valeur de L pour les statistiques
        self.ttl = ttl  # Time-to-Live pour les copies du message
        self.distribution_rate = max(0.1, min(1.0, distribution_rate))  # Borné entre 0.1 et 1.0
        
        # Nouveaux attributs pour un meilleur suivi des copies
        self.copies_created_at = {source: 0}  # Timestamp de création de la copie pour chaque nœud
        self.copies_history = []  # Historique complet des copies à chaque étape
        self.delivery_history = []  # Historique des livraisons
        self.message_delivered = False  # Si le message a été livré à la destination
        
        # Nouveau compteur pour suivre toutes les copies créées/transmises
        self.total_copies_created = L  # Initialiser avec le nombre initial
        self.copy_transmissions = []  # Liste pour suivre chaque transmission
    
    def step(self, t: int, adjacency: dict[int, set[int]]):
        """
        Exécute une étape de simulation du protocole Spray-and-Wait au temps t.
        
        Args:
            t (int): Temps courant de la simulation
            adjacency (dict[int, set[int]]): Dictionnaire d'adjacence représentant les liens entre les nœuds
        """
        # Capturer l'état actuel pour l'historique
        self.copies_history.append({
            't': t,
            'copies': self.copies.copy()
        })
        
        # Vérifier les expirations des TTL et supprimer les copies expirées
        if self.ttl is not None:
            for node, creation_time in list(self.copies_created_at.items()):
                if node not in self.copies or self.copies[node] == 0:
                    continue
                
                if t - creation_time >= self.ttl:
                    # La copie a expiré
                    self.copies[node] = 0
        
        # Parcourir uniquement les nœuds présents dans la matrice d'adjacence (nœuds actifs)
        active_nodes = list(adjacency.keys())
        
        # Liste de tous les nœuds à traiter, puis mélange pour éviter les biais directionnels
        import random
        nodes_to_process = active_nodes.copy()
        random.shuffle(nodes_to_process)
        
        for i in nodes_to_process:
            # Skip si le nœud n'a pas de copies
            if self.copies.get(i, 0) == 0:
                continue
                
            # Mélanger aussi les voisins pour un traitement équitable
            neighbors = list(adjacency[i])
            random.shuffle(neighbors)
            
            for j in neighbors:
                # Vérifier que le nœud voisin est aussi actif
                if j not in active_nodes:
                    continue
                
                # Probabilité de distribution basée sur le taux défini
                # Cela ralentit la diffusion des copies lorsque distribution_rate < 1.0
                if random.random() > self.distribution_rate:
                    continue
                
                # Phase SPRAY: Si i a plus d'une copie et j n'en a pas, transférer la moitié
                if self.copies[i] > 1 and self.copies.get(j, 0) == 0:
                    # Calculer le nombre de copies à donner selon le mode (binary ou source)
                    if self.binary:
                        # Limiter le nombre de copies à donner pour ralentir la propagation
                        # Au lieu de la moitié, donner un quart des copies mais au moins 1
                        to_give = max(1, self.copies[i] // 4)  # Au moins 1 copie, mais seulement 1/4 des copies
                    else:
                        # En mode Source Spray, seule la source donne une copie à la fois
                        to_give = 1 if i == self.source else 0
                    
                    # Transférer les copies
                    self.copies[i] -= to_give
                    self.copies[j] = to_give
                    
                    # Suivre quand cette copie a été créée
                    self.copies_created_at[j] = t
                    
                    # Enregistrer la transmission pour le comptage des copies
                    self.record_copy_transmission(i, j, to_give, t)
                    
                    # Mise à jour du nombre de sauts et timestamp d'émission pour le nœud j
                    self.num_hops[j] = self.num_hops.get(i, 0) + 1
                    if j not in self.t_emit:
                        self.t_emit[j] = float(t) + 0.5  # On ajoute 0.5 pour indiquer la moitié du pas de temps
                
                # Phase WAIT: Si j est la destination et i a au moins une copie, livrer le message
                if j == self.dest and j not in self.delivered_at and self.copies.get(i, 0) >= 1:
                    self.delivered_at[j] = t
                    t_recv = float(t) + 0.5  # Timestamp précis de réception
                    
                    # Marquer comme livré pour le suivi
                    self.message_delivered = True
                    
                    # Enregistrer explicitement le nombre de sauts pour la destination
                    hops_to_dest = self.num_hops.get(i, 0) + 1
                    self.num_hops[j] = hops_to_dest
                    
                    # Enregistrer la livraison dans l'historique
                    self.delivery_history.append({
                        't': t,
                        'delivering_node': i,
                        'destination': j,
                        'hops': hops_to_dest
                    })
                    
                    # Ajouter au log de paquets
                    packet_id = f"spray_{i}_{j}_{t}"
                    self.packet_logs.append({
                        'protocol': 'Spray-and-Wait',
                        'packet_id': packet_id,
                        'src': self.source,
                        'dst': self.dest,
                        't_emit': self.t_emit.get(self.source, 0.0),
                        't_recv': t_recv,
                        'num_hops': hops_to_dest
                    })
    
    def record_copy_transmission(self, from_node: int, to_node: int, num_copies: int, t: int):
        """
        Enregistre une transmission de copies entre deux nœuds.
        
        Args:
            from_node (int): Nœud qui envoie les copies
            to_node (int): Nœud qui reçoit les copies
            num_copies (int): Nombre de copies transmises
            t (int): Temps de la transmission
        """
        self.copy_transmissions.append({
            'time': t,
            'from': from_node,
            'to': to_node,
            'copies': num_copies
        })
        
        # Incrémenter le compteur total de copies créées
        self.total_copies_created += num_copies
    
    def delivery_ratio(self) -> float:
        """
        Calcule le ratio de livraison.
        
        Returns:
            float: Ratio entre 0.0 et 1.0
        """
        # La destination est-elle dans les nœuds livrés ?
        return 1.0 if self.dest in self.delivered_at else 0.0
    
    def delivery_delay(self) -> float:
        """
        Calcule le délai de livraison.
        
        Returns:
            float: Délai en nombre d'étapes, ou inf si aucune livraison
        """
        if self.dest in self.delivered_at:
            return self.delivered_at[self.dest]
        else:
            return float('inf')
    
    def overhead_ratio(self) -> float:
        """
        Calcule le ratio d'overhead (surcharge réseau).
        
        Returns:
            float: Nombre de transmissions par message livré
        """
        # Utiliser le compteur total de copies créées au lieu des copies actuelles
        # Cette valeur sera incrémentée à chaque transmission
        total_copies = self.total_copies_created
        
        # Livré ou pas
        delivered = 1 if self.dest in self.delivered_at else 0
        
        # Overhead = (copies totales - 1) / messages livrés, ou inf si non livré
        return (total_copies - 1) / delivered if delivered > 0 else float('inf')
    
    def get_hop_stats(self) -> dict:
        """
        Récupère des statistiques détaillées sur les sauts du message.
        
        Returns:
            dict: Statistiques sur les sauts (min, max, moyenne, médiane)
        """
        import numpy as np
        hops = list(self.num_hops.values())
        if not hops:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
        
        return {
            'min': min(hops),
            'max': max(hops),
            'mean': np.mean(hops),
            'median': np.median(hops),
            'destination': self.num_hops.get(self.dest, float('inf')) if self.dest in self.num_hops else None
        }
    
    def __str__(self) -> str:
        """
        Retourne une représentation textuelle du protocole.
        
        Returns:
            str: Chaîne décrivant le protocole et ses paramètres
        """
        mode = "Binary" if self.binary else "Source"
        ttl_info = f", TTL={self.ttl}" if self.ttl is not None else ""
        rate_info = f", Rate={self.distribution_rate:.2f}" if self.distribution_rate != 1.0 else ""
        return f"{mode} Spray and Wait (L={self.L}{ttl_info}{rate_info})"
