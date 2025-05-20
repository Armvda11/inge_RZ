#!/usr/bin/env python3
# protocols/base.py
"""
Classe de base pour les protocoles DTN (Delay-Tolerant Networking).
Définit l'interface commune à tous les protocoles de routage DTN.
"""

class DTNProtocol:
    """
    Classe de base pour les protocoles DTN.
    Cette classe définit l'interface commune à tous les protocoles de routage DTN.
    """
    
    def __init__(self, num_nodes: int, source: int, dest: int):
        """
        Initialise un protocole DTN.
        
        Args:
            num_nodes (int): Nombre total de nœuds dans le réseau
            source (int): ID du nœud source du message
            dest (int): ID du nœud destinataire du message
        """
        self.source = source
        self.dest = dest
        self.delivered_at = {}  # Dictionnaire pour stocker les temps de livraison
        self.packet_logs = []   # Liste pour stocker les informations détaillées des paquets
        self.t_emit = {}        # Timestamp d'émission des paquets
        self.num_hops = {}      # Nombre de sauts pour chaque paquet
    
    def step(self, t: int, adjacency: dict[int, set[int]]):
        """
        Exécute une étape de simulation au temps t.
        
        Args:
            t (int): Temps courant de la simulation
            adjacency (dict[int, set[int]]): Dictionnaire d'adjacence représentant les liens entre les nœuds
                      Format: {id_nœud: {id_voisin1, id_voisin2, ...}, ...}
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def delivery_ratio(self) -> float:
        """
        Calcule le ratio de livraison.
        
        Returns:
            float: Ratio entre 0.0 et 1.0
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def delivery_delay(self) -> float:
        """
        Calcule le délai moyen de livraison.
        
        Returns:
            float: Délai moyen en nombre d'étapes, ou inf si aucune livraison
        """
        if not self.delivered_at:
            return float('inf')
        return sum(self.delivered_at.values()) / len(self.delivered_at)
    
    def overhead_ratio(self) -> float:
        """
        Calcule le ratio d'overhead (surcharge réseau).
        
        Returns:
            float: Nombre de transmissions par message livré
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
