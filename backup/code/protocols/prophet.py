# protocols/prophet.py
from protocols.base import DTNProtocol

class Prophet(DTNProtocol):
    """
    Implémentation du protocole PRoPHET (Probabilistic Routing Protocol using History of Encounters and Transitivity).
    
    Ce protocole utilise l'historique des rencontres pour estimer la probabilité 
    de livraison d'un message. Chaque nœud maintient une table de probabilité
    de rencontre avec les autres nœuds, et les messages sont transmis aux nœuds
    ayant une probabilité plus élevée d'atteindre la destination.
    """
    
    def __init__(self, num_nodes: int, p_init: float, source: int, dest: int):
        """
        Initialise le protocole PRoPHET.
        
        Args:
            num_nodes: Nombre total de nœuds dans le réseau
            p_init: Probabilité initiale de rencontre
            source: ID du nœud source du message
            dest: ID du nœud destinataire du message
        """
        super().__init__(num_nodes, source, dest)
        # Initialiser la matrice de probabilités de rencontre
        self.P = {i: {j: (1.0 if i == j else p_init) for j in range(num_nodes)}
                  for i in range(num_nodes)}
        self.has_msg = {i: False for i in range(num_nodes)}  # Indique si le nœud a reçu le message
        self.has_msg[source] = True  # Le nœud source a initialement le message
        self.p_init = p_init  # Stockage de la probabilité initiale pour les mises à jour
        self.gamma = 0.98  # Facteur de vieillissement pour la décroissance temporelle des probabilités
        self.beta = 0.25  # Facteur de mise à l'échelle pour la propriété de transitivité
        self.last_update = {i: {j: 0 for j in range(num_nodes)} for i in range(num_nodes)}
        self.t_emit[source] = 0.0  # Timestamp d'émission du message depuis la source
        self.num_hops[source] = 0  # Nombre initial de sauts (0 pour la source)
    
    def step(self, t: int, adjacency: dict[int, set[int]]):
        """
        Exécute une étape de simulation au temps t.
        
        Pour chaque paire de nœuds actifs qui se rencontrent:
        1. Met à jour les probabilités de rencontre
        2. Transfère le message si la probabilité de rencontre avec la destination est plus élevée
        
        Args:
            t: Temps courant de la simulation
            adjacency: Dictionnaire d'adjacence représentant les liens entre les nœuds
        """
        # Ne considérer que les nœuds actifs (présents dans adjacency)
        active_nodes = list(adjacency.keys())
        
        # Mise à jour des probabilités et des transmissions
        for i in active_nodes:
            for j in adjacency[i]:
                if j in active_nodes:  # Vérifier que j est aussi actif
                    # Mise à jour des probabilités directes (lors de la rencontre)
                    if i in self.P and j in self.P[i]:
                        # Appliquer le vieillissement en fonction du temps écoulé depuis la dernière mise à jour
                        time_since_last_update = t - self.last_update[i][j]
                        if time_since_last_update > 0:
                            self.P[i][j] *= self.gamma ** time_since_last_update
                        
                        # Mise à jour directe lors d'une rencontre
                        self.P[i][j] = self.P[i][j] + (1 - self.P[i][j]) * self.p_init
                        self.last_update[i][j] = t
                        
                        # Mise à jour pour j->i également
                        time_since_last_update = t - self.last_update[j][i]
                        if time_since_last_update > 0:
                            self.P[j][i] *= self.gamma ** time_since_last_update
                        self.P[j][i] = self.P[j][i] + (1 - self.P[j][i]) * self.p_init
                        self.last_update[j][i] = t
                        
                        # Propriété de transitivité
                        for k in range(len(self.P)):
                            if k != i and k != j:
                                self.P[i][k] = self.P[i][k] + (1 - self.P[i][k]) * self.P[i][j] * self.P[j][k] * self.beta
                                self.P[j][k] = self.P[j][k] + (1 - self.P[j][k]) * self.P[j][i] * self.P[i][k] * self.beta
                    
                    # Échange message si la probabilité de rencontrer la destination est plus élevée
                    if self.has_msg.get(i, False) and not self.has_msg.get(j, False) and self.P.get(j, {}).get(self.dest, 0) > self.P.get(i, {}).get(self.dest, 0):
                        self.has_msg[j] = True
                        # Mise à jour du nombre de sauts
                        self.num_hops[j] = self.num_hops.get(i, 0) + 1
                        # Timestamp de première émission
                        if j not in self.t_emit:
                            self.t_emit[j] = float(t) + 0.5
                            
                    if self.has_msg.get(j, False) and not self.has_msg.get(i, False) and self.P.get(i, {}).get(self.dest, 0) > self.P.get(j, {}).get(self.dest, 0):
                        self.has_msg[i] = True
                        # Mise à jour du nombre de sauts
                        self.num_hops[i] = self.num_hops.get(j, 0) + 1
                        # Timestamp de première émission
                        if i not in self.t_emit:
                            self.t_emit[i] = float(t) + 0.5
                    
                    # Enregistrement livraison
                    if j == self.dest and self.has_msg.get(j, False) and j not in self.delivered_at:
                        self.delivered_at[j] = t
                        t_recv = float(t) + 0.5  # Timestamp précis de réception
                        
                        # Ajouter au log de paquets
                        packet_id = f"prophet_{i}_{j}_{t}"
                        self.packet_logs.append({
                            'protocol': 'Prophet',
                            'packet_id': packet_id,
                            'src': self.source,
                            'dst': self.dest,
                            't_emit': self.t_emit[self.source],
                            't_recv': t_recv,
                            'num_hops': self.num_hops.get(j, 1)  # Valeur par défaut de 1 si manquante
                        })
                        
                    if i == self.dest and self.has_msg.get(i, False) and i not in self.delivered_at:
                        self.delivered_at[i] = t
                        t_recv = float(t) + 0.5  # Timestamp précis de réception
                        
                        # Ajouter au log de paquets
                        packet_id = f"prophet_{j}_{i}_{t}"
                        self.packet_logs.append({
                            'protocol': 'Prophet',
                            'packet_id': packet_id,
                            'src': self.source,
                            'dst': self.dest,
                            't_emit': self.t_emit[self.source],
                            't_recv': t_recv,
                            'num_hops': self.num_hops.get(i, 1)  # Valeur par défaut de 1 si manquante
                        })
    
    def delivery_ratio(self) -> float:
        """
        Calcule le ratio de livraison.
        
        Returns:
            float: Ratio entre 0.0 et 1.0
        """
        # Nombre de nœuds qui ont reçu le message, moins le nœud source
        received = sum(1 for node_id, has_msg in self.has_msg.items() if has_msg) - 1
        return received / (len(self.has_msg) - 1) if len(self.has_msg) > 1 else 0.0
    
    def delivery_delay(self) -> float:
        """
        Calcule le délai moyen de livraison.
        
        Returns:
            float: Délai moyen ou inf si aucune livraison
        """
        return (sum(self.delivered_at.values()) / len(self.delivered_at)
                if self.delivered_at else float('inf'))