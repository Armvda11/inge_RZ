# protocols/epidemic.py
from protocols.base import DTNProtocol

class Epidemic(DTNProtocol):
    """
    Implémentation du protocole Epidemic (épidémique).
    
    Ce protocole diffuse les messages comme une épidémie : chaque nœud qui possède 
    le message le transmet à tous les nœuds qu'il rencontre et qui n'ont pas 
    encore reçu le message.
    """
    
    def __init__(self, num_nodes: int, source: int, dest: int):
        """
        Initialise le protocole Epidemic.
        
        Args:
            num_nodes: Nombre total de nœuds dans le réseau
            source: ID du nœud source du message
            dest: ID du nœud destinataire du message
        """
        super().__init__(num_nodes, source, dest)
        self.has_msg = {i: False for i in range(num_nodes)}  # Indique si le nœud a reçu le message
        self.has_msg[source] = True  # Le nœud source a initialement le message
        self.t_emit[source] = 0.0    # Timestamp d'émission du message depuis la source
        self.num_hops[source] = 0    # Nombre initial de sauts (0 pour la source)
    
    def step(self, t: int, adjacency: dict[int, set[int]]):
        """
        Exécute une étape de simulation au temps t.
        
        Pour chaque nœud actif qui possède le message, le transmet à tous ses voisins
        qui ne l'ont pas encore reçu.
        
        Args:
            t: Temps courant de la simulation
            adjacency: Dictionnaire d'adjacence représentant les liens entre les nœuds
        """
        # Ne considérer que les nœuds actifs (présents dans adjacency)
        active_nodes = list(adjacency.keys())
        to_receive = set()
        
        # Identifier les nœuds qui vont recevoir le message pendant cette étape
        for i in active_nodes:
            if self.has_msg.get(i, False):  # Si le nœud i a le message
                for j in adjacency[i]:
                    if j in active_nodes and not self.has_msg.get(j, False):
                        to_receive.add((i, j))  # (émetteur, récepteur) va transmettre le message
        
        # Mettre à jour l'état des nœuds qui reçoivent le message
        for i, j in to_receive:
            self.has_msg[j] = True
            # Mise à jour du nombre de sauts
            self.num_hops[j] = self.num_hops.get(i, 0) + 1
            # Timestamp de première émission
            if j not in self.t_emit:
                self.t_emit[j] = float(t) + 0.5  # On ajoute 0.5 pour indiquer la moitié du pas de temps
            
            # Si le destinataire reçoit le message pour la première fois
            if j == self.dest and j not in self.delivered_at:
                self.delivered_at[j] = t
                t_recv = float(t) + 0.5  # Timestamp précis de réception
                
                # Ajouter au log de paquets
                packet_id = f"epidemic_{i}_{j}_{t}"
                self.packet_logs.append({
                    'protocol': 'Epidemic',
                    'packet_id': packet_id,
                    'src': self.source,
                    'dst': self.dest,
                    't_emit': self.t_emit[self.source],
                    't_recv': t_recv,
                    'num_hops': self.num_hops.get(j, 1)  # Valeur par défaut de 1 si manquante
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