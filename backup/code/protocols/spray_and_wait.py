# protocols/spray_and_wait.py
from protocols.base import DTNProtocol

class SprayAndWait(DTNProtocol):
    """
    Implémentation du protocole Spray-and-Wait.
    
    Ce protocole limite la diffusion des messages en distribuant un nombre limité L de copies
    dans le réseau. Chaque nœud qui possède plusieurs copies peut en transférer la moitié 
    à un nœud rencontré qui n'a pas encore reçu de copie.
    """
    
    def __init__(self, num_nodes: int, L: int, dest: int, source: int = 0):
        """
        Initialise le protocole Spray-and-Wait.
        
        Args:
            num_nodes: Nombre total de nœuds dans le réseau
            L: Nombre initial de copies du message
            dest: ID du nœud destinataire
            source: ID du nœud source (par défaut 0)
        """
        super().__init__(num_nodes, source, dest)
        self.copies = {i: 0 for i in range(num_nodes)}  # Nombre de copies par nœud
        self.copies[source] = L  # Le nœud source reçoit toutes les copies initialement
        self.t_emit[source] = 0.0  # Timestamp d'émission du message depuis la source
        self.num_hops[source] = 0  # Nombre initial de sauts (0 pour la source)
    
    def step(self, t: int, adjacency: dict[int, set[int]]):
        """
        Exécute une étape de simulation au temps t.
        
        Pour chaque nœud actif qui possède des copies, transfère la moitié de ses copies
        aux voisins qui n'ont pas encore reçu de copie.
        
        Args:
            t: Temps courant de la simulation
            adjacency: Dictionnaire d'adjacence représentant les liens entre les nœuds
        """
        # Parcourir uniquement les nœuds présents dans la matrice d'adjacence (nœuds actifs)
        active_nodes = list(adjacency.keys())
        
        for i in active_nodes:
            for j in adjacency[i]:
                # Vérifier que les nœuds sont actifs avant d'échanger des messages
                if j in active_nodes:
                    # Transmettre des copies si i en a plus d'une et j n'en a pas
                    if self.copies.get(i, 0) > 1 and self.copies.get(j, 0) == 0:
                        send = self.copies[i] // 2
                        self.copies[i] -= send
                        self.copies[j] = send
                        
                        # Mise à jour du nombre de sauts et timestamp d'émission pour le nœud j
                        self.num_hops[j] = self.num_hops.get(i, 0) + 1
                        if j not in self.t_emit:
                            self.t_emit[j] = float(t) + 0.5  # On ajoute 0.5 pour indiquer la moitié du pas de temps
                    
                    # Enregistrer la livraison si le nœud courant est la destination 
                    # et qu'il possède au moins une copie du message
                    if self.copies.get(i, 0) >= 1 and j == self.dest and j not in self.delivered_at:
                        self.delivered_at[j] = t
                        t_recv = float(t) + 0.5  # Timestamp précis de réception
                        
                        # Ajouter au log de paquets
                        packet_id = f"spray_{i}_{j}_{t}"
                        self.packet_logs.append({
                            'protocol': 'Spray-and-Wait',
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
        # La destination est-elle dans les nœuds livrés ?
        return len(self.delivered_at) / (len(self.copies) - 1)