#!/usr/bin/env python3
# protocols/prophet.py
"""
Implémentation du protocole PRoPHET (Probabilistic Routing Protocol using History of Encounters and Transitivity)
pour les réseaux tolérants aux délais (DTN).

Principe:
1. Chaque nœud maintient une "matrice de confiance" P(A,B) qui estime la probabilité que A puisse
   livrer un message à B.
2. Cette métrique est mise à jour selon trois règles:
   - Rencontre directe: P(A,B) = P(A,B)_ancien + (1 - P(A,B)_ancien) * P_init
   - Vieillissement: P(A,B) = P(A,B)_ancien * γ^Δt où Δt est le temps écoulé depuis la dernière mise à jour
   - Transitivité: P(A,C) = P(A,C)_ancien + (1 - P(A,C)_ancien) * P(A,B) * P(B,C) * β
3. Politique de transfert: Un nœud A transfère une copie à B si et seulement si P(B,D) > P(A,D)
   où D est la destination finale.

Avantages:
- Adaptatif aux motifs de mobilité récurrents
- Meilleure distribution des copies que l'épidémique
- Exploite les rencontres passées pour de meilleures prédictions

Référence: Lindgren, A., Doria, A., & Schelén, O. (2003).
"Probabilistic routing in intermittently connected networks"
ACM SIGMOBILE mobile computing and communications review, 7(3), 19-20.
"""
from protocols.base import DTNProtocol
import numpy as np
import random
import time

class Prophet(DTNProtocol):
    """
    Implémentation du protocole PRoPHET (Probabilistic Routing Protocol using History of Encounters and Transitivity).
    
    Ce protocole maintient une métrique de probabilité pour chaque destination et utilise cette
    information pour déterminer si un message doit être transmis lors d'une rencontre.
    """
    
    def __init__(self, num_nodes: int, p_init: float, dest: int, source: int = 0, 
                 gamma: float = 0.98, beta: float = 0.25, ttl: int = None,
                 distribution_rate: float = 1.0):
        """
        Initialise le protocole PRoPHET.
        
        Args:
            num_nodes (int): Nombre total de nœuds dans le réseau
            p_init (float): Probabilité initiale lors d'une rencontre (entre 0 et 1)
            dest (int): ID du nœud destinataire
            source (int): ID du nœud source (par défaut 0)
            gamma (float): Facteur de vieillissement (entre 0 et 1)
            beta (float): Facteur de transitivité (entre 0 et 1)
            ttl (int): Time-To-Live pour les copies du message (en pas de temps). Si None, aucune expiration.
            distribution_rate (float): Taux de distribution des copies (1.0 = normal, < 1.0 = plus lent)
        """
        super().__init__(num_nodes, source, dest)
        self.p_init = max(0.01, min(p_init, 0.95))  # Borné entre 0.01 et 0.95 (augmenté)
        self.gamma = max(0.7, min(gamma, 0.995))    # Facteur de vieillissement borné (augmenté pour ralentir l'érosion)
        self.beta = max(0.1, min(beta, 0.9))       # Facteur de transitivité borné
        self.ttl = ttl                            # Time-to-Live pour les messages
        self.distribution_rate = max(0.1, min(1.0, distribution_rate))  # Borné entre 0.1 et 1.0
        
        # Initialisation de la matrice de probabilité P(i,j) : probabilité que i rencontre j
        # Chaque nœud a sa propre ligne dans la matrice
        # Initialiser avec une petite probabilité de base (0.01) plutôt que zéro
        self.probability_matrix = np.ones((num_nodes, num_nodes)) * 0.01
        
        # Pour chaque nœud, la probabilité avec lui-même est 1.0
        for i in range(num_nodes):
            self.probability_matrix[i, i] = 1.0
        
        # Dernière fois que les nœuds se sont rencontrés (pour le calcul du vieillissement)
        self.last_encounter = np.zeros((num_nodes, num_nodes))
        
        # Activer le mode debug pour tracer les valeurs des probabilités
        self.debug_mode = False
        
        # Copies du message par nœud (1 = a une copie, 0 = n'a pas de copie)
        self.copies = {i: 0 for i in range(num_nodes)}
        self.copies[source] = 1  # La source a une copie initialement
        self.t_emit[source] = 0.0  # Timestamp d'émission du message depuis la source
        self.num_hops[source] = 0  # Nombre initial de sauts (0 pour la source)
        self.copies_created_at = {source: 0}  # Timestamp de création de la copie pour chaque nœud
        
        # Historique et statistiques
        self.copies_history = []  # Historique complet des copies à chaque étape
        self.delivery_history = []  # Historique des livraisons
        self.message_delivered = False  # Si le message a été livré à la destination
        self.total_copies_created = 1  # Initialiser avec le nombre initial (1 pour la source)
        self.copy_transmissions = []  # Liste pour suivre chaque transmission
    
    def age_probability(self, i: int, j: int, current_time: int):
        """
        Applique le vieillissement de la probabilité entre les nœuds i et j.
        
        Args:
            i (int): ID du premier nœud
            j (int): ID du second nœud
            current_time (int): Temps actuel
            
        Returns:
            float: La probabilité mise à jour après vieillissement
        """
        # Temps écoulé depuis la dernière rencontre
        time_elapsed = current_time - self.last_encounter[i, j]
        
        # Facteur de vieillissement: gamma^(time_elapsed)
        aging_factor = self.gamma ** time_elapsed
        
        # Application du vieillissement
        self.probability_matrix[i, j] *= aging_factor
        
        return self.probability_matrix[i, j]
    
    def update_direct_probability(self, i: int, j: int, current_time: int):
        """
        Met à jour la probabilité lors d'une rencontre directe entre i et j.
        Version améliorée pour accélérer la convergence des probabilités.
        
        Args:
            i (int): ID du premier nœud
            j (int): ID du second nœud
            current_time (int): Temps actuel
        """
        # D'abord appliquer le vieillissement
        old_prob = self.age_probability(i, j, current_time)
        
        # Facteur d'accélération pour la destination
        boost_factor = 1.0
        if j == self.dest or i == self.dest:
            # Accélérer les mises à jour impliquant la destination
            boost_factor = 2.0
        
        # Mise à jour de la probabilité directe avec renforcement:
        # P(i,j) = P(i,j)_old + (1-P(i,j)_old) * P_init * boost_factor
        self.probability_matrix[i, j] = old_prob + (1 - old_prob) * self.p_init * boost_factor
        
        # Garantir que la valeur reste dans l'intervalle [0, 1]
        self.probability_matrix[i, j] = min(1.0, max(0.0, self.probability_matrix[i, j]))
        
        # Mettre à jour le temps de dernière rencontre
        self.last_encounter[i, j] = current_time
    
    def update_transitive_probability(self, i: int, j: int, k: int):
        """
        Met à jour la probabilité par transitivité: i rencontre j qui a une bonne
        probabilité de rencontrer k. Version améliorée pour accélérer la propagation
        de l'information, particulièrement vers la destination.
        
        Args:
            i (int): ID du nœud qui rencontre j
            j (int): ID du nœud rencontré
            k (int): ID d'un nœud tiers que j a une bonne probabilité de rencontrer
        """
        # Facteur d'amplification pour la destination
        destination_boost = 1.0
        if k == self.dest:
            destination_boost = 1.5  # Amplifier les probabilités impliquant la destination
        
        # Si j a une forte probabilité de rencontrer k, amplifions davantage
        confidence_boost = 1.0
        if self.probability_matrix[j, k] > 0.6:
            confidence_boost = 1.3
        
        # P(i,k) = P(i,k)_old + (1-P(i,k)_old) * P(i,j) * P(j,k) * beta * boosts
        old_prob = self.probability_matrix[i, k]
        transitivity = (self.probability_matrix[i, j] * 
                        self.probability_matrix[j, k] * 
                        self.beta * 
                        destination_boost * 
                        confidence_boost)
        
        self.probability_matrix[i, k] = old_prob + (1 - old_prob) * transitivity
        
        # Garantir que la valeur reste dans l'intervalle [0, 1]
        self.probability_matrix[i, k] = min(1.0, max(0.0, self.probability_matrix[i, k]))
    
    def step(self, t: int, adjacency: dict[int, set[int]]):
        """
        Exécute une étape de simulation du protocole PRoPHET au temps t.
        
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
        nodes_to_process = active_nodes.copy()
        random.shuffle(nodes_to_process)
        
        for i in nodes_to_process:
            # Skip si le nœud n'a pas de copie du message
            if self.copies.get(i, 0) == 0:
                continue
                
            # Mélanger aussi les voisins pour un traitement équitable
            neighbors = list(adjacency[i])
            random.shuffle(neighbors)
            
            for j in neighbors:
                # Vérifier que le nœud voisin est aussi actif
                if j not in active_nodes:
                    continue
                
                # Mise à jour des probabilités pour tous les nœuds actifs:
                
                # 1. Mise à jour directe pour i et j
                self.update_direct_probability(i, j, t)
                self.update_direct_probability(j, i, t)  # Symétrique
                
                # 2. Mise à jour transitive pour tous les nœuds
                for k in range(len(self.probability_matrix)):
                    if k != i and k != j:
                        self.update_transitive_probability(i, j, k)
                        self.update_transitive_probability(j, i, k)
                
                # Probabilité de distribution basée sur le taux défini
                if random.random() > self.distribution_rate:
                    continue
                
                # Forwarding du message si:
                # - i a le message
                # - j n'a pas déjà le message
                # - (Condition assouplie) La probabilité de j pour rencontrer la destination est >= celle de i
                #   ou si la probabilité de j est significative (> 0.1)
                
                # Debug des probabilités plus détaillé
                if self.debug_mode:
                    if random.random() < 0.2:  # Réduire la verbosité en n'affichant que 20% des rencontres
                        print(f"T={t}, Rencontre: {i}-{j}, "
                              f"P({i},{self.dest})={self.probability_matrix[i, self.dest]:.4f}, "
                              f"P({j},{self.dest})={self.probability_matrix[j, self.dest]:.4f}, "
                              f"P({i},{j})={self.probability_matrix[i, j]:.4f}")
                
                # Condition de transfert optimisée:
                # 1. Critère direct: P(j,dest) > P(i,dest) × theta, où theta < 1 permet de favoriser le transfert
                direct_condition = self.probability_matrix[j, self.dest] > self.probability_matrix[i, self.dest] * 0.8
                
                # 2. Critère absolu: P(j,dest) est suffisamment élevé (> 0.25)
                absolute_condition = self.probability_matrix[j, self.dest] > 0.25
                
                # 3. Critère d'opportunité: P(i,j) est élevé, indiquant des rencontres fréquentes
                opportunity_condition = self.probability_matrix[i, j] > 0.6
                
                # Condition finale combinant ces critères
                transfer_condition = (
                    direct_condition or
                    absolute_condition or
                    (opportunity_condition and self.probability_matrix[j, self.dest] > 0.1)
                )
                
                # Limiter les copies à un sous-ensemble du réseau pour réduire l'overhead
                overhead_control = (self.total_copies_created < min(len(self.probability_matrix) * 0.6, 25))
                                
                if (self.copies.get(i, 0) > 0 and 
                    self.copies.get(j, 0) == 0 and
                    transfer_condition and
                    overhead_control):
                    
                    # Transférer une copie du message
                    self.copies[j] = 1
                    self.copies_created_at[j] = t
                    
                    # Enregistrer la transmission
                    self.record_copy_transmission(i, j, 1, t)
                    
                    # Mise à jour du nombre de sauts et timestamp d'émission
                    self.num_hops[j] = self.num_hops.get(i, 0) + 1
                    if j not in self.t_emit:
                        self.t_emit[j] = float(t) + 0.5  # On ajoute 0.5 pour indiquer la moitié du pas de temps
                    
                    # Debug des transferts si le mode est activé
                    if self.debug_mode:
                        print(f"T={t}, Transfert: {i} -> {j}, Probabilités: P({i},{self.dest})={self.probability_matrix[i, self.dest]:.4f}, "
                              f"P({j},{self.dest})={self.probability_matrix[j, self.dest]:.4f}")
                
                # Livraison: Si j est la destination et i a une copie, livrer le message
                if j == self.dest and j not in self.delivered_at and self.copies.get(i, 0) > 0:
                    self.delivered_at[j] = t
                    t_recv = float(t) + 0.5  # Timestamp précis de réception
                    
                    # Marquer comme livré
                    self.message_delivered = True
                    
                    # Enregistrer le nombre de sauts pour la destination
                    self.num_hops[j] = self.num_hops.get(i, 0) + 1
                    
                    # Enregistrer l'historique de livraison
                    self.delivery_history.append({
                        't': t,
                        'from': i,
                        'to': j,
                        'hops': self.num_hops[j],
                        'travel_time': t - self.t_emit[self.source]
                    })
                    
                    # Journalisation
                    log_entry = {
                        'time': t,
                        'event': 'delivery',
                        'from': i,
                        'to': j,
                        'hops': self.num_hops[j],
                        'travel_time': t - self.t_emit[self.source]
                    }
                    self.packet_logs.append(log_entry)
    
    def record_copy_transmission(self, from_node: int, to_node: int, num_copies: int, t: int):
        """
        Enregistre une transmission de copie pour les statistiques.
        
        Args:
            from_node (int): Nœud source
            to_node (int): Nœud destinataire
            num_copies (int): Nombre de copies transmises
            t (int): Temps de la transmission
        """
        self.total_copies_created += num_copies
        
        self.copy_transmissions.append({
            't': t,
            'from': from_node,
            'to': to_node,
            'copies': num_copies
        })
    
    def delivery_ratio(self) -> float:
        """
        Calcule le ratio de livraison.
        
        Returns:
            float: 1.0 si le message est livré, 0.0 sinon
        """
        return 1.0 if self.message_delivered else 0.0
    
    def overhead_ratio(self) -> float:
        """
        Calcule le ratio d'overhead (surcharge réseau).
        
        Returns:
            float: Nombre de transmissions par message livré
        """
        if not self.message_delivered:
            return float('inf')
        
        # Total des copies créées moins 1 (la copie initiale)
        return (self.total_copies_created - 1) / 1.0
        
    def set_debug_mode(self, mode: bool):
        """
        Active ou désactive le mode debug pour afficher l'évolution des probabilités.
        
        Args:
            mode (bool): True pour activer, False pour désactiver
        """
        self.debug_mode = mode
    
    def analyze_probability_state(self):
        """
        Analyse l'état courant des probabilités pour détecter les problèmes de convergence.
        Cette fonction est utile pour le diagnostic.
        
        Returns:
            dict: Statistiques sur les probabilités courantes
        """
        stats = {
            "min_to_dest": float('inf'),
            "max_to_dest": 0,
            "avg_to_dest": 0,
            "nodes_above_threshold": 0,
            "potential_forwarders": []
        }
        
        # Analyser les probabilités vers la destination
        dest_probs = []
        for i in range(len(self.probability_matrix)):
            if i != self.dest:  # Exclure la destination elle-même
                prob = self.probability_matrix[i, self.dest]
                dest_probs.append(prob)
                
                if prob < stats["min_to_dest"]:
                    stats["min_to_dest"] = prob
                if prob > stats["max_to_dest"]:
                    stats["max_to_dest"] = prob
                
                # Compter les nœuds avec une probabilité significative
                if prob > 0.3:
                    stats["nodes_above_threshold"] += 1
                    stats["potential_forwarders"].append((i, prob))
        
        # Calculer la moyenne
        if dest_probs:
            stats["avg_to_dest"] = sum(dest_probs) / len(dest_probs)
            
        # Trier les forwarders potentiels par probabilité décroissante
        stats["potential_forwarders"].sort(key=lambda x: x[1], reverse=True)
        
        return stats
    
    def get_hop_stats(self):
        """
        Récupère les statistiques sur le nombre de sauts.
        
        Returns:
            dict: Dictionnaire avec les statistiques de sauts
        """
        return {
            'destination': self.num_hops.get(self.dest, None),
            'max': max(self.num_hops.values()) if self.num_hops else None,
            'average': sum(self.num_hops.values()) / len(self.num_hops) if self.num_hops else None
        }
    
    def get_probability_to_destination(self, node_id: int) -> float:
        """
        Retourne la probabilité qu'un nœud rencontre la destination.
        
        Args:
            node_id (int): ID du nœud
            
        Returns:
            float: Probabilité entre 0 et 1
        """
        return self.probability_matrix[node_id, self.dest]
