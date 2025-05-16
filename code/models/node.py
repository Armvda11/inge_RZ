# models/node.py
import numpy as np
from numpy.random import binomial
import math
from random import seed, choice

class Node:
    """
    Représente un nœud satellite dans un essaim.
    """
    
    def __init__(self, id, x=0.0, y=0.0, z=0.0):
        """
        Constructeur d'un objet Node
        
        Args:
            id (int): numéro d'identification du satellite (obligatoire)
            x (float, optional): coordonnée x du satellite. Par défaut 0.0.
            y (float, optional): coordonnée y du satellite. Par défaut 0.0.
            z (float, optional): coordonnée z du satellite. Par défaut 0.0.
        """
        self.id = int(id)
        self.pos = np.array([float(x), float(y), float(z)], dtype=float)
        self.neighbors = []  # Liste des nœuds voisins
        self.group = -1  # ID du groupe auquel appartient le nœud
        self.active = True  # État de fonctionnement du nœud (actif/inactif)
    
    def __str__(self):
        """
        Descripteur de l'objet Node
        
        Returns:
            str: description textuelle du nœud
        """
        nb_neigh = len(self.neighbors)
        x, y, z = self.pos
        return f"Node ID {self.id} ({x},{y},{z}) has {nb_neigh} neighbor(s)\tGroup: {self.group}"
    
    #*************** Opérations courantes ****************
    def add_neighbor(self, node):
        """
        Ajoute un nœud à la liste des voisins s'il n'y est pas déjà.
        
        Args:
            node (Node): le nœud à ajouter.
        """
        if node not in self.neighbors:
            self.neighbors.append(node)
    
    def distance_to(self, node):
        """
        Calcule la distance euclidienne entre deux nœuds.
        
        Args:
            node (Node): le nœud avec lequel calculer la distance.

        Returns:
            float: la distance euclidienne entre les deux nœuds.
        """
        return np.linalg.norm(self.pos - node.pos)
    
    def is_neighbor(self, node, connection_range=0):
        """
        Vérifie si deux nœuds sont voisins ou non, selon la portée de connexion.
        Ajoute ou supprime le second nœud de la liste des voisins du premier.
        
        Args:
            node (Node): le second nœud à analyser.
            connection_range (int, optional): distance maximale pour établir une connexion. Par défaut 0.

        Returns:
            int: 1 si voisins, 0 sinon.
        """
        # On vérifie que les deux nœuds sont actifs avant de calculer s'ils sont voisins
        if not self.active or not node.active:
            return 0
            
        if node.id != self.id:
            if self.distance_to(node) <= connection_range:
                self.add_neighbor(node)
                return 1 
            self.remove_neighbor(node)
        return 0
    
    def remove_neighbor(self, node):
        """
        Supprime un nœud de la liste des voisins s'il y est présent.
        
        Args:
            node (Node): le nœud à supprimer
        """
        if node in self.neighbors:
            self.neighbors.remove(node)   
     
    def set_group(self, c):
        """
        Attribue un identifiant de groupe au nœud.

        Args:
            c (int): identifiant de groupe.
        """
        self.group = c
    
    #*********** Métriques ***************   
    def cluster_coef(self):
        """
        Calcule le coefficient de clustering d'un nœud, défini comme
        le nombre de liens existants entre les voisins du nœud divisé par le nombre maximal
        possible de tels liens.

        Returns:
            float: le coefficient de clustering du nœud entre 0 et 1.
        """
        dv = self.degree()
        max_edges = dv*(dv-1)/2
        if max_edges == 0:
            return 0
        edges = 0
        for v in self.neighbors:
            common_elem = set(v.neighbors).intersection(self.neighbors)
            edges += len(common_elem)
        return edges/(2*max_edges)  # Division par 2 car chaque arête est comptée deux fois
                    
    def degree(self):
        """
        Calcule le degré (nombre de voisins) du nœud. Les listes de voisins doivent être établies
        avant d'exécuter cette fonction.
        
        Returns:
            int: le nombre de voisins du nœud.
        """
        return len(self.neighbors)
    
    def get_neighbors_ids(self):
        """
        Récupère les IDs des voisins du nœud.
        
        Returns:
            list(int): liste des IDs des voisins
        """
        return [neighbor.id for neighbor in self.neighbors]
                
    def k_vicinity(self, depth=1):
        """
        Calcule le k-voisinage (voisinage étendu) du nœud.
        Le k-voisinage correspond au nombre de voisins directs et indirects à au plus k sauts du nœud.

        Args:
            depth (int, optional): le nombre de sauts pour l'extension. Par défaut 1.

        Returns:
            int: la taille de la liste de voisins étendue du nœud.
        """
        kv = self.neighbors.copy()
        for i in range(depth-1):
            nodes = kv
            kv.extend([n for node in nodes for n in node.neighbors])
        return len(set(kv))   
    
    #*************** Algorithmes d'échantillonnage ****************
    def proba_walk(self, p:float, s=1, overlap=False):
        """
        Effectue un saut probabiliste du nœud vers son/ses voisin(s), généralement utilisé avec l'algorithme Forest Fire.
        Chaque nœud voisin a une probabilité p d'être choisi pour le prochain saut.

        Args:
            p (float): probabilité de succès entre 0 et 1.
            s (int, optional): graine aléatoire. Par défaut 1.
            overlap (bool, optional): si True, les groupes de nœuds peuvent se chevaucher. Par défaut False.

        Returns:
            list(Node): liste des nœuds voisins sélectionnés comme prochains sauts.
        """
        seed(s)
        search_list = self.neighbors
        if not overlap:  # Limiter la liste de recherche aux nœuds non assignés
            search_list = [n for n in self.neighbors if n.group == -1]
        trial = binomial(1, p, len(search_list))
        nodes = [n for i,n in enumerate(search_list) if trial[i] == 1]  # Sélectionner les nœuds qui ont obtenu un succès
        return nodes
    
    def random_group(self, clist, s=1):
        """
        Attribue un identifiant de groupe choisi aléatoirement dans la liste fournie.

        Args:
            clist (list(int)): liste des identifiants de groupe.
            s (int, optional): graine aléatoire. Par défaut 1.
        """
        seed(s)
        self.set_group(choice(clist))
        
    def random_walk(self, s=1, overlap=False):
        """
        Effectue une marche aléatoire à partir du nœud courant. Un de ses nœuds voisins est choisi aléatoirement comme prochain saut.

        Args:
            s (int, optional): graine aléatoire pour l'expérience. Par défaut 1.
            overlap (bool, optional): si True, les groupes de nœuds peuvent se chevaucher. Par défaut False.

        Returns:
            Node: le nœud voisin sélectionné comme prochain saut.
        """
        seed(s)
        search_list = self.neighbors
        if not overlap:  # Limiter la liste de recherche aux nœuds non assignés
            search_list = [n for n in self.neighbors if n.group == -1]
        return choice(search_list)