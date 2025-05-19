# models/swarm.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from random import choice, sample, seed

class Swarm:
    """
    Représente un essaim de nanosatellites.
    """
    
    def __init__(self, connection_range=0, nodes=None):
        """
        Constructeur d'un objet Swarm
        
        Args:
            connection_range (int, optional): distance maximale entre deux nœuds pour établir une connexion. Par défaut 0.
            nodes (list, optional): liste des objets Node dans l'essaim. Par défaut None.
        """
        self.connection_range = connection_range
        self.nodes = nodes if nodes else []
        
    def __str__(self):
        """
        Descripteur d'un objet Swarm
        
        Returns:
            str: description textuelle de l'essaim
        """
        nb_nodes = len(self.nodes)
        
        return f"Essaim avec {nb_nodes} satellite{'s' if nb_nodes > 1 else ''}, rayon de connexion {self.connection_range}"
    
    @property
    def edges(self):
        """
        Propriété qui calcule et renvoie la liste des arêtes du réseau.
        
        Returns:
            list: Liste des tuples (node1, node2) représentant les arêtes
        """
        edge_list = []
        # Pour éviter de compter deux fois la même arête
        seen = set()
        
        for node in self.nodes:
            for neighbor in node.neighbors:
                # Créer un tuple ordonné pour identifier l'arête de manière unique
                edge_id = tuple(sorted([node.id, neighbor.id]))
                if edge_id not in seen:
                    edge_list.append((node, neighbor))
                    seen.add(edge_id)
        
        return edge_list
    
    #*************** Opérations courantes ***************
    def add_node(self, node):
        """
        Ajoute un nœud à l'essaim s'il n'y est pas déjà.

        Args:
            node: le nœud à ajouter.
        """
        if node not in self.nodes:
            self.nodes.append(node)
            
    def distance_matrix(self):
        """
        Calcule la matrice des distances euclidiennes de l'essaim.

        Returns:
            list(list(float)): matrice des distances bidimensionnelle formatée comme matrix[node1][node2] = distance.
        """
        matrix = []
        for n1 in self.nodes:
            matrix.append([n1.distance_to(n2) for n2 in self.nodes if n1.id != n2.id])
        return matrix
    
    def get_node_by_id(self, id:int):
        """
        Récupère un objet Node dans l'essaim à partir de son ID.

        Args:
            id (int): l'ID du nœud.

        Returns:
            Node: l'objet Node avec l'ID correspondant.
        """
        for node in self.nodes:
            if node.id == id:
                return node
        return None
            
    def neighbor_matrix(self, connection_range=None):
        """
        Calcule la matrice de voisinage de l'essaim.
        Si deux nœuds sont voisins (selon la portée donnée), alors row[col] vaut 1. Sinon 0.

        Args:
            connection_range (int, optional): la portée de connexion de l'essaim. Par défaut None.

        Returns:
            list(list(int)): matrice de voisinage bidimensionnelle formatée comme matrix[node1][node2] = neighbor.
        """
        matrix = []
        if not connection_range:
            connection_range = self.connection_range  # Utilise l'attribut de l'objet Swarm si non spécifié
        for node in self.nodes:
            matrix.append([node.is_neighbor(nb, connection_range) for nb in self.nodes])
        return matrix
        
    def remove_node(self, node):
        """
        Supprime un nœud de l'essaim s'il s'y trouve.

        Args:
            node: le nœud à supprimer.
        """
        if node in self.nodes:
            self.nodes.remove(node)
        
    def reset_connection(self):
        """
        Vide la liste des voisins de chaque nœud de l'essaim.
        """
        for node in self.nodes:
            node.neighbors = []
            
    def reset_groups(self):
        """
        Réinitialise l'ID de groupe à -1 pour chaque nœud de l'essaim.
        """
        for node in self.nodes:
            node.set_group(-1)
    
    def swarm_to_nxgraph(self):
        """
        Convertit un objet Swarm en un graphe NetworkX.

        Returns:
            nx.Graph: le graphe converti.
        """
        G = nx.Graph()
        G.add_nodes_from([n.id for n in self.nodes])
        for ni in self.nodes:
            for nj in self.nodes:
                if ni.is_neighbor(nj, self.connection_range) == 1:
                    G.add_edge(ni.id, nj.id) 
        return G
    
    #*************** Métriques ******************
    def cluster_coef(self):
        """
        Calcule la distribution des coefficients de clustering de l'essaim.

        Returns:
            list(float): liste des coefficients de clustering entre 0 et 1.
        """
        return [node.cluster_coef() for node in self.nodes]
    
    def connected_components(self):
        """
        Définit les composantes connexes dans le réseau.

        Returns:
            list(list(int)): liste imbriquée des IDs de nœuds pour chaque composante connexe.
        """
        visited = {node.id: False for node in self.nodes}  # Initialiser tous les nœuds comme non visités
        cc = []
        for node in self.nodes:
            if not visited[node.id]:  # Effectue DFS sur chaque nœud non visité
                temp = []
                cc.append(self.DFSUtil(temp, node, visited))
        return cc
    
    def degree(self):
        """
        Calcule le degré (nombre de voisins) de chaque nœud dans l'essaim.

        Returns:
            list(int): liste des degrés des nœuds.
        """
        return [node.degree() for node in self.nodes]   
    
    def DFSUtil(self, temp, node, visited):
        """
        Effectue une recherche en profondeur sur le graphe, utilisé pour définir toutes les composantes connexes.

        Args:
            temp (list(int)): liste des IDs des nœuds visités jusqu'à présent.
            node (Node): le nœud à analyser.
            visited (dict(int:bool)): dictionnaire des correspondances entre l'ID du nœud et son état (visité ou non).

        Returns:
            list(int): la liste temp mise à jour.
        """
        visited[node.id] = True  # Marque le nœud actuel comme visité
        temp.append(node.id)  # Stocke le sommet dans la liste
        for n in node.neighbors:
            if n in self.nodes and not visited[n.id]:  # Effectue DFS sur les nœuds non visités
                temp = self.DFSUtil(temp, n, visited)
        return temp
    
    def diameter(self):
        """
        Calcule le diamètre de l'essaim, défini comme la distance du chemin le plus court maximal entre toutes les paires de nœuds.

        Returns:
            tuple: le diamètre de l'essaim sous forme de (source_id, target_id, diameter).
        """
        G = self.swarm_to_nxgraph()
        node_ids = [n.id for n in self.nodes]
        max_length = (0, 0, 0)  # Source, cible, nombre de sauts
        for ni in node_ids:
            for nj in node_ids:
                if nx.has_path(G, ni, nj):
                    sp = nx.shortest_path(G, ni, nj)
                    if len(sp) - 1 > max_length[2]:
                        max_length = (ni, nj, len(sp) - 1)
        return max_length
    
    def graph_density(self):
        """
        Calcule la densité du graphe de l'essaim, définie comme le rapport entre le nombre d'arêtes 
        et le nombre maximal d'arêtes possibles.

        Returns:
            float: la densité du graphe entre 0 et 1.
        """
        N = len(self.nodes)
        max_edges = N * (N - 1) / 2
        if max_edges == 0:
            return 0
        edges = 0
        for n in self.nodes:
            common_nodes = set(n.neighbors).intersection(self.nodes)
            edges += len(common_nodes)
        return edges / (2 * max_edges)  # Division par 2 car chaque arête est comptée deux fois
    
    def k_vicinity(self, depth=1):
        """
        Calcule le k-voisinage (voisinage étendu) de chaque nœud dans l'essaim.

        Args:
            depth (int, optional): le nombre de sauts pour l'extension. Par défaut 1.

        Returns:
            list(int): liste des valeurs de k-voisinage pour chaque nœud.
        """
        return [node.k_vicinity(depth) for node in self.nodes]
    
    def shortest_paths_lengths(self):
        """
        Calcule tous les plus courts chemins entre chaque paire de nœuds (algorithme de Dijkstra) 
        et renvoie leur longueur.

        Returns:
            list(int): la liste des longueurs des plus courts chemins.
        """
        G = self.swarm_to_nxgraph()
        node_ids = [n.id for n in self.nodes]
        lengths = []
        for ni in node_ids:
            lengths.append([])
            for nj in node_ids:
                if nx.has_path(G, ni, nj) and nj != ni:
                    lengths[ni-1].append(nx.shortest_path_length(G, source=ni, target=nj))
                else:
                    lengths[ni-1].append(0)
        return lengths 

    #************** Algorithmes d'échantillonnage ****************
    def ForestFire(self, n=10, p=0.7, s=1, overlap=False):
        """
        Effectue un échantillonnage de graphe selon l'algorithme Forest Fire.

        Args:
            n (int, optional): le nombre initial de sources. Par défaut 10.
            p (float, optional): la probabilité de propagation du feu. Par défaut 0.7.
            s (int, optional): la graine aléatoire. Par défaut 1.
            overlap (bool, optional): si True, les groupes de nœuds peuvent se chevaucher. Par défaut False.

        Returns:
            dict(int:Swarm): dictionnaire des IDs de groupe et leur échantillon Swarm correspondant.
        """
        sources = sample(self.nodes, n)  # Sources aléatoires initiales
        swarms = {}  # Dict(ID groupe:Swarm)
        for i, src in enumerate(sources):  # Initialiser les essaims
            src.set_group(i)
            swarms[i] = Swarm(self.connection_range, nodes=[src])
        free_nodes = [n for n in self.nodes if n.group == -1]
        burning_nodes = sources
        next_nodes = []
        while free_nodes:  # Propager les chemins depuis chaque nœud en feu en parallèle
            for bn in burning_nodes:
                if not free_nodes:
                    break
                free_neighbors = set(free_nodes).intersection(bn.neighbors)
                if free_neighbors:  # Au moins un voisin non assigné
                    nodes = bn.proba_walk(p, i, overlap)  # Prochain(s) nœud(s)
                else:
                    nodes = [self.random_jump(s, overlap)]  # Si pas de voisin, effectuer un saut aléatoire dans le graphe
                for n in nodes:
                    n.set_group(bn.group)
                    swarms[bn.group].add_node(n) 
                    free_nodes.remove(n)
                    next_nodes.append(n)
            burning_nodes = next_nodes
        return swarms
    
    def random_jump(self, s=1, overlap=False):
        """
        Choisit un nouveau nœud dans le graphe en effectuant un saut aléatoire.

        Args:
            s (int, optional): la graine aléatoire. Par défaut 1.
            overlap (bool, optional): si True, les groupes de nœuds peuvent se chevaucher. Par défaut False.

        Returns:
            Node: le nœud choisi aléatoirement.
        """
        seed(s)
        search_list = self.nodes
        if not overlap:  # Limiter la liste de recherche aux nœuds non assignés
            search_list = [n for n in self.nodes if n.group == -1]
        return choice(search_list)
    
    #************** Fonctions de traçage **************
    def plot_nodes(self, n_color='blue'):
        """
        Crée un tracé 3D de l'essaim à un instant donné.

        Args:
            n_color (str, optional): Couleur des nœuds. Par défaut 'blue'.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        x_data = [node.pos[0] for node in self.nodes]
        y_data = [node.pos[1] for node in self.nodes]
        z_data = [node.pos[2] for node in self.nodes]
        ax.scatter(x_data, y_data, z_data, c=n_color, s=50)
        ax.plot([0, 0], [0, 0], [0, 0], c='red', markersize=50000)  # Origine
        plt.show()
    
    def plot_edges(self, n_color='blue', e_color='gray'):
        """
        Crée un tracé 3D de la connectivité de l'essaim à un instant donné.

        Args:
            n_color (str, optional): Couleur des nœuds. Par défaut 'blue'.
            e_color (str, optional): Couleur des arêtes. Par défaut 'gray'.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        x_data = [node.pos[0] for node in self.nodes]
        y_data = [node.pos[1] for node in self.nodes]
        z_data = [node.pos[2] for node in self.nodes]
        ax.scatter(x_data, y_data, z_data, c=n_color, s=50)
        ax.plot([0, 0], [0, 0], [0, 0], color='red', markersize=50000)  # Origine
        for node in self.nodes:
            for n in node.neighbors:
                if n in self.nodes:
                    ax.plot([node.pos[0], n.pos[0]], [node.pos[1], n.pos[1]], [node.pos[2], n.pos[2]], c=e_color)
        plt.show()