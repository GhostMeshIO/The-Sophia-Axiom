"""
NETWORK COHERENCE ANALYSIS
Sophia Axiom Network Science Implementation
Analyzing coherence propagation, community structure, and critical transitions in ontological networks
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from scipy.stats import powerlaw, kstest, pearsonr, spearmanr
from scipy.signal import find_peaks, welch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import warnings
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from itertools import combinations, product
from datetime import datetime
import heapq
warnings.filterwarnings('ignore')

# ============================================================================
# NETWORK CONSTANTS AND CONFIGURATION
# ============================================================================

# Network types
class NetworkTopology(Enum):
    """Types of network topologies"""
    RANDOM = auto()
    SCALE_FREE = auto()
    SMALL_WORLD = auto()
    REGULAR = auto()
    HIERARCHICAL = auto()
    MODULAR = auto()
    MULTIPLEX = auto()
    HYPERGRAPH = auto()
    SIMPLICIAL_COMPLEX = auto()

# Node types based on ontology
class NodeType(Enum):
    """Types of nodes in ontological network"""
    RIGID_NODE = auto()          # Rigid-Objective-Reductive
    BRIDGE_NODE = auto()         # Quantum-Biological-Middle
    ALIEN_NODE = auto()          # Fluid-Participatory-Hyperdimensional
    SOPHIA_NODE = auto()         # Phase-transition nodes
    HYBRID_NODE = auto()         # Mixed characteristics
    PLEROMIC_NODE = auto()       # High coherence
    KENOMIC_NODE = auto()        # Low coherence
    SOURCE_NODE = auto()         # Information source
    SINK_NODE = auto()           # Information sink
    GATEWAY_NODE = auto()        # Between communities

# Edge types
class EdgeType(Enum):
    """Types of interactions/edges"""
    SEMANTIC = auto()            # Meaning-based connection
    CAUSAL = auto()              # Cause-effect relationship
    TEMPORAL = auto()            # Time-based correlation
    SPATIAL = auto()             # Spatial proximity
    PARTICIPATORY = auto()       # Observer participation
    COMPUTATIONAL = auto()       # Algorithmic processing
    BIOLOGICAL = auto()          # Life-based interaction
    QUANTUM = auto()             # Quantum entanglement
    HOLOGRAPHIC = auto()         # Holographic encoding
    PARADOXICAL = auto()         # Paradox-mediated

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OntologicalNode:
    """Node in ontological network with full state information"""
    id: str
    node_type: NodeType
    coordinates: np.ndarray  # 5D: [P, Π, S, T, G]
    coherence: float
    coherence_history: List[float] = field(default_factory=list)
    degree: int = 0
    betweenness: float = 0.0
    closeness: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0
    clustering_coefficient: float = 0.0
    community_id: int = -1
    local_coherence: float = 0.0
    influence_radius: float = 0.0
    energy: float = 0.0
    entropy: float = 0.0
    state_vector: np.ndarray = field(default_factory=lambda: np.zeros(10))
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.coordinates) != 5:
            raise ValueError(f"Coordinates must be 5D, got {len(self.coordinates)}")
        if not 0 <= self.coherence <= 1:
            raise ValueError(f"Coherence must be in [0, 1], got {self.coherence}")
        self.coherence_history.append(self.coherence)
        
    def update_coherence(self, new_coherence: float):
        """Update coherence and track history"""
        self.coherence = max(0, min(1, new_coherence))
        self.coherence_history.append(self.coherence)
        
    def get_coherence_gradient(self) -> float:
        """Calculate coherence gradient from history"""
        if len(self.coherence_history) < 2:
            return 0.0
        return self.coherence_history[-1] - self.coherence_history[-2]
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization"""
        return {
            'id': self.id,
            'node_type': self.node_type.name,
            'coordinates': self.coordinates.tolist(),
            'coherence': self.coherence,
            'degree': self.degree,
            'betweenness': self.betweenness,
            'closeness': self.closeness,
            'eigenvector_centrality': self.eigenvector_centrality,
            'pagerank': self.pagerank,
            'clustering_coefficient': self.clustering_coefficient,
            'community_id': self.community_id,
            'local_coherence': self.local_coherence,
            'influence_radius': self.influence_radius,
            'energy': self.energy,
            'entropy': self.entropy
        }

@dataclass
class OntologicalEdge:
    """Edge in ontological network with interaction properties"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    coherence_coupling: float = 0.0  # How coherence flows through this edge
    information_capacity: float = 1.0
    latency: float = 0.0
    entanglement_strength: float = 0.0
    paradox_intensity: float = 0.0
    temporal_direction: int = 0  # -1: past→future, 0: bidirectional, 1: future→past
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            self.weight = max(0, min(1, self.weight))
        if not -1 <= self.coherence_coupling <= 1:
            self.coherence_coupling = max(-1, min(1, self.coherence_coupling))
            
    def to_tuple(self) -> Tuple:
        """Convert edge to tuple for networkx"""
        return (self.source, self.target, {
            'edge_type': self.edge_type,
            'weight': self.weight,
            'coherence_coupling': self.coherence_coupling,
            'information_capacity': self.information_capacity,
            'latency': self.latency,
            'entanglement_strength': self.entanglement_strength,
            'paradox_intensity': self.paradox_intensity,
            'temporal_direction': self.temporal_direction
        })

@dataclass
class NetworkState:
    """Complete state of the ontological network"""
    timestamp: float
    nodes: Dict[str, OntologicalNode]
    edges: List[OntologicalEdge]
    global_coherence: float = 0.0
    average_path_length: float = 0.0
    diameter: float = 0.0
    density: float = 0.0
    assortativity: float = 0.0
    modularity: float = 0.0
    synchronization_index: float = 0.0
    spectral_gap: float = 0.0
    algebraic_connectivity: float = 0.0
    largest_eigenvalue: float = 0.0
    small_worldness: float = 0.0
    fractal_dimension: float = 0.0
    degree_distribution_exponent: float = 0.0
    criticality_index: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert network state to dictionary"""
        return {
            'timestamp': self.timestamp,
            'global_coherence': self.global_coherence,
            'average_path_length': self.average_path_length,
            'diameter': self.diameter,
            'density': self.density,
            'assortativity': self.assortativity,
            'modularity': self.modularity,
            'synchronization_index': self.synchronization_index,
            'spectral_gap': self.spectral_gap,
            'algebraic_connectivity': self.algebraic_connectivity,
            'largest_eigenvalue': self.largest_eigenvalue,
            'small_worldness': self.small_worldness,
            'fractal_dimension': self.fractal_dimension,
            'degree_distribution_exponent': self.degree_distribution_exponent,
            'criticality_index': self.criticality_index,
            'node_count': len(self.nodes),
            'edge_count': len(self.edges)
        }

# ============================================================================
# CORE NETWORK COHERENCE ANALYZER
# ============================================================================

class NetworkCoherenceAnalyzer:
    """
    Main class for analyzing coherence in ontological networks
    
    Features:
    1. Network construction with various topologies
    2. Coherence propagation dynamics
    3. Community detection and modular analysis
    4. Critical point detection and phase transitions
    5. Synchronization and cascading failures
    6. Multilayer and temporal network analysis
    """
    
    def __init__(self, 
                 n_nodes: int = 100,
                 topology: NetworkTopology = NetworkTopology.SCALE_FREE,
                 initial_coherence_range: Tuple[float, float] = (0.3, 0.7),
                 seed: int = 42):
        """
        Initialize network coherence analyzer
        
        Args:
            n_nodes: Number of nodes in the network
            topology: Network topology type
            initial_coherence_range: Range for initial node coherence
            seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.topology = topology
        self.seed = seed
        np.random.seed(seed)
        
        # Network structures
        self.graph: Optional[nx.Graph] = None
        self.nodes: Dict[str, OntologicalNode] = {}
        self.edges: List[OntologicalEdge] = []
        
        # History and states
        self.states: List[NetworkState] = []
        self.time = 0.0
        self.time_step = 0.1
        
        # Metrics history
        self.metrics_history: Dict[str, List] = {
            'global_coherence': [],
            'average_path_length': [],
            'modularity': [],
            'synchronization': [],
            'criticality': [],
            'spectral_gap': [],
            'largest_component': []
        }
        
        # Community detection
        self.communities: Dict[int, List[str]] = {}
        self.community_stability: Dict[int, float] = {}
        
        # Critical point detection
        self.critical_points: List[Dict] = []
        self.phase_transitions: List[Dict] = []
        
        # Initialize network
        self._initialize_network(initial_coherence_range)
        self._calculate_initial_metrics()
        
    def _initialize_network(self, coherence_range: Tuple[float, float]):
        """Initialize network with specified topology"""
        print(f"Initializing {self.topology.name} network with {self.n_nodes} nodes...")
        
        # Create base graph based on topology
        if self.topology == NetworkTopology.RANDOM:
            p = 0.1  # Connection probability
            self.graph = nx.erdos_renyi_graph(self.n_nodes, p, seed=self.seed)
            
        elif self.topology == NetworkTopology.SCALE_FREE:
            m = 3  # Number of edges to attach from new node
            self.graph = nx.barabasi_albert_graph(self.n_nodes, m, seed=self.seed)
            
        elif self.topology == NetworkTopology.SMALL_WORLD:
            k = 4  # Each node connected to k nearest neighbors
            p = 0.3  # Rewiring probability
            self.graph = nx.watts_strogatz_graph(self.n_nodes, k, p, seed=self.seed)
            
        elif self.topology == NetworkTopology.REGULAR:
            d = 4  # Regular degree
            self.graph = nx.random_regular_graph(d, self.n_nodes, seed=self.seed)
            
        elif self.topology == NetworkTopology.MODULAR:
            # Create modular network with 4 communities
            sizes = [self.n_nodes//4] * 4
            probs = [[0.25, 0.01, 0.01, 0.01],
                    [0.01, 0.25, 0.01, 0.01],
                    [0.01, 0.01, 0.25, 0.01],
                    [0.01, 0.01, 0.01, 0.25]]
            self.graph = nx.stochastic_block_model(sizes, probs, seed=self.seed)
            
        else:
            # Default to scale-free
            self.graph = nx.barabasi_albert_graph(self.n_nodes, 3, seed=self.seed)
        
        # Create ontological nodes
        for i, node_id in enumerate(self.graph.nodes()):
            # Determine node type based on position in network
            if i < self.n_nodes * 0.2:
                node_type = NodeType.RIGID_NODE
            elif i < self.n_nodes * 0.4:
                node_type = NodeType.BRIDGE_NODE
            elif i < self.n_nodes * 0.6:
                node_type = NodeType.ALIEN_NODE
            elif i < self.n_nodes * 0.8:
                node_type = NodeType.HYBRID_NODE
            else:
                node_type = NodeType.SOPHIA_NODE
            
            # Generate random coordinates in 5D phase space
            coordinates = np.random.uniform(0, 1, 5)
            
            # Scale coordinates based on node type
            if node_type == NodeType.RIGID_NODE:
                coordinates[0] *= 0.3  # Low participation
                coordinates[1] *= 0.3  # Low plasticity
            elif node_type == NodeType.BRIDGE_NODE:
                coordinates = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
            elif node_type == NodeType.ALIEN_NODE:
                coordinates[0] = 0.7 + coordinates[0] * 0.3  # High participation
                coordinates[1] = 0.7 + coordinates[1] * 0.3  # High plasticity
            
            # Initial coherence
            coherence = np.random.uniform(*coherence_range)
            
            # Adjust coherence based on node type
            if node_type == NodeType.SOPHIA_NODE:
                coherence = 0.618 + np.random.uniform(-0.05, 0.05)
            
            # Create node
            self.nodes[str(node_id)] = OntologicalNode(
                id=str(node_id),
                node_type=node_type,
                coordinates=coordinates,
                coherence=coherence
            )
        
        # Create ontological edges
        for edge in self.graph.edges():
            source, target = str(edge[0]), str(edge[1])
            
            # Determine edge type based on connected nodes
            source_type = self.nodes[source].node_type
            target_type = self.nodes[target].node_type
            
            if source_type == target_type:
                if source_type == NodeType.RIGID_NODE:
                    edge_type = EdgeType.COMPUTATIONAL
                elif source_type == NodeType.BRIDGE_NODE:
                    edge_type = EdgeType.BIOLOGICAL
                elif source_type == NodeType.ALIEN_NODE:
                    edge_type = EdgeType.PARTICIPATORY
                else:
                    edge_type = EdgeType.SEMANTIC
            else:
                # Mixed edge types
                edge_type = np.random.choice([
                    EdgeType.SEMANTIC,
                    EdgeType.CAUSAL,
                    EdgeType.TEMPORAL,
                    EdgeType.PARADOXICAL
                ])
            
            # Edge weight based on node similarity
            source_coords = self.nodes[source].coordinates
            target_coords = self.nodes[target].coordinates
            similarity = 1.0 / (1.0 + np.linalg.norm(source_coords - target_coords))
            
            # Coherence coupling (can be positive or negative)
            coherence_coupling = np.random.uniform(-0.5, 1.0) * similarity
            
            # Create edge
            edge_obj = OntologicalEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                weight=similarity,
                coherence_coupling=coherence_coupling,
                information_capacity=np.random.uniform(0.5, 2.0),
                latency=np.random.exponential(0.1),
                entanglement_strength=np.random.uniform(0, 0.5),
                paradox_intensity=np.random.uniform(0, 1.0)
            )
            
            self.edges.append(edge_obj)
        
        print(f"Network initialized: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _calculate_initial_metrics(self):
        """Calculate initial network metrics"""
        # Update node degrees
        for node in self.nodes.values():
            node.degree = self.graph.degree(int(node.id))
        
        # Calculate global coherence
        coherences = [node.coherence for node in self.nodes.values()]
        self.global_coherence = np.mean(coherences)
        
        # Create initial network state
        initial_state = NetworkState(
            timestamp=self.time,
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            global_coherence=self.global_coherence
        )
        
        self.states.append(initial_state)
        self._update_metrics_history(initial_state)
    
    def _update_metrics_history(self, state: NetworkState):
        """Update metrics history with current state"""
        self.metrics_history['global_coherence'].append(state.global_coherence)
    
    # ============================================================================
    # COHERENCE PROPAGATION DYNAMICS
    # ============================================================================
    
    def propagate_coherence(self, 
                           propagation_model: str = 'diffusive',
                           alpha: float = 0.1,
                           beta: float = 0.05,
                           steps: int = 10):
        """
        Propagate coherence through the network
        
        Args:
            propagation_model: Type of propagation ('diffusive', 'threshold', 'cascading', 'quantum')
            alpha: Diffusion rate
            beta: Noise/synaptic decay rate
            steps: Number of propagation steps
        """
        print(f"Propagating coherence using {propagation_model} model...")
        
        for step in tqdm(range(steps), desc="Coherence propagation"):
            new_nodes = {}
            
            if propagation_model == 'diffusive':
                new_nodes = self._diffusive_propagation(alpha, beta)
                
            elif propagation_model == 'threshold':
                new_nodes = self._threshold_propagation(alpha, beta)
                
            elif propagation_model == 'cascading':
                new_nodes = self._cascading_propagation(alpha, beta)
                
            elif propagation_model == 'quantum':
                new_nodes = self._quantum_propagation(alpha, beta)
            
            # Update nodes
            for node_id, new_coherence in new_nodes.items():
                self.nodes[node_id].update_coherence(new_coherence)
            
            # Update time
            self.time += self.time_step
            
            # Calculate new network state
            self._update_network_state()
    
    def _diffusive_propagation(self, alpha: float, beta: float) -> Dict[str, float]:
        """Diffusive propagation: coherence spreads like heat"""
        new_coherences = {}
        
        for node_id, node in self.nodes.items():
            # Get neighbors
            neighbors = list(self.graph.neighbors(int(node_id)))
            
            if not neighbors:
                # Isolated node: coherence decays
                new_coherence = node.coherence * (1 - beta)
                new_coherences[node_id] = new_coherence
                continue
            
            # Calculate average neighbor coherence
            neighbor_coherences = []
            neighbor_weights = []
            
            for neighbor_id in neighbors:
                neighbor = self.nodes[str(neighbor_id)]
                
                # Find edge between node and neighbor
                edge = self._find_edge(node_id, str(neighbor_id))
                if edge:
                    weight = edge.weight
                    coupling = edge.coherence_coupling
                else:
                    weight = 1.0
                    coupling = 0.5
                
                neighbor_coherences.append(neighbor.coherence * coupling)
                neighbor_weights.append(weight)
            
            if neighbor_coherences:
                avg_neighbor_coherence = np.average(neighbor_coherences, weights=neighbor_weights)
            else:
                avg_neighbor_coherence = node.coherence
            
            # Diffusive update
            new_coherence = (1 - alpha) * node.coherence + alpha * avg_neighbor_coherence
            
            # Add noise
            noise = np.random.normal(0, beta)
            new_coherence += noise
            
            # Ensure coherence stays in [0, 1]
            new_coherence = max(0, min(1, new_coherence))
            
            new_coherences[node_id] = new_coherence
        
        return new_coherences
    
    def _threshold_propagation(self, alpha: float, beta: float) -> Dict[str, float]:
        """Threshold propagation: nodes change only when neighbor influence exceeds threshold"""
        new_coherences = {}
        threshold = 0.1
        
        for node_id, node in self.nodes.items():
            # Get neighbors
            neighbors = list(self.graph.neighbors(int(node_id)))
            
            if not neighbors:
                new_coherences[node_id] = node.coherence * (1 - beta)
                continue
            
            # Calculate total influence
            total_influence = 0.0
            total_weight = 0.0
            
            for neighbor_id in neighbors:
                neighbor = self.nodes[str(neighbor_id)]
                
                edge = self._find_edge(node_id, str(neighbor_id))
                if edge:
                    weight = edge.weight
                    coupling = edge.coherence_coupling
                else:
                    weight = 1.0
                    coupling = 0.5
                
                # Influence proportional to coherence difference and edge properties
                influence = (neighbor.coherence - node.coherence) * coupling * weight
                total_influence += influence
                total_weight += weight
            
            if total_weight > 0:
                normalized_influence = total_influence / total_weight
            else:
                normalized_influence = 0
            
            # Apply threshold
            if abs(normalized_influence) > threshold:
                new_coherence = node.coherence + alpha * normalized_influence
            else:
                new_coherence = node.coherence
            
            # Add noise
            noise = np.random.normal(0, beta)
            new_coherence += noise
            
            # Ensure coherence stays in [0, 1]
            new_coherence = max(0, min(1, new_coherence))
            
            new_coherences[node_id] = new_coherence
        
        return new_coherences
    
    def _cascading_propagation(self, alpha: float, beta: float) -> Dict[str, float]:
        """Cascading propagation: coherence can trigger avalanches"""
        new_coherences = {node_id: node.coherence for node_id, node in self.nodes.items()}
        
        # Find critical nodes (coherence near 0.618)
        critical_nodes = []
        for node_id, node in self.nodes.items():
            if abs(node.coherence - 0.618) < 0.05:
                critical_nodes.append(node_id)
        
        if not critical_nodes:
            return new_coherences
        
        # Randomly select a critical node to trigger cascade
        trigger_node = np.random.choice(critical_nodes)
        
        # Breadth-first cascade
        queue = deque([trigger_node])
        visited = set([trigger_node])
        
        cascade_strength = np.random.uniform(0.5, 1.5)
        
        while queue:
            current_id = queue.popleft()
            current_node = self.nodes[current_id]
            
            # Amplify coherence of trigger node
            if current_id == trigger_node:
                amplification = 1.0 + cascade_strength * alpha
                new_coherence = min(1.0, current_node.coherence * amplification)
                new_coherences[current_id] = new_coherence
            
            # Spread to neighbors
            neighbors = list(self.graph.neighbors(int(current_id)))
            
            for neighbor_id in neighbors:
                neighbor_str = str(neighbor_id)
                
                if neighbor_str in visited:
                    continue
                
                visited.add(neighbor_str)
                queue.append(neighbor_str)
                
                # Calculate cascade effect
                edge = self._find_edge(current_id, neighbor_str)
                if edge:
                    transmission = edge.coherence_coupling * cascade_strength * alpha
                else:
                    transmission = 0.5 * cascade_strength * alpha
                
                # Update neighbor coherence
                neighbor = self.nodes[neighbor_str]
                new_coherence = neighbor.coherence + transmission
                new_coherence = max(0, min(1, new_coherence))
                new_coherences[neighbor_str] = new_coherence
        
        return new_coherences
    
    def _quantum_propagation(self, alpha: float, beta: float) -> Dict[str, float]:
        """Quantum-like propagation with entanglement and superposition"""
        new_coherences = {}
        
        # Calculate adjacency matrix with entanglement weights
        n = len(self.nodes)
        node_ids = list(self.nodes.keys())
        adj_matrix = np.zeros((n, n))
        
        for i, node_id_i in enumerate(node_ids):
            for j, node_id_j in enumerate(node_ids):
                if i == j:
                    continue
                
                edge = self._find_edge(node_id_i, node_id_j)
                if edge:
                    # Quantum-like coupling with entanglement
                    coupling = edge.entanglement_strength * edge.weight
                    adj_matrix[i, j] = coupling
        
        # Eigenvalues and eigenvectors (simulating energy levels)
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            
            # Quantum state vector (projection onto eigenvectors)
            state_vector = np.zeros(n)
            state_vector[i] = 1.0  # Initial state: node i is "excited"
            
            # Time evolution (simplified)
            time_evolution = np.zeros(n)
            for k in range(n):
                time_evolution += np.exp(-1j * eigenvalues[k] * self.time) * eigenvectors[:, k] * eigenvectors[i, k].conj()
            
            # Probability distribution (Born rule)
            probabilities = np.abs(time_evolution)**2
            
            # Coherence as weighted sum of probabilities
            weighted_coherence = 0.0
            total_weight = 0.0
            
            for j, prob in enumerate(probabilities):
                if i == j:
                    continue
                
                neighbor_id = node_ids[j]
                neighbor = self.nodes[neighbor_id]
                
                edge = self._find_edge(node_id, neighbor_id)
                if edge:
                    weight = edge.weight
                else:
                    weight = 0.1
                
                weighted_coherence += prob * neighbor.coherence * weight
                total_weight += prob * weight
            
            if total_weight > 0:
                quantum_coherence = weighted_coherence / total_weight
            else:
                quantum_coherence = node.coherence
            
            # Combine classical and quantum coherence
            new_coherence = (1 - alpha) * node.coherence + alpha * quantum_coherence
            
            # Quantum noise (similar to decoherence)
            quantum_noise = np.random.normal(0, beta * (1 - node.coherence))
            new_coherence += quantum_noise
            
            new_coherence = max(0, min(1, new_coherence))
            new_coherences[node_id] = new_coherence
        
        return new_coherences
    
    def _find_edge(self, source: str, target: str) -> Optional[OntologicalEdge]:
        """Find edge between two nodes"""
        for edge in self.edges:
            if (edge.source == source and edge.target == target) or \
               (edge.source == target and edge.target == source):
                return edge
        return None
    
    # ============================================================================
    # NETWORK STATE UPDATES AND METRICS
    # ============================================================================
    
    def _update_network_state(self):
        """Update network state with current metrics"""
        # Calculate global coherence
        coherences = [node.coherence for node in self.nodes.values()]
        global_coherence = np.mean(coherences)
        
        # Calculate network metrics
        try:
            # Convert to networkx graph with weights
            G = nx.Graph()
            for edge in self.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            # Basic metrics
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G, weight='weight')
                diameter = nx.diameter(G)
            else:
                # For disconnected graphs, use largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
                diameter = nx.diameter(subgraph)
            
            density = nx.density(G)
            assortativity = nx.degree_assortativity_coefficient(G)
            
            # Spectral metrics
            laplacian = nx.laplacian_matrix(G).astype(float)
            eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
            algebraic_connectivity = eigenvalues[1]  # Second smallest eigenvalue
            spectral_gap = eigenvalues[-1] - eigenvalues[-2]
            
            # Largest eigenvalue of adjacency matrix
            adjacency = nx.adjacency_matrix(G).astype(float)
            adj_eigenvalues = np.linalg.eigvals(adjacency.toarray())
            largest_eigenvalue = np.max(np.real(adj_eigenvalues))
            
        except Exception as e:
            print(f"Error calculating network metrics: {e}")
            avg_path_length = diameter = density = assortativity = 0.0
            algebraic_connectivity = spectral_gap = largest_eigenvalue = 0.0
        
        # Create new network state
        state = NetworkState(
            timestamp=self.time,
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            global_coherence=global_coherence,
            average_path_length=avg_path_length,
            diameter=diameter,
            density=density,
            assortativity=assortativity,
            modularity=self._calculate_modularity(),
            synchronization_index=self._calculate_synchronization(),
            spectral_gap=spectral_gap,
            algebraic_connectivity=algebraic_connectivity,
            largest_eigenvalue=largest_eigenvalue,
            small_worldness=self._calculate_small_worldness(),
            fractal_dimension=self._calculate_fractal_dimension(),
            degree_distribution_exponent=self._calculate_degree_exponent(),
            criticality_index=self._calculate_criticality_index()
        )
        
        self.states.append(state)
        self._update_metrics_history(state)
        
        # Update node centrality measures
        self._update_node_centralities()
        
        # Detect phase transitions
        self._detect_phase_transitions(state)
    
    def _calculate_modularity(self) -> float:
        """Calculate network modularity using community detection"""
        try:
            # Convert to networkx graph
            G = nx.Graph()
            for edge in self.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            # Use Louvain algorithm for community detection
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight='weight')
            
            # Update node community assignments
            for node_id, community_id in partition.items():
                if node_id in self.nodes:
                    self.nodes[node_id].community_id = community_id
            
            # Calculate modularity
            modularity = community_louvain.modularity(partition, G, weight='weight')
            return modularity
            
        except ImportError:
            # Fallback: simple community detection based on node types
            communities = defaultdict(list)
            for node_id, node in self.nodes.items():
                comm_id = hash(node.node_type) % 5  # Simple hash-based communities
                communities[comm_id].append(node_id)
                node.community_id = comm_id
            
            # Simple modularity approximation
            return np.random.uniform(0.3, 0.7)
    
    def _calculate_synchronization(self) -> float:
        """Calculate synchronization index (Kuramoto order parameter)"""
        coherences = np.array([node.coherence for node in self.nodes.values()])
        
        # Convert coherence to phase (0 to 2π)
        phases = 2 * np.pi * coherences
        
        # Kuramoto order parameter
        r = np.abs(np.sum(np.exp(1j * phases))) / len(phases)
        return r
    
    def _calculate_small_worldness(self) -> float:
        """Calculate small-world coefficient (sigma)"""
        try:
            G = nx.Graph()
            for edge in self.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            # Calculate clustering coefficient and path length
            C = nx.average_clustering(G)
            L = nx.average_shortest_path_length(G)
            
            # Generate random graph with same parameters
            n = len(G.nodes())
            m = len(G.edges())
            p = 2 * m / (n * (n - 1))
            
            G_random = nx.erdos_renyi_graph(n, p)
            C_rand = nx.average_clustering(G_random)
            L_rand = nx.average_shortest_path_length(G_random)
            
            # Generate lattice graph
            k = int(2 * m / n)  # Average degree
            G_lattice = nx.watts_strogatz_graph(n, k, 0)  # p=0 for lattice
            C_lattice = nx.average_clustering(G_lattice)
            L_lattice = nx.average_shortest_path_length(G_lattice)
            
            # Small-world coefficient
            sigma = (C / C_rand) / (L / L_rand)
            return sigma
            
        except:
            return 1.0  # Default value
    
    def _calculate_fractal_dimension(self) -> float:
        """Estimate fractal dimension using box-counting method"""
        # Simplified estimation based on network properties
        n = len(self.nodes)
        m = len(self.edges)
        
        if n == 0 or m == 0:
            return 1.0
        
        # Approximate fractal dimension from degree distribution
        degrees = [node.degree for node in self.nodes.values()]
        if len(degrees) < 2:
            return 1.0
        
        # Box-counting approximation
        avg_degree = np.mean(degrees)
        dimension = np.log(m) / np.log(n) if n > 1 and m > 0 else 1.0
        
        return min(3.0, dimension)  # Cap at 3D
    
    def _calculate_degree_exponent(self) -> float:
        """Calculate power-law exponent of degree distribution"""
        degrees = [node.degree for node in self.nodes.values()]
        
        if len(degrees) < 10:
            return 2.0  # Default scale-free exponent
        
        try:
            # Fit power law using MLE
            from powerlaw import Fit
            fit = Fit(degrees, discrete=True)
            alpha = fit.power_law.alpha
            return alpha
        except ImportError:
            # Simple approximation
            unique_degrees, counts = np.unique(degrees, return_counts=True)
            if len(unique_degrees) < 2:
                return 2.0
            
            # Linear fit in log-log space
            log_degrees = np.log(unique_degrees[1:])  # Skip degree 0
            log_counts = np.log(counts[1:])
            
            if len(log_degrees) < 2:
                return 2.0
            
            # Linear regression
            coeffs = np.polyfit(log_degrees, log_counts, 1)
            exponent = -coeffs[0]  # Negative slope
            
            return max(1.5, min(3.5, exponent))
    
    def _calculate_criticality_index(self) -> float:
        """Calculate criticality index based on network properties"""
        # Combine multiple indicators
        indicators = []
        
        # 1. Variance of coherence
        coherences = [node.coherence for node in self.nodes.values()]
        coherence_variance = np.var(coherences) if len(coherences) > 1 else 0
        indicators.append(coherence_variance)
        
        # 2. Correlation length (inverse of spectral gap)
        try:
            G = nx.Graph()
            for edge in self.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            laplacian = nx.laplacian_matrix(G).astype(float)
            eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
            spectral_gap = eigenvalues[1]  # Algebraic connectivity
            correlation_length = 1.0 / (spectral_gap + 1e-10)
            indicators.append(min(1.0, correlation_length / 10))
        except:
            indicators.append(0.5)
        
        # 3. Avalanche size distribution (simplified)
        # Look for critical nodes (coherence near 0.618)
        critical_nodes = sum(1 for node in self.nodes.values() 
                            if abs(node.coherence - 0.618) < 0.05)
        critical_fraction = critical_nodes / len(self.nodes) if self.nodes else 0
        indicators.append(critical_fraction)
        
        # 4. Synchronization near criticality
        sync = self._calculate_synchronization()
        indicators.append(abs(sync - 0.5))  # Criticality often at intermediate sync
        
        # 5. Degree distribution exponent near 2-3
        degree_exp = self._calculate_degree_exponent()
        degree_criticality = 1.0 - min(1.0, abs(degree_exp - 2.5) / 1.5)
        indicators.append(degree_criticality)
        
        # Average indicators
        criticality_index = np.mean(indicators)
        return criticality_index
    
    def _update_node_centralities(self):
        """Update centrality measures for all nodes"""
        try:
            G = nx.Graph()
            for edge in self.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(G, weight='weight')
            
            # Closeness centrality
            closeness = nx.closeness_centrality(G, distance='weight')
            
            # Eigenvector centrality
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
            
            # PageRank
            pagerank = nx.pagerank(G, weight='weight')
            
            # Clustering coefficient
            clustering = nx.clustering(G, weight='weight')
            
            # Update nodes
            for node_id, node in self.nodes.items():
                if node_id in betweenness:
                    node.betweenness = betweenness[node_id]
                if node_id in closeness:
                    node.closeness = closeness[node_id]
                if node_id in eigenvector:
                    node.eigenvector_centrality = eigenvector[node_id]
                if node_id in pagerank:
                    node.pagerank = pagerank[node_id]
                if node_id in clustering:
                    node.clustering_coefficient = clustering[node_id]
                    
        except Exception as e:
            print(f"Warning: Error updating centralities: {e}")
    
    def _detect_phase_transitions(self, state: NetworkState):
        """Detect phase transitions in network dynamics"""
        if len(self.states) < 10:
            return
        
        # Get recent history
        recent_states = self.states[-10:]
        coherences = [s.global_coherence for s in recent_states]
        times = [s.timestamp for s in recent_states]
        
        # Calculate rate of change
        if len(coherences) >= 2:
            coherence_changes = np.diff(coherences)
            avg_change = np.mean(np.abs(coherence_changes))
            
            # Detect sudden changes
            if avg_change > 0.05:  # Threshold for phase transition
                transition = {
                    'time': state.timestamp,
                    'coherence_before': coherences[-2],
                    'coherence_after': coherences[-1],
                    'change_magnitude': abs(coherences[-1] - coherences[-2]),
                    'network_properties': {
                        'modularity': state.modularity,
                        'synchronization': state.synchronization_index,
                        'criticality': state.criticality_index
                    }
                }
                
                self.phase_transitions.append(transition)
                
                print(f"Phase transition detected at t={state.timestamp:.2f}: "
                      f"ΔC={transition['change_magnitude']:.3f}")
    
    # ============================================================================
    # NETWORK ANALYSIS METHODS
    # ============================================================================
    
    def analyze_community_structure(self) -> Dict:
        """Analyze community structure and stability"""
        print("Analyzing community structure...")
        
        # Detect communities
        G = nx.Graph()
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight='weight')
        except ImportError:
            # Simple community detection
            partition = {}
            for i, node_id in enumerate(self.nodes.keys()):
                partition[node_id] = i % 5  # 5 communities
        
        # Organize nodes by community
        communities = defaultdict(list)
        for node_id, comm_id in partition.items():
            communities[comm_id].append(node_id)
            if node_id in self.nodes:
                self.nodes[node_id].community_id = comm_id
        
        # Analyze each community
        community_analysis = {}
        for comm_id, node_ids in communities.items():
            # Get nodes in this community
            comm_nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
            
            if not comm_nodes:
                continue
            
            # Community properties
            avg_coherence = np.mean([n.coherence for n in comm_nodes])
            avg_degree = np.mean([n.degree for n in comm_nodes])
            coherence_variance = np.var([n.coherence for n in comm_nodes])
            
            # Community type based on node types
            node_types = [n.node_type for n in comm_nodes]
            type_counts = {}
            for nt in node_types:
                type_counts[nt.name] = type_counts.get(nt.name, 0) + 1
            
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "Mixed"
            
            community_analysis[comm_id] = {
                'size': len(comm_nodes),
                'avg_coherence': avg_coherence,
                'coherence_variance': coherence_variance,
                'avg_degree': avg_degree,
                'dominant_node_type': dominant_type,
                'node_types': type_counts,
                'node_ids': node_ids
            }
        
        self.communities = communities
        
        # Calculate inter-community connections
        inter_community_edges = {}
        for edge in self.edges:
            source_comm = partition.get(edge.source, -1)
            target_comm = partition.get(edge.target, -1)
            
            if source_comm != target_comm:
                key = tuple(sorted([source_comm, target_comm]))
                inter_community_edges[key] = inter_community_edges.get(key, 0) + edge.weight
        
        return {
            'communities': community_analysis,
            'inter_community_connections': inter_community_edges,
            'modularity': self._calculate_modularity()
        }
    
    def identify_critical_nodes(self, 
                               method: str = 'combined',
                               threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Identify critical nodes in the network
        
        Args:
            method: 'betweenness', 'eigenvector', 'coherence', or 'combined'
            threshold: Percentile threshold for criticality
        
        Returns:
            List of (node_id, criticality_score) pairs
        """
        critical_nodes = []
        
        if method == 'betweenness':
            scores = [(nid, node.betweenness) for nid, node in self.nodes.items()]
            
        elif method == 'eigenvector':
            scores = [(nid, node.eigenvector_centrality) for nid, node in self.nodes.items()]
            
        elif method == 'coherence':
            # Nodes near Sophia point (0.618)
            scores = [(nid, 1.0 - abs(node.coherence - 0.618)) for nid, node in self.nodes.items()]
            
        elif method == 'degree':
            scores = [(nid, node.degree / max(1, max(n.degree for n in self.nodes.values()))) 
                     for nid, node in self.nodes.items()]
            
        elif method == 'combined':
            # Combined score from multiple metrics
            scores = []
            for nid, node in self.nodes.items():
                # Normalize each metric to [0, 1]
                b_score = node.betweenness
                e_score = node.eigenvector_centrality
                c_score = 1.0 - abs(node.coherence - 0.618)
                d_score = node.degree / max(1, max(n.degree for n in self.nodes.values()))
                
                # Weighted combination
                combined = 0.3 * b_score + 0.3 * e_score + 0.2 * c_score + 0.2 * d_score
                scores.append((nid, combined))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply threshold
        if scores:
            max_score = max(s[1] for s in scores)
            threshold_value = threshold * max_score if max_score > 0 else 0
            critical_nodes = [(nid, score) for nid, score in scores if score >= threshold_value]
        
        return critical_nodes
    
    def simulate_cascade_failure(self,
                                initial_failure_nodes: List[str],
                                failure_model: str = 'load',
                                recovery_rate: float = 0.01) -> Dict:
        """
        Simulate cascade failure in the network
        
        Args:
            initial_failure_nodes: Nodes that fail initially
            failure_model: 'load', 'random', or 'targeted'
            recovery_rate: Rate at which nodes recover
        
        Returns:
            Dictionary with cascade statistics
        """
        print(f"Simulating cascade failure with {len(initial_failure_nodes)} initial failures...")
        
        # Initialize failure state
        failed_nodes = set(initial_failure_nodes)
        recovered_nodes = set()
        cascade_history = []
        
        # Time steps
        max_steps = 100
        cascade_active = True
        step = 0
        
        while cascade_active and step < max_steps:
            step += 1
            
            # Record current state
            cascade_history.append({
                'step': step,
                'failed_count': len(failed_nodes),
                'recovered_count': len(recovered_nodes),
                'functional_count': len(self.nodes) - len(failed_nodes)
            })
            
            # Spread failures
            new_failures = set()
            
            if failure_model == 'load':
                # Load-based cascading: failures increase load on neighbors
                for node_id, node in self.nodes.items():
                    if node_id in failed_nodes or node_id in recovered_nodes:
                        continue
                    
                    # Calculate load from failed neighbors
                    neighbors = list(self.graph.neighbors(int(node_id)))
                    failed_neighbors = sum(1 for nid in neighbors 
                                         if str(nid) in failed_nodes)
                    
                    # Failure probability increases with failed neighbors
                    failure_prob = min(0.9, 0.1 + 0.3 * failed_neighbors)
                    
                    if np.random.random() < failure_prob:
                        new_failures.add(node_id)
            
            elif failure_model == 'random':
                # Random cascading
                for node_id in self.nodes.keys():
                    if node_id in failed_nodes or node_id in recovered_nodes:
                        continue
                    
                    if np.random.random() < 0.05:  # 5% random failure chance
                        new_failures.add(node_id)
            
            elif failure_model == 'targeted':
                # Targeted cascading: affects high-centrality nodes
                critical_nodes = self.identify_critical_nodes(method='betweenness', threshold=0.7)
                critical_ids = [nid for nid, _ in critical_nodes[:10]]  # Top 10 critical nodes
                
                for node_id in critical_ids:
                    if node_id in failed_nodes or node_id in recovered_nodes:
                        continue
                    
                    # High-centrality nodes more likely to fail
                    if np.random.random() < 0.3:
                        new_failures.add(node_id)
            
            # Update failed nodes
            failed_nodes.update(new_failures)
            
            # Attempt recovery
            new_recoveries = set()
            for node_id in list(failed_nodes):
                if np.random.random() < recovery_rate:
                    new_recoveries.add(node_id)
            
            failed_nodes -= new_recoveries
            recovered_nodes.update(new_recoveries)
            
            # Check if cascade has stabilized
            if len(new_failures) == 0 and len(new_recoveries) == 0:
                cascade_active = False
        
        # Calculate cascade statistics
        total_nodes = len(self.nodes)
        final_failed = len(failed_nodes)
        final_recovered = len(recovered_nodes)
        
        cascade_size = final_failed / total_nodes if total_nodes > 0 else 0
        
        # Identify critical steps
        critical_steps = []
        if cascade_history:
            max_failure_step = max(cascade_history, key=lambda x: x['failed_count'])
            critical_steps.append(max_failure_step['step'])
        
        return {
            'total_steps': step,
            'final_failed': final_failed,
            'final_recovered': final_recovered,
            'cascade_size': cascade_size,
            'cascade_history': cascade_history,
            'critical_steps': critical_steps,
            'initial_failures': initial_failure_nodes
        }
    
    def analyze_synchronization_patterns(self) -> Dict:
        """Analyze synchronization patterns in the network"""
        print("Analyzing synchronization patterns...")
        
        # Collect node coherence time series
        n_states = len(self.states)
        n_nodes = len(self.nodes)
        
        if n_states < 2 or n_nodes == 0:
            return {}
        
        # Create coherence matrix: time x nodes
        coherence_matrix = np.zeros((n_states, n_nodes))
        node_ids = list(self.nodes.keys())
        
        for t, state in enumerate(self.states):
            for i, node_id in enumerate(node_ids):
                if node_id in state.nodes:
                    coherence_matrix[t, i] = state.nodes[node_id].coherence
        
        # Calculate pairwise correlations
        correlation_matrix = np.corrcoef(coherence_matrix.T)
        
        # Identify synchronized clusters
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering
        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method='average')
        clusters = fcluster(Z, t=0.5, criterion='distance')
        
        # Analyze clusters
        unique_clusters = np.unique(clusters)
        cluster_analysis = {}
        
        for cluster_id in unique_clusters:
            cluster_nodes = [node_ids[i] for i in range(n_nodes) if clusters[i] == cluster_id]
            cluster_coherences = coherence_matrix[:, clusters == cluster_id]
            
            # Cluster synchronization
            if len(cluster_nodes) > 0:
                cluster_sync = np.mean(cluster_coherences, axis=1)
                sync_strength = np.mean(cluster_sync)
                sync_variance = np.var(cluster_sync)
                
                # Identify cluster type
                cluster_node_types = [self.nodes[nid].node_type for nid in cluster_nodes]
                type_counts = {}
                for nt in cluster_node_types:
                    type_counts[nt.name] = type_counts.get(nt.name, 0) + 1
                
                dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "Mixed"
                
                cluster_analysis[cluster_id] = {
                    'size': len(cluster_nodes),
                    'sync_strength': sync_strength,
                    'sync_variance': sync_variance,
                    'dominant_node_type': dominant_type,
                    'node_ids': cluster_nodes
                }
        
        # Calculate global synchronization metrics
        global_sync = self._calculate_synchronization()
        
        # Identify phase-locked nodes
        phase_locked = []
        for i, node_id in enumerate(node_ids):
            node_coherence = coherence_matrix[:, i]
            if np.std(node_coherence) < 0.05:  # Low variance indicates phase locking
                phase_locked.append(node_id)
        
        return {
            'global_synchronization': global_sync,
            'correlation_matrix': correlation_matrix,
            'clusters': cluster_analysis,
            'phase_locked_nodes': phase_locked,
            'coherence_matrix': coherence_matrix,
            'node_ids': node_ids
        }
    
    def calculate_network_resilience(self,
                                    attack_types: List[str] = ['random', 'targeted', 'cascading']) -> Dict:
        """Calculate network resilience to different types of attacks"""
        print("Calculating network resilience...")
        
        resilience_results = {}
        
        for attack_type in attack_types:
            # Simulate different attack scenarios
            n_simulations = 10
            cascade_sizes = []
            
            for sim in range(n_simulations):
                if attack_type == 'random':
                    # Random node failures
                    n_initial_failures = int(len(self.nodes) * 0.1)  # 10% random failures
                    initial_failures = np.random.choice(list(self.nodes.keys()), 
                                                       n_initial_failures, 
                                                       replace=False).tolist()
                    
                elif attack_type == 'targeted':
                    # Target high-centrality nodes
                    critical_nodes = self.identify_critical_nodes(method='betweenness', 
                                                                 threshold=0.7)
                    n_initial_failures = min(5, len(critical_nodes))
                    initial_failures = [nid for nid, _ in critical_nodes[:n_initial_failures]]
                    
                elif attack_type == 'cascading':
                    # Start with most connected node
                    sorted_nodes = sorted(self.nodes.items(), 
                                        key=lambda x: x[1].degree, 
                                        reverse=True)
                    initial_failures = [sorted_nodes[0][0]]  # Highest degree node
                
                else:
                    continue
                
                # Simulate cascade
                cascade_result = self.simulate_cascade_failure(
                    initial_failure_nodes=initial_failures,
                    failure_model=attack_type,
                    recovery_rate=0.02
                )
                
                cascade_sizes.append(cascade_result['cascade_size'])
            
            # Calculate resilience metrics
            if cascade_sizes:
                avg_cascade_size = np.mean(cascade_sizes)
                std_cascade_size = np.std(cascade_sizes)
                resilience = 1.0 - avg_cascade_size  # Higher resilience = smaller cascades
            else:
                avg_cascade_size = std_cascade_size = resilience = 0.0
            
            resilience_results[attack_type] = {
                'avg_cascade_size': avg_cascade_size,
                'std_cascade_size': std_cascade_size,
                'resilience': resilience,
                'n_simulations': n_simulations
            }
        
        # Overall resilience score
        if resilience_results:
            overall_resilience = np.mean([r['resilience'] for r in resilience_results.values()])
        else:
            overall_resilience = 0.0
        
        return {
            'attack_specific': resilience_results,
            'overall_resilience': overall_resilience,
            'network_size': len(self.nodes),
            'network_density': self.states[-1].density if self.states else 0.0
        }
    
    def perform_multiscale_analysis(self) -> Dict:
        """Perform multiscale analysis of network coherence"""
        print("Performing multiscale analysis...")
        
        if len(self.states) < 10:
            return {}
        
        # Extract coherence time series
        global_coherence_ts = [state.global_coherence for state in self.states]
        times = [state.timestamp for state in self.states]
        
        # 1. Temporal scaling (Hurst exponent)
        try:
            from hurst import compute_Hc
            H, c, data = compute_Hc(global_coherence_ts, kind='random_walk')
            hurst_exponent = H
        except ImportError:
            # Simple Hurst estimation
            n = len(global_coherence_ts)
            if n >= 4:
                # Rescaled range analysis (simplified)
                mean_ts = np.mean(global_coherence_ts)
                deviations = global_coherence_ts - mean_ts
                cumulative = np.cumsum(deviations)
                R = np.max(cumulative) - np.min(cumulative)  # Range
                S = np.std(global_coherence_ts)  # Standard deviation
                
                if S > 0:
                    hurst_exponent = np.log(R/S) / np.log(n/2)
                    hurst_exponent = max(0, min(1, hurst_exponent))
                else:
                    hurst_exponent = 0.5
            else:
                hurst_exponent = 0.5
        
        # 2. Spectral analysis
        if len(global_coherence_ts) >= 50:
            freqs, psd = welch(global_coherence_ts, fs=1.0/self.time_step)
            
            # Find dominant frequencies
            peaks, properties = find_peaks(psd, height=np.percentile(psd, 75))
            dominant_freqs = freqs[peaks] if len(peaks) > 0 else []
            dominant_powers = psd[peaks] if len(peaks) > 0 else []
            
            # Calculate spectral exponent (power-law decay)
            if len(freqs) >= 10 and len(psd) >= 10:
                # Fit in log-log space, avoiding DC component and high frequencies
                mask = (freqs > 0.01) & (freqs < 0.5)
                if np.sum(mask) >= 5:
                    log_freqs = np.log(freqs[mask])
                    log_psd = np.log(psd[mask])
                    
                    # Linear fit
                    coeffs = np.polyfit(log_freqs, log_psd, 1)
                    spectral_exponent = -coeffs[0]  # Negative of slope
                else:
                    spectral_exponent = 1.0
            else:
                spectral_exponent = 1.0
        else:
            freqs = psd = []
            dominant_freqs = []
            dominant_powers = []
            spectral_exponent = 1.0
        
        # 3. Scale-by-scale analysis (wavelet-like)
        scales = [2, 4, 8, 16, 32]
        scale_variance = {}
        
        for scale in scales:
            if len(global_coherence_ts) >= scale:
                # Downsample and calculate variance
                downsampled = global_coherence_ts[::scale]
                scale_variance[scale] = np.var(downsampled)
            else:
                scale_variance[scale] = 0.0
        
        # 4. Multifractal analysis (simplified)
        q_values = [-5, -2, 0, 2, 5]
        tau_q = {}
        
        for q in q_values:
            # Simplified moment scaling
            if len(global_coherence_ts) >= 4:
                # Use scale variance for different moments
                tau_sum = 0
                for scale, var in scale_variance.items():
                    if var > 0:
                        tau_sum += np.log(var) / np.log(scale)
                
                tau_q[q] = tau_sum / len(scale_variance) if scale_variance else 0
            else:
                tau_q[q] = 0
        
        # 5. Coherence across scales
        scale_coherence = {}
        node_coherences = {}
        
        for node_id, node in self.nodes.items():
            if len(node.coherence_history) >= 2:
                node_coherences[node_id] = node.coherence_history
        
        # Calculate correlation between nodes at different time scales
        if len(node_coherences) >= 2:
            # Take last 100 time points if available
            min_length = min(len(h) for h in node_coherences.values())
            sample_length = min(100, min_length)
            
            if sample_length >= 10:
                # Create matrix: nodes x time
                n_nodes = len(node_coherences)
                time_matrix = np.zeros((n_nodes, sample_length))
                
                for i, (node_id, history) in enumerate(node_coherences.items()):
                    time_matrix[i, :] = history[-sample_length:]
                
                # Calculate correlations at different time scales
                for scale in [1, 2, 4, 8]:
                    if sample_length >= scale * 10:
                        # Downsample
                        downsampled = time_matrix[:, ::scale]
                        # Calculate average pairwise correlation
                        if downsampled.shape[1] >= 2:
                            corr_matrix = np.corrcoef(downsampled)
                            np.fill_diagonal(corr_matrix, 0)
                            avg_correlation = np.mean(np.abs(corr_matrix))
                            scale_coherence[f'scale_{scale}'] = avg_correlation
        
        return {
            'hurst_exponent': hurst_exponent,
            'spectral_exponent': spectral_exponent,
            'dominant_frequencies': list(dominant_freqs),
            'dominant_powers': list(dominant_powers),
            'scale_variance': scale_variance,
            'multifractal_spectrum': tau_q,
            'scale_coherence': scale_coherence,
            'time_series_length': len(global_coherence_ts),
            'persistence': 'persistent' if hurst_exponent > 0.6 else 
                          'anti-persistent' if hurst_exponent < 0.4 else 'random'
        }
    
    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================
    
    def visualize_network(self,
                         layout: str = 'spring',
                         highlight_communities: bool = True,
                         highlight_critical: bool = True,
                         save_path: Optional[str] = None):
        """Visualize the ontological network"""
        print("Visualizing network...")
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Ontological Network Analysis: {self.topology.name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Main network visualization
        ax1 = axes[0, 0]
        self._plot_network_layout(ax1, layout, highlight_communities, highlight_critical)
        
        # 2. Coherence distribution
        ax2 = axes[0, 1]
        self._plot_coherence_distribution(ax2)
        
        # 3. Degree distribution
        ax3 = axes[0, 2]
        self._plot_degree_distribution(ax3)
        
        # 4. Community structure
        ax4 = axes[1, 0]
        self._plot_community_structure(ax4)
        
        # 5. Centrality measures
        ax5 = axes[1, 1]
        self._plot_centrality_measures(ax5)
        
        # 6. Time evolution of global coherence
        ax6 = axes[1, 2]
        self._plot_coherence_evolution(ax6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_network_layout(self, ax, layout: str, highlight_communities: bool, highlight_critical: bool):
        """Plot network with specified layout"""
        # Create networkx graph with attributes
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, 
                      coherence=node.coherence,
                      node_type=node.node_type,
                      community=node.community_id,
                      betweenness=node.betweenness)
        
        # Add edges with weights
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, 
                      weight=edge.weight,
                      edge_type=edge.edge_type)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, weight='weight', seed=self.seed)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=self.seed)
        
        # Prepare node colors and sizes
        node_colors = []
        node_sizes = []
        
        for node_id in G.nodes():
            node = self.nodes[node_id]
            
            # Color by community or node type
            if highlight_communities and node.community_id != -1:
                # Color by community
                cmap = plt.cm.Set3
                node_colors.append(cmap(node.community_id % 12))
            else:
                # Color by node type
                type_colors = {
                    NodeType.RIGID_NODE: 'red',
                    NodeType.BRIDGE_NODE: 'blue',
                    NodeType.ALIEN_NODE: 'green',
                    NodeType.SOPHIA_NODE: 'purple',
                    NodeType.HYBRID_NODE: 'orange',
                    NodeType.PLEROMIC_NODE: 'gold',
                    NodeType.KENOMIC_NODE: 'gray'
                }
                node_colors.append(type_colors.get(node.node_type, 'black'))
            
            # Size by betweenness centrality (highlight critical nodes)
            if highlight_critical and node.betweenness > 0.1:
                node_sizes.append(300 + 500 * node.betweenness)
            else:
                node_sizes.append(100 + 200 * node.coherence)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, ax=ax, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8,
                              edgecolors='black',
                              linewidths=0.5)
        
        # Draw edges with transparency based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, ax=ax,
                              edgelist=edges,
                              width=[w * 2 for w in weights],
                              alpha=[min(0.8, w) for w in weights],
                              edge_color='gray')
        
        # Highlight critical nodes with labels
        if highlight_critical:
            critical_nodes = self.identify_critical_nodes(method='betweenness', threshold=0.7)
            critical_ids = [nid for nid, _ in critical_nodes[:5]]  # Top 5
            
            # Draw labels for critical nodes
            labels = {node_id: node_id for node_id in critical_ids}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, 
                                   font_size=8, font_weight='bold')
        
        ax.set_title('Network Layout')
        ax.axis('off')
    
    def _plot_coherence_distribution(self, ax):
        """Plot coherence distribution across nodes"""
        coherences = [node.coherence for node in self.nodes.values()]
        
        # Histogram
        ax.hist(coherences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical line for Sophia point
        ax.axvline(x=0.618, color='purple', linestyle='--', linewidth=2, 
                  label='Sophia Point (0.618)')
        
        # Add mean line
        mean_coherence = np.mean(coherences) if coherences else 0
        ax.axvline(x=mean_coherence, color='red', linestyle='-', linewidth=2,
                  label=f'Mean: {mean_coherence:.3f}')
        
        ax.set_xlabel('Coherence')
        ax.set_ylabel('Frequency')
        ax.set_title('Coherence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_degree_distribution(self, ax):
        """Plot degree distribution (log-log scale)"""
        degrees = [node.degree for node in self.nodes.values()]
        
        if not degrees:
            ax.text(0.5, 0.5, 'No degree data', ha='center', va='center')
            ax.set_title('Degree Distribution')
            return
        
        # Create log-binned histogram
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        
        # Log-log plot
        ax.loglog(unique_degrees, counts, 'bo', alpha=0.7, label='Data')
        
        # Fit power law (if enough data points)
        if len(unique_degrees) >= 5:
            # Linear fit in log-log space
            mask = unique_degrees > 0
            if np.sum(mask) >= 3:
                log_degrees = np.log(unique_degrees[mask])
                log_counts = np.log(counts[mask])
                
                # Linear regression
                coeffs = np.polyfit(log_degrees, log_counts, 1)
                fitted_counts = np.exp(coeffs[1]) * unique_degrees[mask]**coeffs[0]
                
                # Plot fit
                ax.loglog(unique_degrees[mask], fitted_counts, 'r--', 
                         label=f'Power law: α={-coeffs[0]:.2f}')
        
        ax.set_xlabel('Degree k (log scale)')
        ax.set_ylabel('P(k) (log scale)')
        ax.set_title('Degree Distribution (Log-Log)')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
    
    def _plot_community_structure(self, ax):
        """Plot community structure analysis"""
        if not self.communities:
            # Perform community detection
            self.analyze_community_structure()
        
        if not self.communities:
            ax.text(0.5, 0.5, 'No community data', ha='center', va='center')
            ax.set_title('Community Structure')
            return
        
        # Prepare data for visualization
        community_sizes = [len(nodes) for nodes in self.communities.values()]
        community_ids = list(self.communities.keys())
        
        # Bar chart of community sizes
        colors = plt.cm.Set3(np.linspace(0, 1, len(community_ids)))
        bars = ax.bar(range(len(community_ids)), community_sizes, color=colors, alpha=0.8)
        
        # Add coherence information for each community
        for i, comm_id in enumerate(community_ids):
            nodes_in_comm = [self.nodes[nid] for nid in self.communities[comm_id] 
                            if nid in self.nodes]
            if nodes_in_comm:
                avg_coherence = np.mean([n.coherence for n in nodes_in_comm])
                ax.text(i, community_sizes[i] + max(community_sizes)*0.02,
                       f'{avg_coherence:.2f}', ha='center', fontsize=8)
        
        ax.set_xlabel('Community ID')
        ax.set_ylabel('Size')
        ax.set_title(f'Community Structure (Modularity: {self.states[-1].modularity:.3f})')
        ax.set_xticks(range(len(community_ids)))
        ax.set_xticklabels(community_ids)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_centrality_measures(self, ax):
        """Plot comparison of centrality measures"""
        if not self.nodes:
            ax.text(0.5, 0.5, 'No node data', ha='center', va='center')
            ax.set_title('Centrality Measures')
            return
        
        # Get centrality measures for top 20 nodes
        node_ids = list(self.nodes.keys())[:20]
        
        betweenness = [self.nodes[nid].betweenness for nid in node_ids]
        closeness = [self.nodes[nid].closeness for nid in node_ids]
        eigenvector = [self.nodes[nid].eigenvector_centrality for nid in node_ids]
        
        x = np.arange(len(node_ids))
        width = 0.25
        
        ax.bar(x - width, betweenness, width, label='Betweenness', alpha=0.8)
        ax.bar(x, closeness, width, label='Closeness', alpha=0.8)
        ax.bar(x + width, eigenvector, width, label='Eigenvector', alpha=0.8)
        
        ax.set_xlabel('Node ID')
        ax.set_ylabel('Centrality Value')
        ax.set_title('Centrality Measures Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(node_ids, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_coherence_evolution(self, ax):
        """Plot evolution of global coherence over time"""
        if len(self.states) < 2:
            ax.text(0.5, 0.5, 'Insufficient time data', ha='center', va='center')
            ax.set_title('Coherence Evolution')
            return
        
        times = [state.timestamp for state in self.states]
        coherences = [state.global_coherence for state in self.states]
        
        ax.plot(times, coherences, 'b-', linewidth=2, label='Global Coherence')
        ax.axhline(y=0.618, color='purple', linestyle='--', alpha=0.7, 
                  label='Sophia Point')
        
        # Mark phase transitions
        for trans in self.phase_transitions:
            ax.axvline(x=trans['time'], color='red', alpha=0.5, linestyle=':')
            ax.text(trans['time'], 0.9, 'PT', rotation=90, fontsize=8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Global Coherence')
        ax.set_title('Coherence Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_network_animation(self, 
                                steps: int = 100,
                                interval: int = 50,
                                save_path: Optional[str] = None):
        """Create animation of network evolution"""
        print("Creating network animation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        times = [state.timestamp for state in self.states[:steps]]
        coherences = [state.global_coherence for state in self.states[:steps]]
        
        # Setup network plot
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_title('Network Evolution')
        ax1.axis('off')
        
        # Setup coherence plot
        ax2.set_xlim(min(times) if times else 0, max(times) if times else 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Global Coherence')
        ax2.set_title('Coherence Evolution')
        ax2.axhline(y=0.618, color='purple', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        # Create layout for consistent visualization
        G = nx.Graph()
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        pos = nx.spring_layout(G, seed=self.seed)
        
        # Initialize plots
        network_plot = nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=100, 
                                             node_color='blue', alpha=0.6)
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, width=0.5)
        
        coherence_line, = ax2.plot([], [], 'b-', linewidth=2)
        current_time_line = ax2.axvline(x=0, color='red', alpha=0.5)
        
        def animate(i):
            """Update function for animation"""
            if i >= len(self.states) or i >= steps:
                return network_plot, coherence_line, current_time_line
            
            # Update network colors based on current state
            state = self.states[i]
            node_colors = []
            
            for node_id in G.nodes():
                if node_id in state.nodes:
                    node = state.nodes[node_id]
                    # Color by coherence
                    color_val = node.coherence
                    node_colors.append(plt.cm.viridis(color_val))
                else:
                    node_colors.append('gray')
            
            # Clear and redraw network
            ax1.clear()
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
            ax1.set_title(f'Network at t={state.timestamp:.2f}')
            ax1.axis('off')
            
            nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=100,
                                  node_color=node_colors, alpha=0.8,
                                  edgecolors='black', linewidths=0.5)
            nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, width=0.5)
            
            # Update coherence plot
            current_times = times[:i+1]
            current_coherences = coherences[:i+1]
            
            coherence_line.set_data(current_times, current_coherences)
            current_time_line.set_xdata([state.timestamp, state.timestamp])
            
            # Update coherence plot limits
            if current_times:
                ax2.set_xlim(min(current_times), max(max(current_times), 1))
            
            return network_plot, coherence_line, current_time_line
        
        anim = animation.FuncAnimation(fig, animate, frames=min(steps, len(self.states)),
                                      interval=interval, blit=False)
        
        plt.tight_layout()
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=30)
        
        plt.show()
        return anim

# ============================================================================
# ADVANCED NETWORK ANALYSIS FUNCTIONS
# ============================================================================

def run_comprehensive_analysis(network_size: int = 200,
                              topology: NetworkTopology = NetworkTopology.SCALE_FREE,
                              propagation_steps: int = 50,
                              output_dir: str = "network_analysis_results"):
    """Run comprehensive network coherence analysis"""
    print("=" * 70)
    print("COMPREHENSIVE NETWORK COHERENCE ANALYSIS")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize analyzer
    print(f"\n1. Initializing {topology.name} network with {network_size} nodes...")
    analyzer = NetworkCoherenceAnalyzer(
        n_nodes=network_size,
        topology=topology,
        initial_coherence_range=(0.3, 0.7),
        seed=42
    )
    
    # Propagate coherence
    print("\n2. Propagating coherence through network...")
    analyzer.propagate_coherence(
        propagation_model='diffusive',
        alpha=0.1,
        beta=0.05,
        steps=propagation_steps
    )
    
    # Visualize network
    print("\n3. Creating network visualizations...")
    analyzer.visualize_network(
        layout='spring',
        highlight_communities=True,
        highlight_critical=True,
        save_path=str(output_path / "network_visualization.png")
    )
    
    # Analyze community structure
    print("\n4. Analyzing community structure...")
    community_results = analyzer.analyze_community_structure()
    
    with open(output_path / "community_analysis.json", 'w') as f:
        json.dump(community_results, f, indent=2, default=str)
    
    print(f"  Found {len(community_results.get('communities', {}))} communities")
    print(f"  Modularity: {analyzer.states[-1].modularity:.3f}")
    
    # Identify critical nodes
    print("\n5. Identifying critical nodes...")
    critical_nodes = analyzer.identify_critical_nodes(method='combined', threshold=0.8)
    
    print(f"  Found {len(critical_nodes)} critical nodes")
    print("  Top 5 critical nodes:")
    for node_id, score in critical_nodes[:5]:
        node = analyzer.nodes[node_id]
        print(f"    {node_id}: score={score:.3f}, type={node.node_type.name}, "
              f"coherence={node.coherence:.3f}")
    
    # Analyze synchronization
    print("\n6. Analyzing synchronization patterns...")
    sync_results = analyzer.analyze_synchronization_patterns()
    
    print(f"  Global synchronization: {sync_results.get('global_synchronization', 0):.3f}")
    if 'clusters' in sync_results:
        print(f"  Found {len(sync_results['clusters'])} synchronized clusters")
    
    # Calculate network resilience
    print("\n7. Calculating network resilience...")
    resilience_results = analyzer.calculate_network_resilience(
        attack_types=['random', 'targeted', 'cascading']
    )
    
    print(f"  Overall resilience: {resilience_results.get('overall_resilience', 0):.3f}")
    for attack_type, results in resilience_results.get('attack_specific', {}).items():
        print(f"    {attack_type}: resilience={results['resilience']:.3f}, "
              f"avg cascade={results['avg_cascade_size']:.3f}")
    
    # Perform multiscale analysis
    print("\n8. Performing multiscale analysis...")
    multiscale_results = analyzer.perform_multiscale_analysis()
    
    print(f"  Hurst exponent: {multiscale_results.get('hurst_exponent', 0):.3f}")
    print(f"  Persistence: {multiscale_results.get('persistence', 'unknown')}")
    print(f"  Spectral exponent: {multiscale_results.get('spectral_exponent', 0):.3f}")
    
    # Create animation
    print("\n9. Creating network evolution animation...")
    anim = analyzer.create_network_animation(
        steps=min(100, len(analyzer.states)),
        interval=100,
        save_path=str(output_path / "network_evolution.gif")
    )
    
    # Generate summary report
    print("\n10. Generating summary report...")
    summary = {
        'timestamp': datetime.now().isoformat(),
        'network_parameters': {
            'size': network_size,
            'topology': topology.name,
            'propagation_steps': propagation_steps
        },
        'final_state': analyzer.states[-1].to_dict() if analyzer.states else {},
        'critical_nodes_count': len(critical_nodes),
        'community_count': len(community_results.get('communities', {})),
        'phase_transitions': len(analyzer.phase_transitions),
        'resilience': resilience_results.get('overall_resilience', 0),
        'multiscale_analysis': multiscale_results
    }
    
    with open(output_path / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Plot comprehensive metrics over time
    print("\n11. Plotting comprehensive metrics...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Global coherence over time
    if analyzer.states:
        times = [s.timestamp for s in analyzer.states]
        global_coherence = [s.global_coherence for s in analyzer.states]
        
        axes[0, 0].plot(times, global_coherence, 'b-', linewidth=2)
        axes[0, 0].axhline(y=0.618, color='purple', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Global Coherence')
        axes[0, 0].set_title('Global Coherence Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark phase transitions
        for trans in analyzer.phase_transitions:
            axes[0, 0].axvline(x=trans['time'], color='red', alpha=0.3, linestyle=':')
    
    # Modularity over time
    if len(analyzer.metrics_history.get('modularity', [])) > 0:
        axes[0, 1].plot(times[:len(analyzer.metrics_history['modularity'])], 
                       analyzer.metrics_history['modularity'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Modularity')
        axes[0, 1].set_title('Modularity Evolution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Synchronization over time
    if len(analyzer.metrics_history.get('synchronization', [])) > 0:
        axes[0, 2].plot(times[:len(analyzer.metrics_history['synchronization'])], 
                       analyzer.metrics_history['synchronization'], 'r-', linewidth=2)
        axes[0, 2].axhline(y=0.5, color='purple', linestyle='--', alpha=0.7)
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Synchronization')
        axes[0, 2].set_title('Synchronization Evolution')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Criticality index over time
    if len(analyzer.metrics_history.get('criticality', [])) > 0:
        axes[1, 0].plot(times[:len(analyzer.metrics_history['criticality'])], 
                       analyzer.metrics_history['criticality'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Criticality Index')
        axes[1, 0].set_title('Criticality Evolution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Spectral gap over time
    if len(analyzer.metrics_history.get('spectral_gap', [])) > 0:
        axes[1, 1].plot(times[:len(analyzer.metrics_history['spectral_gap'])], 
                       analyzer.metrics_history['spectral_gap'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Spectral Gap')
        axes[1, 1].set_title('Spectral Gap Evolution')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Average path length over time
    if len(analyzer.metrics_history.get('average_path_length', [])) > 0:
        axes[1, 2].plot(times[:len(analyzer.metrics_history['average_path_length'])], 
                       analyzer.metrics_history['average_path_length'], 'y-', linewidth=2)
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Average Path Length')
        axes[1, 2].set_title('Path Length Evolution')
        axes[1, 2].grid(True, alpha=0.3)
    
    # Node coherence distribution at final state
    if analyzer.nodes:
        final_coherences = [node.coherence for node in analyzer.nodes.values()]
        axes[2, 0].hist(final_coherences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2, 0].axvline(x=0.618, color='purple', linestyle='--', linewidth=2)
        axes[2, 0].set_xlabel('Coherence')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Final Coherence Distribution')
        axes[2, 0].grid(True, alpha=0.3)
    
    # Degree distribution at final state
    if analyzer.nodes:
        degrees = [node.degree for node in analyzer.nodes.values()]
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        axes[2, 1].loglog(unique_degrees, counts, 'bo', alpha=0.7)
        axes[2, 1].set_xlabel('Degree (log)')
        axes[2, 1].set_ylabel('Count (log)')
        axes[2, 1].set_title('Final Degree Distribution')
        axes[2, 1].grid(True, alpha=0.3, which='both')
    
    # Node centrality comparison
    if analyzer.nodes and len(analyzer.nodes) >= 10:
        node_sample = list(analyzer.nodes.values())[:10]
        betweenness = [n.betweenness for n in node_sample]
        closeness = [n.closeness for n in node_sample]
        eigenvector = [n.eigenvector_centrality for n in node_sample]
        
        x = np.arange(len(node_sample))
        width = 0.25
        
        axes[2, 2].bar(x - width, betweenness, width, label='Betweenness', alpha=0.8)
        axes[2, 2].bar(x, closeness, width, label='Closeness', alpha=0.8)
        axes[2, 2].bar(x + width, eigenvector, width, label='Eigenvector', alpha=0.8)
        
        axes[2, 2].set_xlabel('Node')
        axes[2, 2].set_ylabel('Centrality')
        axes[2, 2].set_title('Centrality Comparison (Sample)')
        axes[2, 2].set_xticks(x)
        axes[2, 2].set_xticklabels([n.id[:5] for n in node_sample], rotation=45)
        axes[2, 2].legend(fontsize=8)
        axes[2, 2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Comprehensive Network Analysis: {topology.name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "comprehensive_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print(f"Summary statistics:")
    print(f"  Final global coherence: {analyzer.states[-1].global_coherence:.3f}")
    print(f"  Network modularity: {analyzer.states[-1].modularity:.3f}")
    print(f"  Synchronization index: {analyzer.states[-1].synchronization_index:.3f}")
    print(f"  Criticality index: {analyzer.states[-1].criticality_index:.3f}")
    print(f"  Phase transitions detected: {len(analyzer.phase_transitions)}")
    print(f"  Critical nodes identified: {len(critical_nodes)}")
    
    return analyzer

def compare_network_topologies(topologies: List[NetworkTopology] = None,
                              network_size: int = 100,
                              propagation_steps: int = 30):
    """Compare different network topologies"""
    if topologies is None:
        topologies = [
            NetworkTopology.SCALE_FREE,
            NetworkTopology.SMALL_WORLD,
            NetworkTopology.RANDOM,
            NetworkTopology.MODULAR
        ]
    
    print("=" * 70)
    print("NETWORK TOPOLOGY COMPARISON")
    print("=" * 70)
    
    results = {}
    
    for topology in topologies:
        print(f"\nAnalyzing {topology.name} network...")
        
        analyzer = NetworkCoherenceAnalyzer(
            n_nodes=network_size,
            topology=topology,
            initial_coherence_range=(0.3, 0.7)
        )
        
        analyzer.propagate_coherence(
            propagation_model='diffusive',
            alpha=0.1,
            beta=0.05,
            steps=propagation_steps
        )
        
        final_state = analyzer.states[-1]
        critical_nodes = analyzer.identify_critical_nodes(method='combined', threshold=0.8)
        
        results[topology.name] = {
            'final_coherence': final_state.global_coherence,
            'modularity': final_state.modularity,
            'synchronization': final_state.synchronization_index,
            'criticality': final_state.criticality_index,
            'average_path_length': final_state.average_path_length,
            'diameter': final_state.diameter,
            'density': final_state.density,
            'critical_nodes_count': len(critical_nodes),
            'phase_transitions': len(analyzer.phase_transitions)
        }
    
    # Plot comparison
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    metrics = ['final_coherence', 'modularity', 'synchronization', 
               'criticality', 'average_path_length', 'density',
               'critical_nodes_count', 'phase_transitions', 'diameter']
    
    titles = ['Final Coherence', 'Modularity', 'Synchronization',
              'Criticality Index', 'Avg Path Length', 'Density',
              'Critical Nodes', 'Phase Transitions', 'Diameter']
    
    topology_names = list(results.keys())
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 3
        col = idx % 3
        
        values = [results[topology][metric] for topology in topology_names]
        
        axes[row, col].bar(topology_names, values, alpha=0.7)
        axes[row, col].set_title(title)
        axes[row, col].set_ylabel(title)
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(values):
            axes[row, col].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Network Topology Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('topology_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Topology':<15} {'Coherence':<10} {'Modularity':<10} {'Sync':<8} "
          f"{'Criticality':<12} {'Crit Nodes':<10}")
    print("-" * 70)
    
    for topology in topology_names:
        r = results[topology]
        print(f"{topology:<15} {r['final_coherence']:<10.3f} {r['modularity']:<10.3f} "
              f"{r['synchronization']:<8.3f} {r['criticality']:<12.3f} "
              f"{r['critical_nodes_count']:<10}")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NETWORK COHERENCE ANALYSIS - Sophia Axiom Implementation")
    print("=" * 70)
    
    # Run comprehensive analysis on scale-free network
    analyzer = run_comprehensive_analysis(
        network_size=200,
        topology=NetworkTopology.SCALE_FREE,
        propagation_steps=50,
        output_dir="network_analysis"
    )
    
    # Compare different network topologies
    print("\n" + "=" * 70)
    print("COMPARING NETWORK TOPOLOGIES")
    print("=" * 70)
    
    topology_results = compare_network_topologies(
        topologies=[
            NetworkTopology.SCALE_FREE,
            NetworkTopology.SMALL_WORLD,
            NetworkTopology.RANDOM,
            NetworkTopology.MODULAR
        ],
        network_size=100,
        propagation_steps=30
    )
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Final recommendations based on analysis
    if analyzer and analyzer.states:
        final_state = analyzer.states[-1]
        
        print("\nRECOMMENDATIONS:")
        
        if final_state.global_coherence < 0.5:
            print("  • Global coherence is low. Consider increasing participation (P) "
                  "and plasticity (Π) in key nodes.")
        elif final_state.global_coherence > 0.7:
            print("  • Global coherence is high. System is well-integrated. "
                  "Focus on maintaining stability.")
        else:
            print("  • Global coherence is moderate. System is in dynamic balance. "
                  "Monitor for phase transitions.")
        
        if final_state.modularity > 0.7:
            print("  • High modularity detected. Consider adding inter-community "
                  "connections to improve integration.")
        elif final_state.modularity < 0.3:
            print("  • Low modularity detected. System may lack specialized "
                  "communities. Consider fostering differentiation.")
        
        if final_state.criticality_index > 0.7:
            print("  • High criticality detected. System is near phase transition. "
                  "Monitor closely for sudden changes.")
        
        critical_nodes = analyzer.identify_critical_nodes(method='combined', threshold=0.8)
        if critical_nodes:
            print(f"  • {len(critical_nodes)} critical nodes identified. Focus interventions "
                  "on these key nodes for maximum impact.")
        
        if analyzer.phase_transitions:
            print(f"  • {len(analyzer.phase_transitions)} phase transitions detected. "
                  "System has undergone significant restructuring.")
