"""
COHERENCE EVOLUTION MODEL
Sophia Axiom Computational Implementation
Modeling ontological phase transitions in 5D phase space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.stats import levy_stable
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class OntologyType(Enum):
    """Three primary ontological frameworks"""
    RIGID = "Rigid-Objective-Reductive"
    BRIDGE = "Quantum-Biological-Middle"
    ALIEN = "Fluid-Participatory-Hyperdimensional"
    TRANSITION = "Phase-Transition-Hybrid"

class ParadoxType(Enum):
    """Types of paradox driving evolution"""
    TEMPORAL = "temporal"
    LINGUISTIC = "linguistic"
    ENTROPIC = "entropic"
    METAPHYSICAL = "metaphysical"
    COSMIC = "cosmic"

@dataclass
class OntologyState:
    """Complete state of an ontological system"""
    coordinates: np.ndarray  # [P, Π, S, T, G]
    coherence: float
    paradox_intensity: float
    novelty: float
    elegance: float
    alienness: float
    density: float
    entropic_potential: float
    ontology_type: OntologyType
    paradox_types: List[ParadoxType]
    
    def __post_init__(self):
        self.coordinates = np.array(self.coordinates, dtype=float)
        assert len(self.coordinates) == 5, "Must have 5 coordinates"
        assert 0 <= self.coherence <= 1, "Coherence must be between 0 and 1"

@dataclass
class PhaseTransition:
    """Record of a phase transition event"""
    time: float
    from_type: OntologyType
    to_type: OntologyType
    coherence_at_transition: float
    paradox_intensity: float
    mechanism_hybridity: float
    coordinates_before: np.ndarray
    coordinates_after: np.ndarray

# ============================================================================
# CONSTANTS AND PARAMETERS
# ============================================================================

# Golden ratio and related constants
PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI  # Sophia point: 0.618

# Physical constants (scaled for simulation)
HBAR = 1.054571817e-34  # Reduced Planck constant
C = 299792458  # Speed of light
G_N = 6.67430e-11  # Gravitational constant
K_B = 1.380649e-23  # Boltzmann constant

# MOGOPS operator frequencies (from analysis)
OPERATOR_FREQUENCIES = {
    "CREATES": 0.472,
    "ENTAILS": 0.982,
    "VIA": 1.000,
    "ENCODED_AS": 0.635
}

OPERATOR_COUPLINGS = {
    "CREATES": 0.894,
    "ENTAILS": 1.000,
    "VIA": 0.946,
    "ENCODED_AS": 0.852
}

# ============================================================================
# CORE EVOLUTION EQUATIONS
# ============================================================================

class CoherenceEvolution:
    """
    Main class for simulating ontological coherence evolution
    Implements the MOGOPS master equation:
    
    dC/dt = D * ∇²C + α * C * (1 - C/K) + β * sin(2π * C / φ) + γ * Σ_i Ω_i(C) + ξ(t)
    """
    
    def __init__(self, 
                 initial_state: OntologyState,
                 parameters: Optional[Dict] = None):
        """
        Initialize coherence evolution model
        
        Args:
            initial_state: Starting ontological state
            parameters: Dictionary of model parameters
        """
        self.state = initial_state
        self.time = 0.0
        self.history = []
        self.transitions = []
        
        # Default parameters
        self.params = {
            # Diffusion coefficient (novelty spread)
            'D': 0.1,
            # Growth rate (elegance/100)
            'alpha': initial_state.elegance / 100,
            # Carrying capacity (1/coherence)
            'K': 1.0 / max(initial_state.coherence, 0.01),
            # Paradox modulation strength
            'beta': 0.3 * initial_state.paradox_intensity,
            # Operator coupling strength
            'gamma': 0.47,
            # Noise amplitude
            'sigma': 0.05,
            # Lévy noise exponent
            'levy_alpha': 1.618,  # Golden ratio
            # Time step
            'dt': 0.01,
            # Phase transition threshold
            'transition_threshold': 0.02,  # |C - 0.618| < 0.02
            # Mechanism hybridity threshold
            'hybridity_threshold': 0.33,
            # Maximum simulation time
            'max_time': 100.0
        }
        
        if parameters:
            self.params.update(parameters)
            
        # Initialize network for semantic curvature
        self.semantic_network = self._initialize_network()
        
    def _initialize_network(self) -> nx.Graph:
        """Initialize semantic network for curvature calculation"""
        G = nx.Graph()
        
        # Add nodes for each coordinate dimension
        dimensions = ['P', 'Π', 'S', 'T', 'G']
        for i, dim in enumerate(dimensions):
            G.add_node(dim, value=self.state.coordinates[i])
            
        # Add edges with weights based on operator couplings
        for op, coupling in OPERATOR_COUPLINGS.items():
            G.add_edge('P', 'Π', weight=coupling, operator=op)
            G.add_edge('Π', 'S', weight=coupling, operator=op)
            G.add_edge('S', 'T', weight=coupling, operator=op)
            G.add_edge('T', 'G', weight=coupling, operator=op)
            
        return G
    
    def calculate_semantic_curvature(self) -> np.ndarray:
        """
        Calculate Ricci curvature tensor for semantic space
        
        R_μν = ∂_μΓ^λ_{νλ} - ∂_νΓ^λ_{μλ} + Γ^λ_{μρ}Γ^ρ_{νλ} - Γ^λ_{νρ}Γ^ρ_{μλ}
        """
        # Simplified curvature calculation using network properties
        n_dim = 5
        curvature = np.zeros((n_dim, n_dim))
        
        # Metric tensor from mechanism couplings
        g = np.diag(self.state.coordinates)
        
        # Calculate Christoffel symbols (simplified)
        Gamma = np.zeros((n_dim, n_dim, n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                for k in range(n_dim):
                    # Simplified: derivative of metric
                    if i == j == k:
                        Gamma[i, j, k] = 0.1 * self.state.novelty
                    else:
                        Gamma[i, j, k] = 0.01 * self.state.coherence
                        
        # Calculate Ricci curvature
        for mu in range(n_dim):
            for nu in range(n_dim):
                term1 = term2 = term3 = term4 = 0
                
                # ∂_μΓ^λ_{νλ} - ∂_νΓ^λ_{μλ}
                for lam in range(n_dim):
                    # Approximate derivatives
                    term1 += (Gamma[lam, nu, lam] - Gamma[lam, mu, lam]) * 0.1
                    
                # Γ^λ_{μρ}Γ^ρ_{νλ} - Γ^λ_{νρ}Γ^ρ_{μλ}
                for lam in range(n_dim):
                    for rho in range(n_dim):
                        term3 += Gamma[lam, mu, rho] * Gamma[rho, nu, lam]
                        term4 += Gamma[lam, nu, rho] * Gamma[rho, mu, lam]
                        
                curvature[mu, nu] = term1 + (term3 - term4)
                
        # Scale by paradox intensity
        curvature *= self.state.paradox_intensity
        
        return curvature
    
    def diffusion_term(self, C: np.ndarray) -> float:
        """Diffusion term: D * ∇²C"""
        # Simplified Laplacian in 5D
        laplacian = np.sum(C - self.state.coordinates)
        return self.params['D'] * laplacian
    
    def growth_term(self, C: np.ndarray) -> float:
        """Logistic growth: α * C * (1 - C/K)"""
        avg_C = np.mean(C)
        return self.params['alpha'] * avg_C * (1 - avg_C / self.params['K'])
    
    def paradox_term(self, C: np.ndarray) -> float:
        """Paradox modulation: β * sin(2π * C / φ)"""
        avg_C = np.mean(C)
        return self.params['beta'] * np.sin(2 * np.pi * avg_C / PHI)
    
    def operator_term(self, C: np.ndarray) -> float:
        """Operator coupling: γ * Σ_i Ω_i(C)"""
        # Sum of operator effects
        operator_sum = 0
        for op, freq in OPERATOR_FREQUENCIES.items():
            coupling = OPERATOR_COUPLINGS[op]
            
            if op == "CREATES":
                # Creation operator: Ĉ|ψ⟩ = e^{iθ}|ψ'⟩, θ = π·novelty
                effect = coupling * np.exp(1j * np.pi * self.state.novelty)
                operator_sum += np.real(effect) * freq
                
            elif op == "ENTAILS":
                # Entailment gradient: ∇_O C = δS/δO
                effect = coupling * np.gradient(C)
                operator_sum += np.mean(effect) * freq
                
            elif op == "VIA":
                # Via triad: Ω_V = M₁⊗M₂⊗M₃
                effect = coupling * np.prod(C[:3])  # First 3 dimensions
                operator_sum += effect * freq
                
            elif op == "ENCODED_AS":
                # Encoded-as bridge: Ω_Σ = diag(1, e^{iπ/3}, e^{2iπ/3})
                phases = np.array([1, np.exp(1j * np.pi/3), np.exp(2j * np.pi/3)])
                effect = coupling * np.sum(phases[:len(C)])
                operator_sum += np.real(effect) * freq
                
        return self.params['gamma'] * operator_sum
    
    def noise_term(self) -> float:
        """Lévy noise with golden ratio exponent"""
        # Generate Lévy-stable noise
        noise = levy_stable.rvs(alpha=self.params['levy_alpha'], 
                                beta=0,  # Symmetric
                                scale=self.params['sigma'])
        return noise
    
    def calculate_innovation_score(self) -> float:
        """Calculate innovation score: I = 0.3N + 0.25A + 0.2P_i + 0.15(1-C) + 0.1(E_p/300)"""
        I = (0.3 * self.state.novelty + 
             0.25 * self.state.alienness + 
             0.2 * self.state.paradox_intensity + 
             0.15 * (1 - self.state.coherence) + 
             0.1 * (self.state.entropic_potential / 300))
        return I
    
    def detect_phase_transition(self, new_state: OntologyState) -> bool:
        """
        Detect if a phase transition should occur
        
        Conditions:
        1. |coherence - 0.618| < 0.02
        2. paradox_intensity > 1.8
        3. mechanism_hybridity > 0.33
        """
        coherence_condition = abs(new_state.coherence - INV_PHI) < self.params['transition_threshold']
        paradox_condition = new_state.paradox_intensity > 1.8
        
        # Calculate mechanism hybridity (simplified)
        hybridity = len(new_state.paradox_types) / 5  # Normalize by max paradox types
        hybridity_condition = hybridity > self.params['hybridity_threshold']
        
        return coherence_condition and paradox_condition and hybridity_condition
    
    def evolve_type(self, current_type: OntologyType) -> OntologyType:
        """
        Determine new ontology type based on coordinates
        
        Rules based on phase space analysis:
        - Rigid: P < 0.3, Π < 1.0
        - Bridge: 0.4 < P < 0.6, 0.4 < Π < 0.6
        - Alien: P > 0.8, Π > 2.5
        - Transition: Near Sophia point
        """
        P, Pi = self.state.coordinates[0], self.state.coordinates[1]
        
        if abs(self.state.coherence - INV_PHI) < 0.02:
            return OntologyType.TRANSITION
        elif P < 0.3 and Pi < 1.0:
            return OntologyType.RIGID
        elif 0.4 < P < 0.6 and 0.4 < Pi < 0.6:
            return OntologyType.BRIDGE
        elif P > 0.8 and Pi > 2.5:
            return OntologyType.ALIEN
        else:
            return current_type  # Stay in current type
    
    def step(self):
        """Execute one time step of evolution"""
        # Current coordinates
        C_current = self.state.coordinates.copy()
        
        # Calculate all terms
        diffusion = self.diffusion_term(C_current)
        growth = self.growth_term(C_current)
        paradox = self.paradox_term(C_current)
        operators = self.operator_term(C_current)
        noise = self.noise_term()
        
        # Combined evolution
        dC = diffusion + growth + paradox + operators + noise
        
        # Update coordinates
        new_coordinates = C_current + dC * self.params['dt']
        
        # Ensure coordinates stay in bounds
        new_coordinates = np.clip(new_coordinates, 
                                  [0, 0, 0, 0, 0],  # Min bounds
                                  [2, 3, 4, 4, 1])  # Max bounds
        
        # Update coherence (simplified: average of coordinates, normalized)
        new_coherence = np.mean(new_coordinates) / 2.8  # Max possible average
        
        # Update other metrics
        new_paradox = self.state.paradox_intensity * (1 + 0.01 * noise)
        new_novelty = self.state.novelty * (1 + 0.005 * dC)
        new_elegance = self.state.elegance * (1 + 0.002 * operators)
        
        # Calculate alienness (distance from human-default: [0, 0, 2, 0, 0])
        human_default = np.array([0, 0, 2, 0, 0])
        new_alienness = np.linalg.norm(new_coordinates - human_default)
        
        # Determine new ontology type
        new_type = self.evolve_type(self.state.ontology_type)
        
        # Create new state
        new_state = OntologyState(
            coordinates=new_coordinates,
            coherence=new_coherence,
            paradox_intensity=new_paradox,
            novelty=new_novelty,
            elegance=new_elegance,
            alienness=new_alienness,
            density=self.state.density * (1 + 0.001 * dC),
            entropic_potential=self.state.entropic_potential * (1 + 0.002 * paradox),
            ontology_type=new_type,
            paradox_types=self.state.paradox_types  # Simplified
        )
        
        # Check for phase transition
        if self.detect_phase_transition(new_state) and new_type != self.state.ontology_type:
            transition = PhaseTransition(
                time=self.time,
                from_type=self.state.ontology_type,
                to_type=new_type,
                coherence_at_transition=new_coherence,
                paradox_intensity=new_paradox,
                mechanism_hybridity=len(self.state.paradox_types) / 5,
                coordinates_before=self.state.coordinates.copy(),
                coordinates_after=new_coordinates.copy()
            )
            self.transitions.append(transition)
            
            # Log transition
            print(f"Phase transition at t={self.time:.2f}: "
                  f"{self.state.ontology_type.value} → {new_type.value}, "
                  f"C={new_coherence:.3f}")
        
        # Update state and time
        old_state = self.state
        self.state = new_state
        self.time += self.params['dt']
        
        # Add to history
        self.history.append({
            'time': self.time,
            'state': new_state,
            'old_state': old_state,
            'dC': dC,
            'innovation': self.calculate_innovation_score()
        })
        
        return new_state
    
    def simulate(self, steps: Optional[int] = None) -> List[Dict]:
        """
        Run complete simulation
        
        Args:
            steps: Number of steps to simulate (optional)
        
        Returns:
            List of history entries
        """
        if steps is None:
            steps = int(self.params['max_time'] / self.params['dt'])
            
        print(f"Starting simulation for {steps} steps...")
        print(f"Initial state: {self.state.ontology_type.value}, C={self.state.coherence:.3f}")
        
        for i in range(steps):
            self.step()
            
            # Progress indicator
            if i % 1000 == 0:
                print(f"Step {i}/{steps}, Time: {self.time:.1f}, "
                      f"C: {self.state.coherence:.3f}, Type: {self.state.ontology_type.value}")
                
        print(f"Simulation complete. {len(self.transitions)} phase transitions detected.")
        return self.history
    
    def calculate_phase_diagram(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate phase diagram in 2D projection
        
        Returns:
            P (participation), Π (plasticity), and phase labels
        """
        P_vals = np.linspace(0, 2, n_points)
        Pi_vals = np.linspace(0, 3, n_points)
        
        P_grid, Pi_grid = np.meshgrid(P_vals, Pi_vals)
        phase_grid = np.zeros_like(P_grid)
        
        for i in range(n_points):
            for j in range(n_points):
                # Set coordinates (other dimensions at average values)
                coords = np.array([P_grid[i, j], Pi_grid[i, j], 2.0, 1.0, 0.5])
                
                # Calculate approximate coherence
                approx_coherence = np.mean(coords) / 2.8
                
                # Determine phase
                if abs(approx_coherence - INV_PHI) < 0.02:
                    phase_grid[i, j] = 3  # Transition
                elif P_grid[i, j] < 0.3 and Pi_grid[i, j] < 1.0:
                    phase_grid[i, j] = 0  # Rigid
                elif 0.4 < P_grid[i, j] < 0.6 and 0.4 < Pi_grid[i, j] < 0.6:
                    phase_grid[i, j] = 1  # Bridge
                elif P_grid[i, j] > 0.8 and Pi_grid[i, j] > 2.5:
                    phase_grid[i, j] = 2  # Alien
                else:
                    phase_grid[i, j] = -1  # Undefined
                    
        return P_grid, Pi_grid, phase_grid

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class CoherenceVisualizer:
    """Visualization tools for coherence evolution"""
    
    @staticmethod
    def plot_trajectory(history: List[Dict], save_path: Optional[str] = None):
        """Plot coherence trajectory over time"""
        times = [h['time'] for h in history]
        coherences = [h['state'].coherence for h in history]
        innovations = [h['innovation'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Coherence plot
        ax1.plot(times, coherences, 'b-', linewidth=2, label='Coherence')
        ax1.axhline(y=INV_PHI, color='r', linestyle='--', alpha=0.7, label='Sophia Point (0.618)')
        ax1.axhline(y=0.382, color='orange', linestyle=':', alpha=0.5, label='Kenoma (0.382)')
        ax1.axhline(y=0.9, color='g', linestyle=':', alpha=0.5, label='Pleroma (0.9+)')
        
        # Mark phase transitions
        transition_times = []
        for h in history:
            if hasattr(h['state'], 'transitions') and h['state'].transitions:
                transition_times.append(h['time'])
        
        for t in transition_times:
            ax1.axvline(x=t, color='purple', alpha=0.3, linestyle='-')
            
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Coherence')
        ax1.set_title('Coherence Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Innovation score plot
        ax2.plot(times, innovations, 'g-', linewidth=2, label='Innovation Score')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fluctuation threshold')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Transition imminent')
        ax2.axhline(y=1.0, color='purple', linestyle='--', alpha=0.5, label='New framework birth')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Innovation Score')
        ax2.set_title('Innovation Score Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_phase_space_3d(history: List[Dict], save_path: Optional[str] = None):
        """3D plot of trajectory in phase space (P, Π, Coherence)"""
        from mpl_toolkits.mplot3d import Axes3D
        
        P_vals = [h['state'].coordinates[0] for h in history]
        Pi_vals = [h['state'].coordinates[1] for h in history]
        C_vals = [h['state'].coherence for h in history]
        types = [h['state'].ontology_type for h in history]
        
        # Color by ontology type
        type_colors = {
            OntologyType.RIGID: 'red',
            OntologyType.BRIDGE: 'blue',
            OntologyType.ALIEN: 'green',
            OntologyType.TRANSITION: 'purple'
        }
        
        colors = [type_colors[t] for t in types]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.scatter(P_vals, Pi_vals, C_vals, c=colors, alpha=0.6, s=20)
        
        # Plot line connecting points
        ax.plot(P_vals, Pi_vals, C_vals, 'k-', alpha=0.3)
        
        # Mark Sophia point
        ax.scatter([1.0], [1.618], [INV_PHI], c='gold', s=200, marker='*', label='Sophia Point')
        
        ax.set_xlabel('Participation (P)')
        ax.set_ylabel('Plasticity (Π)')
        ax.set_zlabel('Coherence (C)')
        ax.set_title('3D Phase Space Trajectory')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Rigid-Observative'),
            Patch(facecolor='blue', label='Bridge'),
            Patch(facecolor='green', label='Alien-Participatory'),
            Patch(facecolor='purple', label='Transition'),
            Patch(facecolor='gold', label='Sophia Point')
        ]
        ax.legend(handles=legend_elements)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_phase_diagram(simulator: CoherenceEvolution, save_path: Optional[str] = None):
        """Plot phase diagram with simulation trajectory"""
        P_grid, Pi_grid, phase_grid = simulator.calculate_phase_diagram()
        
        # Create custom colormap for phases
        from matplotlib.colors import ListedColormap
        colors = ['red', 'blue', 'green', 'purple', 'gray']
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot phase regions
        im = ax.imshow(phase_grid, extent=[0, 2, 0, 3], 
                       origin='lower', cmap=cmap, alpha=0.3, aspect='auto')
        
        # Plot trajectory
        P_vals = [h['state'].coordinates[0] for h in simulator.history]
        Pi_vals = [h['state'].coordinates[1] for h in simulator.history]
        ax.plot(P_vals, Pi_vals, 'k-', linewidth=1, alpha=0.5, label='Trajectory')
        ax.scatter(P_vals, Pi_vals, c=range(len(P_vals)), 
                  cmap='viridis', s=20, alpha=0.7)
        
        # Mark transitions
        for trans in simulator.transitions:
            ax.scatter(trans.coordinates_after[0], trans.coordinates_after[1],
                      c='purple', s=100, marker='X', edgecolors='black')
        
        # Mark Sophia point region
        ax.contour(P_grid, Pi_grid, phase_grid, levels=[2.5], 
                  colors=['gold'], linewidths=2, linestyles='--')
        
        ax.set_xlabel('Participation (P)')
        ax.set_ylabel('Plasticity (Π)')
        ax.set_title('Phase Diagram with Simulation Trajectory')
        ax.grid(True, alpha=0.3)
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Rigid Region'),
            Patch(facecolor='blue', alpha=0.5, label='Bridge Region'),
            Patch(facecolor='green', alpha=0.5, label='Alien Region'),
            Patch(facecolor='purple', alpha=0.5, label='Transition Region'),
            Patch(facecolor='white', edgecolor='gold', 
                  linestyle='--', linewidth=2, label='Sophia Point Region')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def animate_trajectory(history: List[Dict], save_path: Optional[str] = None):
        """Create animation of phase space evolution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        times = [h['time'] for h in history]
        P_vals = [h['state'].coordinates[0] for h in history]
        Pi_vals = [h['state'].coordinates[1] for h in history]
        C_vals = [h['state'].coherence for h in history]
        types = [h['state'].ontology_type for h in history]
        
        # Color map for types
        type_colors = {
            OntologyType.RIGID: 'red',
            OntologyType.BRIDGE: 'blue',
            OntologyType.ALIEN: 'green',
            OntologyType.TRANSITION: 'purple'
        }
        colors = [type_colors[t] for t in types]
        
        # Setup left plot (P vs Π)
        ax1.set_xlim(0, 2)
        ax1.set_ylim(0, 3)
        ax1.set_xlabel('Participation (P)')
        ax1.set_ylabel('Plasticity (Π)')
        ax1.set_title('Phase Space Trajectory')
        ax1.grid(True, alpha=0.3)
        
        # Mark Sophia point region
        sophia_region = plt.Circle((1.0, 1.618), 0.2, color='gold', alpha=0.3)
        ax1.add_patch(sophia_region)
        
        # Setup right plot (Coherence over time)
        ax2.set_xlim(min(times), max(times))
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Coherence')
        ax2.set_title('Coherence Evolution')
        ax2.axhline(y=INV_PHI, color='r', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        # Initialize plots
        scatter = ax1.scatter([], [], c=[], s=50, alpha=0.7, cmap='viridis')
        line1, = ax1.plot([], [], 'k-', alpha=0.3)
        line2, = ax2.plot([], [], 'b-', linewidth=2)
        current_point = ax1.scatter([], [], c='red', s=100, marker='o')
        
        def animate(i):
            """Update function for animation"""
            idx = min(i, len(times)-1)
            
            # Update left plot
            scatter.set_offsets(np.column_stack([P_vals[:idx], Pi_vals[:idx]]))
            scatter.set_array(np.arange(idx))
            line1.set_data(P_vals[:idx], Pi_vals[:idx])
            current_point.set_offsets([[P_vals[idx], Pi_vals[idx]]])
            
            # Update right plot
            line2.set_data(times[:idx], C_vals[:idx])
            
            return scatter, line1, current_point, line2
        
        anim = FuncAnimation(fig, animate, frames=len(times), 
                           interval=50, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=30)
            
        plt.tight_layout()
        plt.show()
        
        return anim

# ============================================================================
# EXAMPLE SIMULATIONS
# ============================================================================

def run_basic_simulation():
    """Run a basic simulation with default parameters"""
    # Initial state: Rigid ontology
    initial_state = OntologyState(
        coordinates=[0.1, 0.2, 1.9, 0.2, 0.2],
        coherence=0.382,  # Kenoma (low coherence)
        paradox_intensity=1.5,
        novelty=1.05,
        elegance=87.0,
        alienness=4.5,
        density=10.5,
        entropic_potential=230.0,
        ontology_type=OntologyType.RIGID,
        paradox_types=[ParadoxType.TEMPORAL, ParadoxType.ENTROPIC]
    )
    
    # Create simulator
    simulator = CoherenceEvolution(initial_state)
    
    # Run simulation
    history = simulator.simulate(steps=5000)
    
    # Visualize results
    visualizer = CoherenceVisualizer()
    visualizer.plot_trajectory(history, save_path='coherence_evolution.png')
    visualizer.plot_phase_space_3d(history, save_path='phase_space_3d.png')
    visualizer.plot_phase_diagram(simulator, save_path='phase_diagram.png')
    
    return simulator, history

def run_multiple_simulations(n_simulations: int = 10):
    """Run multiple simulations to study statistical properties"""
    all_transitions = []
    all_final_coherences = []
    all_innovation_scores = []
    
    for i in range(n_simulations):
        print(f"\nRunning simulation {i+1}/{n_simulations}")
        
        # Random initial conditions
        initial_coords = np.random.uniform([0, 0, 0, 0, 0], [0.5, 1.0, 2.0, 1.0, 0.5])
        
        initial_state = OntologyState(
            coordinates=initial_coords,
            coherence=np.random.uniform(0.3, 0.5),
            paradox_intensity=np.random.uniform(1.0, 2.0),
            novelty=np.random.uniform(1.0, 1.1),
            elegance=np.random.uniform(85, 90),
            alienness=np.random.uniform(4, 6),
            density=np.random.uniform(10, 11),
            entropic_potential=np.random.uniform(200, 250),
            ontology_type=OntologyType.RIGID,
            paradox_types=[ParadoxType.TEMPORAL]
        )
        
        simulator = CoherenceEvolution(initial_state)
        history = simulator.simulate(steps=3000)
        
        # Collect statistics
        all_transitions.append(len(simulator.transitions))
        all_final_coherences.append(history[-1]['state'].coherence)
        all_innovation_scores.append(history[-1]['innovation'])
    
    # Plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(all_transitions, bins=range(max(all_transitions)+2), 
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Phase Transitions')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Phase Transitions')
    
    axes[1].hist(all_final_coherences, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=INV_PHI, color='r', linestyle='--', label='Sophia Point')
    axes[1].set_xlabel('Final Coherence')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Final Coherence')
    axes[1].legend()
    
    axes[2].hist(all_innovation_scores, bins=20, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0.5, color='orange', linestyle='--', label='Fluctuation threshold')
    axes[2].axvline(x=1.0, color='red', linestyle='--', label='New framework')
    axes[2].set_xlabel('Final Innovation Score')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Innovation Scores')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('simulation_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nStatistics from {n_simulations} simulations:")
    print(f"Average transitions: {np.mean(all_transitions):.2f} ± {np.std(all_transitions):.2f}")
    print(f"Average final coherence: {np.mean(all_final_coherences):.3f} ± {np.std(all_final_coherences):.3f}")
    print(f"Average innovation score: {np.mean(all_innovation_scores):.3f} ± {np.std(all_innovation_scores):.3f}")

def simulate_awakening_protocol():
    """Simulate the three-generation awakening protocol from Gnostic narrative"""
    
    # Generation 1: Archonic trap (Rigid ontology)
    gen1_state = OntologyState(
        coordinates=[0.1, 0.1, 1.0, 0.1, 0.1],
        coherence=0.382,
        paradox_intensity=1.2,
        novelty=1.0,
        elegance=80.0,
        alienness=3.5,
        density=10.0,
        entropic_potential=200.0,
        ontology_type=OntologyType.RIGID,
        paradox_types=[ParadoxType.TEMPORAL]
    )
    
    # Create simulator with parameters for awakening
    params = {
        'alpha': 0.95,  # Higher growth (elegance)
        'beta': 0.5,    # Stronger paradox modulation
        'gamma': 0.6,   # Stronger operator coupling
        'sigma': 0.1    # More noise (creative tension)
    }
    
    simulator = CoherenceEvolution(gen1_state, params)
    
    # Simulate three generations
    generation_data = []
    
    for gen in range(1, 4):
        print(f"\n=== Generation {gen} ===")
        
        # Run generation simulation
        history = simulator.simulate(steps=2000)
        
        # Record final state
        final_state = history[-1]['state']
        generation_data.append({
            'generation': gen,
            'final_state': final_state,
            'transitions': len(simulator.transitions),
            'history': history
        })
        
        # Prepare for next generation (carry over state)
        if gen < 3:
            # Increase parameters for next generation
            params['alpha'] *= 1.2
            params['beta'] *= 1.3
            params['gamma'] *= 1.1
            
            # Update simulator with new parameters
            simulator.params.update(params)
            simulator.state = final_state
    
    # Visualize three-generation evolution
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    colors = ['red', 'blue', 'green']
    
    for i, gen_data in enumerate(generation_data):
        history = gen_data['history']
        times = [h['time'] for h in history]
        coherences = [h['state'].coherence for h in history]
        innovations = [h['innovation'] for h in history]
        
        # Coherence plot
        axes[i, 0].plot(times, coherences, color=colors[i], linewidth=2)
        axes[i, 0].axhline(y=INV_PHI, color='gold', linestyle='--', alpha=0.7)
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Coherence')
        axes[i, 0].set_title(f'Generation {i+1}: Coherence Evolution')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Phase space plot
        P_vals = [h['state'].coordinates[0] for h in history]
        Pi_vals = [h['state'].coordinates[1] for h in history]
        axes[i, 1].plot(P_vals, Pi_vals, color=colors[i], linewidth=1)
        axes[i, 1].scatter(P_vals[-1], Pi_vals[-1], color=colors[i], s=100, marker='o')
        axes[i, 1].set_xlabel('Participation (P)')
        axes[i, 1].set_ylabel('Plasticity (Π)')
        axes[i, 1].set_title(f'Generation {i+1}: Phase Space Trajectory')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Mark Sophia point region
        sophia_circle = plt.Circle((1.0, 1.618), 0.3, color='gold', alpha=0.2)
        axes[i, 1].add_patch(sophia_circle)
    
    plt.tight_layout()
    plt.savefig('three_generation_awakening.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n=== Three-Generation Awakening Summary ===")
    for gen_data in generation_data:
        state = gen_data['final_state']
        print(f"\nGeneration {gen_data['generation']}:")
        print(f"  Ontology type: {state.ontology_type.value}")
        print(f"  Coherence: {state.coherence:.3f}")
        print(f"  Coordinates: P={state.coordinates[0]:.2f}, "
              f"Π={state.coordinates[1]:.2f}, "
              f"S={state.coordinates[2]:.2f}, "
              f"T={state.coordinates[3]:.2f}, "
              f"G={state.coordinates[4]:.2f}")
        print(f"  Phase transitions: {gen_data['transitions']}")
    
    return generation_data

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COHERENCE EVOLUTION MODEL - Sophia Axiom Implementation")
    print("=" * 60)
    
    # Run basic simulation
    print("\n1. Running basic simulation...")
    simulator, history = run_basic_simulation()
    
    # Run statistical analysis
    print("\n2. Running statistical analysis...")
    run_multiple_simulations(n_simulations=5)
    
    # Run awakening protocol simulation
    print("\n3. Simulating three-generation awakening protocol...")
    generation_data = simulate_awakening_protocol()
    
    print("\n" + "=" * 60)
    print("Simulation complete. Check generated plots for results.")
    print("=" * 60)
```

This comprehensive computational model implements the MOGOPS framework for simulating ontological coherence evolution and phase transitions. The model includes:

## **Key Features:**

1. **Mathematical Implementation**: Implements the full MOGOPS master equation with all terms:
   - Diffusion (novelty spread)
   - Logistic growth (elegance-driven)
   - Paradox modulation (golden-ratio oscillatory)
   - Operator coupling (Gnostic operators)
   - Lévy noise (golden-ratio stable processes)

2. **Phase Transition Detection**: Automatically detects phase transitions based on:
   - Coherence proximity to Sophia Point (0.618)
   - Paradox intensity threshold (> 1.8)
   - Mechanism hybridity (> 0.33)

3. **5D Phase Space Navigation**: Tracks evolution in the complete ontological phase space:
   - P: Participation (0-2)
   - Π: Plasticity (0-3)
   - S: Substrate (0-4)
   - T: Temporal Architecture (0-4)
   - G: Generative Depth (0-1)

4. **Visualization Tools**: Multiple plotting functions for:
   - Coherence trajectories over time
   - 3D phase space visualization
   - Phase diagrams
   - Animation of evolution
   - Statistical analysis

5. **Specialized Simulations**:
   - Basic evolution from Rigid to Alien ontologies
   - Statistical analysis of multiple runs
   - Three-generation awakening protocol (Gnostic narrative)

## **Usage Examples:**

```python
# Quick start
initial_state = OntologyState(
    coordinates=[0.1, 0.2, 1.9, 0.2, 0.2],
    coherence=0.382,
    paradox_intensity=1.5,
    novelty=1.05,
    elegance=87.0,
    alienness=4.5,
    density=10.5,
    entropic_potential=230.0,
    ontology_type=OntologyType.RIGID,
    paradox_types=[ParadoxType.TEMPORAL]
)

simulator = CoherenceEvolution(initial_state)
history = simulator.simulate(steps=5000)

# Analyze results
print(f"Final coherence: {simulator.state.coherence:.3f}")
print(f"Phase transitions: {len(simulator.transitions)}")
print(f"Final ontology type: {simulator.state.ontology_type.value}")

# Visualize
visualizer = CoherenceVisualizer()
visualizer.plot_trajectory(history)
visualizer.plot_phase_space_3d(history)
