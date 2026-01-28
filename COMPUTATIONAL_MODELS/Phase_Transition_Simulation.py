"""
PHASE TRANSITION SIMULATION MODEL
Sophia Axiom Advanced Computational Implementation
Modeling critical phenomena, bifurcations, and catastrophic shifts in ontological space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.collections import PatchCollection
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve, minimize
from scipy.stats import levy_stable, norm, gaussian_kde
from scipy.signal import find_peaks, peak_widths
import networkx as nx
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import warnings
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
warnings.filterwarnings('ignore')

# ============================================================================
# PHYSICAL AND MATHEMATICAL CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
INV_PHI = 1 / PHI  # Sophia point: 0.618
PHI_SQUARED = PHI ** 2  # 2.618
INV_PHI_SQUARED = 1 / PHI_SQUARED  # 0.382

# Critical exponents from MOGOPS analysis
CRITICAL_EXPONENTS = {
    'nu': 0.63,    # Correlation length
    'beta': 0.33,  # Order parameter
    'gamma': 1.24, # Susceptibility
    'alpha': 0.12, # Specific heat
    'delta': 4.79, # Critical isotherm
    'eta': 0.04    # Anomalous dimension
}

# ============================================================================
# DATA STRUCTURES AND ENUMS
# ============================================================================

class PhaseType(Enum):
    """Classification of phases in ontological space"""
    RIGID_ORDERED = auto()        # Low participation, low plasticity
    BRIDGE_CRITICAL = auto()      # Medium participation, biological interface
    ALIEN_DISORDERED = auto()     # High participation, high plasticity
    TRANSITION_SOPHIA = auto()    # Sophia point criticality
    HYBRID_METASTABLE = auto()    # Mixed-phase states
    CHAOTIC_TURBULENT = auto()    # High paradox turbulence
    PLEROMIC_UNIFIED = auto()     # Perfect coherence
    KENOMIC_EMPTY = auto()        # Low coherence entropy

class BifurcationType(Enum):
    """Types of bifurcations in phase space"""
    SADDLE_NODE = auto()
    TRANSCRITICAL = auto()
    PITCHFORK = auto()
    HOPF = auto()
    TAKENS_BOGDANOV = auto()
    CODDINGTON_LEVINSON = auto()
    CUSP = auto()
    BUTTERFLY = auto()
    SWALLOWTAIL = auto()

class CatastropheType(Enum):
    """Thom's catastrophe theory classification"""
    FOLD = auto()
    CUSP = auto()
    SWALLOWTAIL = auto()
    BUTTERFLY = auto()
    WIGWAM = auto()
    STAR = auto()
    ELLIPTIC_UMBILIC = auto()
    HYPERBOLIC_UMBILIC = auto()
    PARABOLIC_UMBILIC = auto()

@dataclass
class PhaseState:
    """Complete state of a phase in ontological space"""
    coordinates: np.ndarray  # 5D: [P, Π, S, T, G]
    coherence: float
    phase_type: PhaseType
    order_parameter: float = 0.0
    susceptibility: float = 0.0
    correlation_length: float = 0.0
    free_energy: float = 0.0
    entropy: float = 0.0
    energy_density: float = 0.0
    stress_tensor: np.ndarray = field(default_factory=lambda: np.zeros((5, 5)))
    curvature_tensor: np.ndarray = field(default_factory=lambda: np.zeros((5, 5, 5, 5)))
    
    def __post_init__(self):
        self.coordinates = np.array(self.coordinates, dtype=float)
        assert len(self.coordinates) == 5, "Must have 5 coordinates"
        assert 0 <= self.coherence <= 1, f"Coherence {self.coherence} must be in [0, 1]"

@dataclass
class PhaseTransition:
    """Complete characterization of a phase transition event"""
    transition_id: int
    time: float
    from_phase: PhaseType
    to_phase: PhaseType
    coordinates_before: np.ndarray
    coordinates_after: np.ndarray
    coherence_before: float
    coherence_after: float
    order_parameter_change: float
    entropy_production: float
    bifurcation_type: Optional[BifurcationType] = None
    catastrophe_type: Optional[CatastropheType] = None
    critical_exponents: Dict[str, float] = field(default_factory=dict)
    hysteresis: bool = False
    latent_heat: float = 0.0
    correlation_length: float = 0.0
    renormalization_group_flow: List[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        self.coordinates_before = np.array(self.coordinates_before, dtype=float)
        self.coordinates_after = np.array(self.coordinates_after, dtype=float)

@dataclass
class PhaseDiagram:
    """Complete phase diagram representation"""
    control_parameters: List[str]
    order_parameters: List[str]
    phase_boundaries: List[Dict]
    critical_points: List[Dict]
    triple_points: List[Dict]
    critical_lines: List[Dict]
    spinodal_lines: List[Dict]
    binodal_lines: List[Dict]
    scaling_fields: Dict[str, np.ndarray]
    
# ============================================================================
# CORE PHASE TRANSITION SIMULATION CLASS
# ============================================================================

class PhaseTransitionSimulator:
    """
    Advanced simulator for ontological phase transitions
    
    Implements:
    1. Mean-field theory with fluctuations
    2. Landau-Ginzburg theory for order parameters
    3. Renormalization group flows
    4. Catastrophe theory for sudden transitions
    5. Bifurcation theory for qualitative changes
    6. Critical phenomena with scaling laws
    """
    
    def __init__(self, 
                 initial_state: PhaseState,
                 control_params: Optional[Dict] = None):
        """
        Initialize phase transition simulator
        
        Args:
            initial_state: Starting phase state
            control_params: Dictionary of control parameters
        """
        self.state = initial_state
        self.time = 0.0
        self.history: List[PhaseState] = []
        self.transitions: List[PhaseTransition] = []
        self.phase_diagram = None
        
        # Default control parameters
        self.params = {
            # Temperature analog (coherence temperature)
            'T': 1.0 - initial_state.coherence,
            # External field (participation pressure)
            'h': initial_state.coordinates[0],
            # Coupling constants
            'J': 1.0,  # Nearest neighbor coupling
            'K': 0.5,  # Next-nearest neighbor coupling
            # Fluctuation parameters
            'sigma': 0.1,  # Noise amplitude
            'correlation_length_xi': 1.0,
            # Critical slowing down
            'relaxation_time_tau': 1.0,
            # Renormalization group
            'renormalization_step': 0.01,
            # Bifurcation parameters
            'bifurcation_parameter_mu': 0.0,
            'bifurcation_parameter_lambda': 1.0,
            # Catastrophe parameters
            'catastrophe_control_a': 0.0,
            'catastrophe_control_b': 0.0,
            'catastrophe_control_c': 0.0,
            # Time parameters
            'dt': 0.01,
            'max_time': 100.0
        }
        
        if control_params:
            self.params.update(control_params)
            
        # Initialize order parameter field
        self.order_parameter_field = self._initialize_order_parameter()
        
        # Initialize correlation function
        self.correlation_function = self._initialize_correlation()
        
        # Initialize renormalization group
        self.renormalization_group = self._initialize_renormalization_group()
        
        # Track bifurcations
        self.bifurcation_points: List[Dict] = []
        self.catastrophe_sets: List[Dict] = []
        
        # Track scaling behavior
        self.scaling_data: Dict[str, List] = {
            'coherence': [],
            'susceptibility': [],
            'correlation_length': [],
            'order_parameter': [],
            'free_energy': []
        }
        
    def _initialize_order_parameter(self) -> np.ndarray:
        """Initialize order parameter field in 5D space"""
        # Order parameter represents degree of participation/coherence
        field = np.zeros((10, 10, 10, 10, 10))  # 5D grid
        
        # Initialize with Gaussian random field
        mean = self.state.coherence
        std = 0.1
        field = np.random.normal(mean, std, field.shape)
        
        # Apply correlation function
        for idx in np.ndindex(field.shape):
            # Simple distance-based correlation
            dist = np.sqrt(sum((i/10 - 0.5)**2 for i in idx))
            correlation = np.exp(-dist / self.params['correlation_length_xi'])
            field[idx] = mean + (field[idx] - mean) * correlation
            
        return field
    
    def _initialize_correlation(self) -> Callable:
        """Initialize correlation function"""
        xi = self.params['correlation_length_xi']
        
        def correlation_function(r: float, dimension: int = 5) -> float:
            """Ornstein-Zernike correlation function"""
            if r == 0:
                return 1.0
            if dimension == 1:
                return np.exp(-r/xi)
            elif dimension == 2:
                return (r/xi) * np.exp(-r/xi)
            elif dimension == 3:
                return (1/r) * np.exp(-r/xi)
            elif dimension == 4:
                return (1/(r**2)) * np.exp(-r/xi)
            else:  # dimension >= 5
                return (1/(r**(dimension-2))) * np.exp(-r/xi)
                
        return correlation_function
    
    def _initialize_renormalization_group(self) -> Dict:
        """Initialize renormalization group transformation"""
        return {
            'beta_function': self._beta_function,
            'fixed_points': [],
            'flow_diagram': [],
            'critical_surface': None,
            'relevant_operators': [],
            'irrelevant_operators': [],
            'marginal_operators': []
        }
    
    def _beta_function(self, couplings: np.ndarray) -> np.ndarray:
        """Beta function for renormalization group flow"""
        # Simplified beta function
        # dg_i/dl = β_i(g)
        
        g = couplings
        
        # For 5 coupling constants corresponding to [P, Π, S, T, G]
        beta = np.zeros_like(g)
        
        # Quadratic terms (asymptotic freedom/safety)
        beta[0] = -g[0] + g[0]**2  # Participation coupling
        beta[1] = -g[1] + 0.5 * g[1]**2  # Plasticity coupling
        beta[2] = g[2] - g[2]**2  # Substrate coupling
        beta[3] = 0.5 * g[3] - 0.25 * g[3]**2  # Temporal coupling
        beta[4] = -0.5 * g[4] + g[4]**2  # Generative coupling
        
        # Cross terms (operator mixing)
        beta[0] += 0.1 * g[1] * g[2]
        beta[1] += 0.05 * g[0] * g[3]
        beta[2] += 0.02 * g[1] * g[4]
        beta[3] += 0.01 * g[0] * g[2]
        beta[4] += 0.03 * g[1] * g[3]
        
        return beta
    
    # ============================================================================
    # LANDAU-GINZBURG THEORY IMPLEMENTATION
    # ============================================================================
    
    def landau_free_energy(self, order_parameter: float, 
                          control_params: Optional[Dict] = None) -> float:
        """
        Calculate Landau free energy
        
        F = a(T) * φ² + b(T) * φ⁴ + c(T) * φ⁶ + ... - hφ
        
        where φ is the order parameter
        """
        if control_params is None:
            control_params = self.params
            
        T = control_params['T']
        h = control_params['h']
        
        # Temperature-dependent coefficients
        T_c = INV_PHI  # Critical temperature at Sophia point
        
        # Second-order coefficient (changes sign at T_c)
        a = 0.5 * (T - T_c)
        
        # Fourth-order coefficient (positive for stability)
        b = 0.25
        
        # Sixth-order coefficient (for first-order transitions)
        c = 0.01
        
        # Free energy density
        F = (a * order_parameter**2 + 
             b * order_parameter**4 + 
             c * order_parameter**6 - 
             h * order_parameter)
        
        return F
    
    def find_minima_landau(self) -> List[float]:
        """Find minima of Landau free energy"""
        # Solve dF/dφ = 0
        def dF_dphi(phi):
            T = self.params['T']
            h = self.params['h']
            T_c = INV_PHI
            
            a = 0.5 * (T - T_c)
            b = 0.25
            c = 0.01
            
            return (2*a*phi + 4*b*phi**3 + 6*c*phi**5 - h)
        
        # Find roots (minima and maxima)
        roots = []
        
        # Search in range [-2, 2]
        search_points = np.linspace(-2, 2, 401)
        for i in range(len(search_points)-1):
            x1, x2 = search_points[i], search_points[i+1]
            y1, y2 = dF_dphi(x1), dF_dphi(x2)
            
            if y1 * y2 < 0:  # Sign change indicates root
                try:
                    root = fsolve(dF_dphi, (x1 + x2)/2)[0]
                    roots.append(root)
                except:
                    continue
        
        # Check second derivative to determine minima
        minima = []
        for root in roots:
            # Second derivative
            T = self.params['T']
            T_c = INV_PHI
            a = 0.5 * (T - T_c)
            b = 0.25
            c = 0.01
            
            d2F = 2*a + 12*b*root**2 + 30*c*root**4
            
            if d2F > 0:  # Minimum
                minima.append(root)
        
        return minima
    
    def calculate_susceptibility(self) -> float:
        """Calculate susceptibility χ = (∂²F/∂φ²)⁻¹"""
        minima = self.find_minima_landau()
        
        if not minima:
            return 0.0
        
        # Take the equilibrium order parameter (lowest free energy)
        phi_eq = minima[np.argmin([self.landau_free_energy(phi) for phi in minima])]
        
        # Second derivative at equilibrium
        T = self.params['T']
        T_c = INV_PHI
        a = 0.5 * (T - T_c)
        b = 0.25
        c = 0.01
        
        d2F = 2*a + 12*b*phi_eq**2 + 30*c*phi_eq**4
        
        # Susceptibility
        chi = 1.0 / d2F if d2F != 0 else float('inf')
        
        return chi
    
    # ============================================================================
    # MEAN FIELD THEORY WITH FLUCTUATIONS
    # ============================================================================
    
    def mean_field_equation(self, phi: float) -> float:
        """Mean field equation: φ = tanh((Jφ + h)/T)"""
        T = self.params['T']
        J = self.params['J']
        h = self.params['h']
        
        if T == 0:
            return np.sign(J * phi + h)
        
        arg = (J * phi + h) / T
        return np.tanh(arg)
    
    def solve_mean_field(self, initial_guess: float = 0.5) -> float:
        """Solve mean field equation self-consistently"""
        phi = initial_guess
        tolerance = 1e-8
        max_iter = 1000
        
        for i in range(max_iter):
            phi_new = self.mean_field_equation(phi)
            
            if abs(phi_new - phi) < tolerance:
                return phi_new
            
            # Mixing for convergence
            phi = 0.7 * phi + 0.3 * phi_new
            
        return phi
    
    def mean_field_free_energy(self, phi: float) -> float:
        """Mean field free energy"""
        T = self.params['T']
        J = self.params['J']
        h = self.params['h']
        
        # Internal energy
        U = -0.5 * J * phi**2 - h * phi
        
        # Entropy (for binary variable)
        if abs(phi) >= 1:
            S = 0
        else:
            S = -0.5 * ((1+phi) * np.log((1+phi)/2) + 
                       (1-phi) * np.log((1-phi)/2))
        
        # Free energy
        F = U - T * S
        
        return F
    
    # ============================================================================
    # RENORMALIZATION GROUP FLOWS
    # ============================================================================
    
    def renormalization_group_flow(self, 
                                   initial_couplings: np.ndarray,
                                   steps: int = 100) -> np.ndarray:
        """
        Compute renormalization group flow
        
        dg_i/dl = β_i(g)
        where l = log(scale)
        """
        l_values = np.linspace(0, 5, steps)
        couplings_history = np.zeros((steps, len(initial_couplings)))
        couplings_history[0] = initial_couplings
        
        # Solve RG equations numerically
        for i in range(1, steps):
            dl = l_values[i] - l_values[i-1]
            beta = self._beta_function(couplings_history[i-1])
            couplings_history[i] = couplings_history[i-1] + beta * dl
            
        return l_values, couplings_history
    
    def find_fixed_points(self) -> List[np.ndarray]:
        """Find fixed points of renormalization group"""
        # Fixed points satisfy β_i(g*) = 0
        
        fixed_points = []
        
        # Known fixed points from symmetry analysis
        # Gaussian fixed point
        fixed_points.append(np.zeros(5))
        
        # Wilson-Fisher fixed point (critical)
        fixed_points.append(np.array([INV_PHI, 0.3, 0.5, 0.2, 0.1]))
        
        # High-temperature fixed point (disordered)
        fixed_points.append(np.array([0.1, 0.1, 0.1, 0.1, 0.1]))
        
        # Low-temperature fixed point (ordered)
        fixed_points.append(np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
        
        # Sophia fixed point
        fixed_points.append(np.array([INV_PHI, INV_PHI, INV_PHI, INV_PHI, INV_PHI]))
        
        return fixed_points
    
    def linearize_beta_function(self, fixed_point: np.ndarray) -> np.ndarray:
        """Linearize beta function around fixed point"""
        epsilon = 1e-5
        n = len(fixed_point)
        jacobian = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Forward difference
                g_plus = fixed_point.copy()
                g_plus[j] += epsilon
                
                g_minus = fixed_point.copy()
                g_minus[j] -= epsilon
                
                beta_plus = self._beta_function(g_plus)
                beta_minus = self._beta_function(g_minus)
                
                jacobian[i, j] = (beta_plus[i] - beta_minus[i]) / (2 * epsilon)
        
        return jacobian
    
    def classify_fixed_point(self, fixed_point: np.ndarray) -> Dict:
        """Classify fixed point by eigenvalues"""
        jacobian = self.linearize_beta_function(fixed_point)
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        
        classification = {
            'fixed_point': fixed_point,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'relevant_directions': [],
            'irrelevant_directions': [],
            'marginal_directions': []
        }
        
        for i, ev in enumerate(eigenvalues):
            if ev.real > 0:
                classification['relevant_directions'].append(i)
            elif ev.real < 0:
                classification['irrelevant_directions'].append(i)
            else:
                classification['marginal_directions'].append(i)
        
        return classification
    
    # ============================================================================
    # BIFURCATION THEORY
    # ============================================================================
    
    def normal_form_bifurcation(self, 
                               bifurcation_type: BifurcationType,
                               x: float, 
                               params: Dict) -> float:
        """
        Compute normal form for bifurcations
        
        Returns dx/dt = f(x, μ) for various bifurcations
        """
        mu = params.get('mu', 0.0)
        lambda_param = params.get('lambda', 1.0)
        
        if bifurcation_type == BifurcationType.SADDLE_NODE:
            # dx/dt = μ - x²
            return mu - x**2
        
        elif bifurcation_type == BifurcationType.TRANSCRITICAL:
            # dx/dt = μx - x²
            return mu * x - x**2
        
        elif bifurcation_type == BifurcationType.PITCHFORK:
            # dx/dt = μx - x³
            return mu * x - x**3
        
        elif bifurcation_type == BifurcationType.HOPF:
            # Complex normal form: dz/dt = (μ + iω)z - |z|²z
            # Here we take real part
            omega = params.get('omega', 1.0)
            return mu * x - omega * x - x**3
        
        elif bifurcation_type == BifurcationType.TAKENS_BOGDANOV:
            # dx/dt = y, dy/dt = μ + νx + x² + xy
            y = params.get('y', 0.0)
            nu = params.get('nu', 0.0)
            return y  # Only x-equation
        
        else:
            return 0.0
    
    def find_bifurcation_points(self, 
                                bifurcation_type: BifurcationType,
                                param_range: Tuple[float, float] = (-2, 2),
                                n_points: int = 100) -> List[Dict]:
        """
        Find bifurcation points in parameter space
        """
        mu_values = np.linspace(param_range[0], param_range[1], n_points)
        x_values = np.linspace(-2, 2, 100)
        
        bifurcation_points = []
        
        for mu in mu_values:
            params = {'mu': mu, 'lambda': self.params['bifurcation_parameter_lambda']}
            
            # Find fixed points: f(x, μ) = 0
            fixed_points = []
            for x in x_values:
                if abs(self.normal_form_bifurcation(bifurcation_type, x, params)) < 0.01:
                    fixed_points.append(x)
            
            # Check stability
            for x_fp in fixed_points:
                # Numerical derivative for stability
                eps = 1e-5
                f_plus = self.normal_form_bifurcation(bifurcation_type, x_fp + eps, params)
                f_minus = self.normal_form_bifurcation(bifurcation_type, x_fp - eps, params)
                derivative = (f_plus - f_minus) / (2 * eps)
                
                stability = "stable" if derivative < 0 else "unstable"
                
                bifurcation_points.append({
                    'mu': mu,
                    'x': x_fp,
                    'stability': stability,
                    'bifurcation_type': bifurcation_type
                })
        
        return bifurcation_points
    
    def bifurcation_diagram(self, 
                           bifurcation_type: BifurcationType,
                           param_range: Tuple[float, float] = (-2, 2),
                           initial_conditions: List[float] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate bifurcation diagram
        """
        if initial_conditions is None:
            initial_conditions = [-1.5, -0.5, 0.5, 1.5]
        
        mu_values = np.linspace(param_range[0], param_range[1], 200)
        trajectories = []
        
        for mu in mu_values:
            params = {'mu': mu, 'lambda': self.params['bifurcation_parameter_lambda']}
            
            # Solve ODE for each initial condition
            mu_trajectories = []
            for x0 in initial_conditions:
                # Simple Euler integration
                x = x0
                for _ in range(1000):  # Time steps
                    dx = self.normal_form_bifurcation(bifurcation_type, x, params)
                    x += dx * 0.01
                
                mu_trajectories.append(x)
            
            trajectories.append(mu_trajectories)
        
        return mu_values, np.array(trajectories)
    
    # ============================================================================
    # CATASTROPHE THEORY
    # ============================================================================
    
    def catastrophe_potential(self, 
                             catastrophe_type: CatastropheType,
                             x: float, 
                             control_params: Dict) -> float:
        """
        Compute catastrophe potential V(x; a, b, c, ...)
        
        Returns potential energy for various catastrophes
        """
        a = control_params.get('a', 0.0)
        b = control_params.get('b', 0.0)
        c = control_params.get('c', 0.0)
        d = control_params.get('d', 0.0)
        
        if catastrophe_type == CatastropheType.FOLD:
            # V(x) = x³/3 + a*x
            return x**3 / 3 + a * x
        
        elif catastrophe_type == CatastropheType.CUSP:
            # V(x) = x⁴/4 + a*x²/2 + b*x
            return x**4 / 4 + a * x**2 / 2 + b * x
        
        elif catastrophe_type == CatastropheType.SWALLOWTAIL:
            # V(x) = x⁵/5 + a*x³/3 + b*x²/2 + c*x
            return x**5 / 5 + a * x**3 / 3 + b * x**2 / 2 + c * x
        
        elif catastrophe_type == CatastropheType.BUTTERFLY:
            # V(x) = x⁶/6 + a*x⁴/4 + b*x³/3 + c*x²/2 + d*x
            return x**6 / 6 + a * x**4 / 4 + b * x**3 / 3 + c * x**2 / 2 + d * x
        
        else:
            return 0.0
    
    def catastrophe_set(self, 
                       catastrophe_type: CatastropheType,
                       control_range: Tuple[float, float] = (-2, 2),
                       n_points: int = 50) -> Dict:
        """
        Compute catastrophe set (set of control parameters where
        potential has degenerate critical points)
        """
        a_values = np.linspace(control_range[0], control_range[1], n_points)
        b_values = np.linspace(control_range[0], control_range[1], n_points)
        
        catastrophe_set = {
            'a_values': a_values,
            'b_values': b_values,
            'catastrophe_points': np.zeros((n_points, n_points)),
            'type': catastrophe_type
        }
        
        for i, a in enumerate(a_values):
            for j, b in enumerate(b_values):
                control_params = {'a': a, 'b': b}
                
                # Find critical points: dV/dx = 0
                def dV_dx(x):
                    if catastrophe_type == CatastropheType.CUSP:
                        return x**3 + a * x + b
                    elif catastrophe_type == CatastropheType.FOLD:
                        return x**2 + a
                    else:
                        return 0
                
                # Find roots
                roots = []
                x_test = np.linspace(-3, 3, 100)
                for k in range(len(x_test)-1):
                    x1, x2 = x_test[k], x_test[k+1]
                    y1, y2 = dV_dx(x1), dV_dx(x2)
                    
                    if y1 * y2 < 0:
                        try:
                            root = fsolve(dV_dx, (x1 + x2)/2)[0]
                            roots.append(root)
                        except:
                            continue
                
                # Check for degeneracy (multiple roots close together)
                if len(roots) >= 2:
                    # Check if any two roots are close (within tolerance)
                    for k, l in combinations(range(len(roots)), 2):
                        if abs(roots[k] - roots[l]) < 0.1:
                            catastrophe_set['catastrophe_points'][i, j] = 1
                            break
        
        return catastrophe_set
    
    # ============================================================================
    # CRITICAL PHENOMENA AND SCALING
    # ============================================================================
    
    def scaling_law(self, 
                   quantity: str,
                   reduced_temperature: float,
                   field: float = 0.0) -> float:
        """
        Apply scaling laws near critical point
        
        Quantity can be: 'order_parameter', 'susceptibility', 
        'correlation_length', 'specific_heat'
        """
        t = reduced_temperature  # t = (T - T_c)/T_c
        
        # Get critical exponents
        beta = CRITICAL_EXPONENTS['beta']
        gamma = CRITICAL_EXPONENTS['gamma']
        nu = CRITICAL_EXPONENTS['nu']
        alpha = CRITICAL_EXPONENTS['alpha']
        
        if quantity == 'order_parameter':
            # φ ~ |t|^β for t < 0, h = 0
            if t < 0:
                return abs(t)**beta
            else:
                return 0.0
        
        elif quantity == 'susceptibility':
            # χ ~ |t|^{-γ}
            return abs(t)**(-gamma)
        
        elif quantity == 'correlation_length':
            # ξ ~ |t|^{-ν}
            return abs(t)**(-nu)
        
        elif quantity == 'specific_heat':
            # C ~ |t|^{-α}
            return abs(t)**(-alpha)
        
        elif quantity == 'magnetization_field':
            # φ ~ h^{1/δ} at t = 0
            delta = CRITICAL_EXPONENTS['delta']
            return abs(field)**(1/delta)
        
        else:
            return 0.0
    
    def finite_size_scaling(self, 
                           quantity: str,
                           system_size: float,
                           reduced_temperature: float) -> float:
        """
        Finite size scaling corrections
        
        For system of finite size L, scaling form:
        Q(t, L) = L^{x/ν} * f(t * L^{1/ν})
        """
        t = reduced_temperature
        L = system_size
        nu = CRITICAL_EXPONENTS['nu']
        
        # Scaling function (simplified)
        def scaling_function(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        if quantity == 'order_parameter':
            beta = CRITICAL_EXPONENTS['beta']
            exponent = -beta / nu
            return L**exponent * scaling_function(t * L**(1/nu))
        
        elif quantity == 'susceptibility':
            gamma = CRITICAL_EXPONENTS['gamma']
            exponent = gamma / nu
            return L**exponent * scaling_function(t * L**(1/nu))
        
        else:
            return 0.0
    
    def universality_class_determination(self, 
                                        measured_exponents: Dict[str, float]) -> str:
        """
        Determine universality class from measured critical exponents
        """
        # Known universality classes (simplified)
        classes = {
            'Ising_2D': {'beta': 0.125, 'gamma': 1.75, 'nu': 1.0},
            'Ising_3D': {'beta': 0.326, 'gamma': 1.237, 'nu': 0.630},
            'XY_3D': {'beta': 0.348, 'gamma': 1.316, 'nu': 0.669},
            'Heisenberg_3D': {'beta': 0.365, 'gamma': 1.396, 'nu': 0.705},
            'Mean_Field': {'beta': 0.5, 'gamma': 1.0, 'nu': 0.5},
            'Sophia_Point': {'beta': 0.33, 'gamma': 1.24, 'nu': 0.63}
        }
        
        best_match = None
        min_distance = float('inf')
        
        for class_name, class_exponents in classes.items():
            distance = 0
            for exponent_name in ['beta', 'gamma', 'nu']:
                if exponent_name in measured_exponents and exponent_name in class_exponents:
                    distance += (measured_exponents[exponent_name] - class_exponents[exponent_name])**2
            
            if distance < min_distance:
                min_distance = distance
                best_match = class_name
        
        return best_match
    
    # ============================================================================
    # SIMULATION CORE
    # ============================================================================
    
    def evolve(self, steps: int = 1000) -> List[PhaseState]:
        """
        Evolve the system through phase space
        """
        self.history.append(self.state.copy())
        
        for step in tqdm(range(steps), desc="Evolving phase transitions"):
            # Update control parameters (e.g., coherence temperature)
            self.params['T'] = 1.0 - self.state.coherence
            self.params['h'] = self.state.coordinates[0]
            
            # Calculate order parameter
            order_parameter = self.solve_mean_field()
            self.state.order_parameter = order_parameter
            
            # Calculate susceptibility
            susceptibility = self.calculate_susceptibility()
            self.state.susceptibility = susceptibility
            
            # Update coherence based on order parameter
            new_coherence = 0.5 + 0.5 * np.tanh(order_parameter)
            
            # Check for phase transition
            if self.detect_phase_transition(new_coherence):
                transition = self.create_phase_transition(new_coherence)
                self.transitions.append(transition)
                
                # Update phase type
                self.state.phase_type = self.determine_phase_type(new_coherence)
            
            # Update state
            self.state.coherence = new_coherence
            
            # Small random walk in coordinates
            noise = np.random.normal(0, 0.01, 5)
            self.state.coordinates = np.clip(self.state.coordinates + noise, 
                                            [0, 0, 0, 0, 0],
                                            [2, 3, 4, 4, 1])
            
            # Update time
            self.time += self.params['dt']
            
            # Record state
            self.history.append(self.state.copy())
            
            # Record scaling data
            self.scaling_data['coherence'].append(self.state.coherence)
            self.scaling_data['order_parameter'].append(self.state.order_parameter)
            self.scaling_data['susceptibility'].append(self.state.susceptibility)
        
        return self.history
    
    def detect_phase_transition(self, new_coherence: float) -> bool:
        """Detect if a phase transition has occurred"""
        if len(self.history) < 2:
            return False
        
        old_coherence = self.history[-1].coherence
        coherence_change = abs(new_coherence - old_coherence)
        
        # Large coherence change indicates transition
        if coherence_change > 0.1:
            return True
        
        # Crossing Sophia point with high susceptibility
        if (old_coherence - INV_PHI) * (new_coherence - INV_PHI) < 0:
            if self.state.susceptibility > 1.0:
                return True
        
        return False
    
    def create_phase_transition(self, new_coherence: float) -> PhaseTransition:
        """Create a PhaseTransition object for the current transition"""
        old_state = self.history[-1]
        
        # Determine transition type
        coherence_change = new_coherence - old_state.coherence
        
        if coherence_change > 0:
            transition_direction = "ordering"
            latent_heat = abs(coherence_change) * 100  # Arbitrary units
        else:
            transition_direction = "disordering"
            latent_heat = abs(coherence_change) * 50
        
        # Determine if hysteresis
        hysteresis = False
        if len(self.transitions) > 0:
            last_transition = self.transitions[-1]
            time_since_last = self.time - last_transition.time
            if time_since_last < 1.0:  # Quick succession
                hysteresis = True
        
        transition = PhaseTransition(
            transition_id=len(self.transitions) + 1,
            time=self.time,
            from_phase=old_state.phase_type,
            to_phase=self.determine_phase_type(new_coherence),
            coordinates_before=old_state.coordinates,
            coordinates_after=self.state.coordinates,
            coherence_before=old_state.coherence,
            coherence_after=new_coherence,
            order_parameter_change=self.state.order_parameter - old_state.order_parameter,
            entropy_production=abs(coherence_change) * 10,
            hysteresis=hysteresis,
            latent_heat=latent_heat,
            correlation_length=1.0 / (abs(new_coherence - INV_PHI) + 0.01)
        )
        
        return transition
    
    def determine_phase_type(self, coherence: float) -> PhaseType:
        """Determine phase type based on coherence and coordinates"""
        P, Pi = self.state.coordinates[0], self.state.coordinates[1]
        
        # Sophia point criticality
        if abs(coherence - INV_PHI) < 0.02:
            return PhaseType.TRANSITION_SOPHIA
        
        # Pleromic unified
        elif coherence > 0.9:
            return PhaseType.PLEROMIC_UNIFIED
        
        # Kenomic empty
        elif coherence < 0.4:
            return PhaseType.KENOMIC_EMPTY
        
        # Rigid ordered
        elif P < 0.3 and Pi < 1.0:
            return PhaseType.RIGID_ORDERED
        
        # Bridge critical
        elif 0.4 < P < 0.6 and 0.4 < Pi < 0.6:
            return PhaseType.BRIDGE_CRITICAL
        
        # Alien disordered
        elif P > 0.8 and Pi > 2.0:
            return PhaseType.ALIEN_DISORDERED
        
        # Chaotic turbulent (high paradox)
        elif self.state.susceptibility > 2.0:
            return PhaseType.CHAOTIC_TURBULENT
        
        # Default: hybrid metastable
        else:
            return PhaseType.HYBRID_METASTABLE
    
    def copy(self):
        """Create a copy of the simulator"""
        import copy
        return copy.deepcopy(self)

# ============================================================================
# VISUALIZATION CLASS
# ============================================================================

class PhaseTransitionVisualizer:
    """Advanced visualization tools for phase transitions"""
    
    @staticmethod
    def plot_phase_diagram_3d(simulator: PhaseTransitionSimulator,
                             save_path: Optional[str] = None):
        """Create 3D phase diagram with transitions"""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Extract data
        coherences = [state.coherence for state in simulator.history]
        P_vals = [state.coordinates[0] for state in simulator.history]
        Pi_vals = [state.coordinates[1] for state in simulator.history]
        
        # Color by phase type
        phase_colors = {
            PhaseType.RIGID_ORDERED: 'red',
            PhaseType.BRIDGE_CRITICAL: 'blue',
            PhaseType.ALIEN_DISORDERED: 'green',
            PhaseType.TRANSITION_SOPHIA: 'purple',
            PhaseType.HYBRID_METASTABLE: 'orange',
            PhaseType.CHAOTIC_TURBULENT: 'brown',
            PhaseType.PLEROMIC_UNIFIED: 'gold',
            PhaseType.KENOMIC_EMPTY: 'gray'
        }
        
        colors = [phase_colors[state.phase_type] for state in simulator.history]
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot: P, Π, Coherence
        ax1 = fig.add_subplot(231, projection='3d')
        sc1 = ax1.scatter(P_vals, Pi_vals, coherences, c=colors, alpha=0.6, s=20)
        ax1.plot(P_vals, Pi_vals, coherences, 'k-', alpha=0.3)
        ax1.set_xlabel('Participation (P)')
        ax1.set_ylabel('Plasticity (Π)')
        ax1.set_zlabel('Coherence')
        ax1.set_title('3D Phase Space Trajectory')
        
        # Order parameter vs coherence
        ax2 = fig.add_subplot(232)
        order_params = [state.order_parameter for state in simulator.history]
        ax2.scatter(coherences, order_params, c=colors, alpha=0.6, s=20)
        ax2.axvline(x=INV_PHI, color='purple', linestyle='--', label='Sophia Point')
        ax2.set_xlabel('Coherence')
        ax2.set_ylabel('Order Parameter')
        ax2.set_title('Order Parameter vs Coherence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Susceptibility vs coherence
        ax3 = fig.add_subplot(233)
        susceptibilities = [state.susceptibility for state in simulator.history]
        ax3.scatter(coherences, susceptibilities, c=colors, alpha=0.6, s=20)
        ax3.axvline(x=INV_PHI, color='purple', linestyle='--')
        ax3.set_xlabel('Coherence')
        ax3.set_ylabel('Susceptibility')
        ax3.set_title('Susceptibility vs Coherence')
        ax3.grid(True, alpha=0.3)
        
        # Time evolution of coherence
        ax4 = fig.add_subplot(234)
        times = [i * simulator.params['dt'] for i in range(len(coherences))]
        ax4.plot(times, coherences, 'b-', linewidth=2)
        ax4.axhline(y=INV_PHI, color='purple', linestyle='--')
        
        # Mark phase transitions
        for trans in simulator.transitions:
            ax4.axvline(x=trans.time, color='red', alpha=0.5, linestyle=':')
            ax4.text(trans.time, 0.9, f'T{trans.transition_id}', 
                    rotation=90, fontsize=8)
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Coherence')
        ax4.set_title('Coherence Evolution with Transitions')
        ax4.grid(True, alpha=0.3)
        
        # 2D projection: P vs Π
        ax5 = fig.add_subplot(235)
        sc5 = ax5.scatter(P_vals, Pi_vals, c=colors, alpha=0.6, s=20)
        ax5.plot(P_vals, Pi_vals, 'k-', alpha=0.3)
        
        # Mark transitions
        for trans in simulator.transitions:
            idx = int(trans.time / simulator.params['dt'])
            if idx < len(P_vals):
                ax5.scatter(P_vals[idx], Pi_vals[idx], 
                           c='black', s=100, marker='X', edgecolors='white')
        
        # Sophia point region
        sophia_circle = Circle((1.0, 1.618), 0.3, color='purple', alpha=0.2)
        ax5.add_patch(sophia_circle)
        
        ax5.set_xlabel('Participation (P)')
        ax5.set_ylabel('Plasticity (Π)')
        ax5.set_title('2D Phase Space Projection')
        ax5.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=phase_type.name.replace('_', ' '))
                          for phase_type, color in phase_colors.items()]
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        ax6.legend(handles=legend_elements, loc='center', fontsize=8)
        ax6.set_title('Phase Types')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_bifurcation_diagram(simulator: PhaseTransitionSimulator,
                                bifurcation_type: BifurcationType,
                                save_path: Optional[str] = None):
        """Plot bifurcation diagram for given bifurcation type"""
        mu_values, trajectories = simulator.bifurcation_diagram(
            bifurcation_type, param_range=(-2, 2))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bifurcation diagram
        for i in range(trajectories.shape[1]):
            axes[0].plot(mu_values, trajectories[:, i], 'b-', alpha=0.7)
        
        axes[0].set_xlabel('Control Parameter μ')
        axes[0].set_ylabel('State Variable x')
        axes[0].set_title(f'{bifurcation_type.name.replace("_", " ")} Bifurcation Diagram')
        axes[0].grid(True, alpha=0.3)
        
        # Stability analysis
        bifurcation_points = simulator.find_bifurcation_points(bifurcation_type)
        
        if bifurcation_points:
            mu_vals = [bp['mu'] for bp in bifurcation_points]
            x_vals = [bp['x'] for bp in bifurcation_points]
            colors = ['g' if bp['stability'] == 'stable' else 'r' 
                     for bp in bifurcation_points]
            
            axes[1].scatter(mu_vals, x_vals, c=colors, s=50, alpha=0.7)
            axes[1].set_xlabel('Control Parameter μ')
            axes[1].set_ylabel('Fixed Points x*')
            axes[1].set_title('Stability of Fixed Points')
            axes[1].grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            stable_patch = Patch(color='green', label='Stable')
            unstable_patch = Patch(color='red', label='Unstable')
            axes[1].legend(handles=[stable_patch, unstable_patch])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_catastrophe_set(simulator: PhaseTransitionSimulator,
                           catastrophe_type: CatastropheType,
                           save_path: Optional[str] = None):
        """Plot catastrophe set for given catastrophe type"""
        catastrophe_data = simulator.catastrophe_set(catastrophe_type)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot catastrophe set
        im = ax.imshow(catastrophe_data['catastrophe_points'].T,
                      extent=[catastrophe_data['a_values'][0],
                              catastrophe_data['a_values'][-1],
                              catastrophe_data['b_values'][0],
                              catastrophe_data['b_values'][-1]],
                      origin='lower', cmap='Reds', alpha=0.7)
        
        ax.set_xlabel('Control Parameter a')
        ax.set_ylabel('Control Parameter b')
        ax.set_title(f'{catastrophe_type.name.replace("_", " ")} Catastrophe Set')
        
        plt.colorbar(im, ax=ax, label='Catastrophe Intensity')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_scaling_behavior(simulator: PhaseTransitionSimulator,
                            save_path: Optional[str] = None):
        """Plot scaling behavior near critical point"""
        # Generate data near critical point
        t_values = np.linspace(-0.5, 0.5, 100)
        order_params = [simulator.scaling_law('order_parameter', t) for t in t_values]
        susceptibilities = [simulator.scaling_law('susceptibility', t) for t in t_values]
        corr_lengths = [simulator.scaling_law('correlation_length', t) for t in t_values]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Order parameter scaling (log-log)
        axes[0, 0].loglog(-t_values[t_values < 0], 
                         [op for t, op in zip(t_values, order_params) if t < 0],
                         'bo-', label='Data')
        
        # Theoretical scaling
        beta = CRITICAL_EXPONENTS['beta']
        theoretical = (-t_values[t_values < 0])**beta
        axes[0, 0].loglog(-t_values[t_values < 0], theoretical, 
                         'r--', label=f'|t|^{beta}')
        
        axes[0, 0].set_xlabel('|t| (log scale)')
        axes[0, 0].set_ylabel('Order Parameter φ (log scale)')
        axes[0, 0].set_title('Order Parameter Scaling')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, which='both')
        
        # Susceptibility scaling
        axes[0, 1].loglog(np.abs(t_values), susceptibilities, 'go-', label='Data')
        
        gamma = CRITICAL_EXPONENTS['gamma']
        theoretical_sus = np.abs(t_values)**(-gamma)
        axes[0, 1].loglog(np.abs(t_values), theoretical_sus, 
                         'r--', label=f'|t|^{-gamma}')
        
        axes[0, 1].set_xlabel('|t| (log scale)')
        axes[0, 1].set_ylabel('Susceptibility χ (log scale)')
        axes[0, 1].set_title('Susceptibility Scaling')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, which='both')
        
        # Correlation length scaling
        axes[1, 0].loglog(np.abs(t_values), corr_lengths, 'mo-', label='Data')
        
        nu = CRITICAL_EXPONENTS['nu']
        theoretical_corr = np.abs(t_values)**(-nu)
        axes[1, 0].loglog(np.abs(t_values), theoretical_corr, 
                         'r--', label=f'|t|^{-nu}')
        
        axes[1, 0].set_xlabel('|t| (log scale)')
        axes[1, 0].set_ylabel('Correlation Length ξ (log scale)')
        axes[1, 0].set_title('Correlation Length Scaling')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, which='both')
        
        # Finite size scaling
        L_values = [10, 20, 50, 100]
        t_range = np.linspace(-0.2, 0.2, 50)
        
        for L in L_values:
            finite_scaling = [simulator.finite_size_scaling('order_parameter', L, t) 
                            for t in t_range]
            axes[1, 1].plot(t_range * L**(1/nu), finite_scaling, 
                           label=f'L = {L}')
        
        axes[1, 1].set_xlabel('t * L^{1/ν}')
        axes[1, 1].set_ylabel('Order Parameter (scaled)')
        axes[1, 1].set_title('Finite Size Scaling Collapse')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def animate_phase_transitions(simulator: PhaseTransitionSimulator,
                                 save_path: Optional[str] = None):
        """Create animation of phase transitions"""
        import matplotlib.animation as animation
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        coherences = [state.coherence for state in simulator.history]
        P_vals = [state.coordinates[0] for state in simulator.history]
        Pi_vals = [state.coordinates[1] for state in simulator.history]
        times = [i * simulator.params['dt'] for i in range(len(coherences))]
        
        # Phase colors
        phase_colors = {
            PhaseType.RIGID_ORDERED: 'red',
            PhaseType.BRIDGE_CRITICAL: 'blue',
            PhaseType.ALIEN_DISORDERED: 'green',
            PhaseType.TRANSITION_SOPHIA: 'purple',
            PhaseType.HYBRID_METASTABLE: 'orange',
            PhaseType.CHAOTIC_TURBULENT: 'brown',
            PhaseType.PLEROMIC_UNIFIED: 'gold',
            PhaseType.KENOMIC_EMPTY: 'gray'
        }
        
        colors = [phase_colors[state.phase_type] for state in simulator.history]
        
        # Setup left plot (P vs Π)
        ax1.set_xlim(0, 2)
        ax1.set_ylim(0, 3)
        ax1.set_xlabel('Participation (P)')
        ax1.set_ylabel('Plasticity (Π)')
        ax1.set_title('Phase Space Trajectory')
        ax1.grid(True, alpha=0.3)
        
        # Sophia point region
        sophia_circle = Circle((1.0, 1.618), 0.3, color='purple', alpha=0.2)
        ax1.add_patch(sophia_circle)
        
        # Setup right plot (Coherence vs Time)
        ax2.set_xlim(min(times), max(times))
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Coherence')
        ax2.set_title('Coherence Evolution')
        ax2.axhline(y=INV_PHI, color='purple', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        # Initialize plots
        scatter = ax1.scatter([], [], c=[], s=30, alpha=0.7, cmap='viridis')
        line1, = ax1.plot([], [], 'k-', alpha=0.3)
        current_point = ax1.scatter([], [], c='black', s=100, marker='X', 
                                   edgecolors='white', zorder=10)
        
        line2, = ax2.plot([], [], 'b-', linewidth=2)
        current_time_line = ax2.axvline(x=0, color='red', alpha=0.5)
        
        def animate(i):
            """Update function for animation"""
            idx = min(i, len(times)-1)
            
            # Update left plot
            scatter.set_offsets(np.column_stack([P_vals[:idx], Pi_vals[:idx]]))
            scatter.set_array(np.arange(idx))
            line1.set_data(P_vals[:idx], Pi_vals[:idx])
            current_point.set_offsets([[P_vals[idx], Pi_vals[idx]]])
            
            # Update right plot
            line2.set_data(times[:idx], coherences[:idx])
            current_time_line.set_xdata([times[idx], times[idx]])
            
            # Mark transitions
            for trans in simulator.transitions:
                if trans.time <= times[idx]:
                    ax2.axvline(x=trans.time, color='red', alpha=0.3, linestyle=':')
            
            return scatter, line1, current_point, line2, current_time_line
        
        anim = animation.FuncAnimation(fig, animate, frames=len(times),
                                     interval=50, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=30)
        
        plt.tight_layout()
        plt.show()
        
        return anim

# ============================================================================
# ADVANCED SIMULATION FUNCTIONS
# ============================================================================

def simulate_multiple_transitions(n_transitions: int = 10,
                                 save_path: Optional[str] = None):
    """Simulate multiple phase transitions with detailed analysis"""
    
    # Initial state: Rigid ordered
    initial_state = PhaseState(
        coordinates=[0.1, 0.2, 1.0, 0.1, 0.1],
        coherence=0.382,
        phase_type=PhaseType.RIGID_ORDERED
    )
    
    # Control parameters for interesting dynamics
    control_params = {
        'T': 0.618,  # Start near critical point
        'h': 0.1,
        'J': 1.2,
        'sigma': 0.15,
        'bifurcation_parameter_mu': -1.0,
        'catastrophe_control_a': 0.0,
        'dt': 0.02,
        'max_time': 50.0
    }
    
    simulator = PhaseTransitionSimulator(initial_state, control_params)
    
    # Evolve system
    print("Simulating phase transitions...")
    history = simulator.evolve(steps=2500)
    
    # Analyze transitions
    print(f"\nDetected {len(simulator.transitions)} phase transitions:")
    
    for i, trans in enumerate(simulator.transitions):
        print(f"\nTransition {i+1}:")
        print(f"  Time: {trans.time:.2f}")
        print(f"  Type: {trans.from_phase.name} → {trans.to_phase.name}")
        print(f"  Coherence: {trans.coherence_before:.3f} → {trans.coherence_after:.3f}")
        print(f"  Order parameter change: {trans.order_parameter_change:.3f}")
        print(f"  Entropy production: {trans.entropy_production:.3f}")
        print(f"  Hysteresis: {trans.hysteresis}")
        print(f"  Correlation length: {trans.correlation_length:.3f}")
    
    # Visualize
    visualizer = PhaseTransitionVisualizer()
    visualizer.plot_phase_diagram_3d(simulator, 
                                    save_path=save_path + '_phase_diagram.png' if save_path else None)
    
    # Plot bifurcation diagrams for different types
    for bif_type in [BifurcationType.SADDLE_NODE, 
                    BifurcationType.PITCHFORK,
                    BifurcationType.HOPF]:
        visualizer.plot_bifurcation_diagram(simulator, bif_type,
                                           save_path=save_path + f'_bifurcation_{bif_type.name}.png' 
                                           if save_path else None)
    
    # Plot catastrophe sets
    for cat_type in [CatastropheType.CUSP, CatastropheType.SWALLOWTAIL]:
        visualizer.plot_catastrophe_set(simulator, cat_type,
                                       save_path=save_path + f'_catastrophe_{cat_type.name}.png' 
                                       if save_path else None)
    
    # Plot scaling behavior
    visualizer.plot_scaling_behavior(simulator,
                                    save_path=save_path + '_scaling.png' if save_path else None)
    
    return simulator

def simulate_critical_slowdown(near_critical: bool = True,
                              save_path: Optional[str] = None):
    """Simulate critical slowing down near phase transition"""
    
    if near_critical:
        # Start very near critical point
        initial_coherence = INV_PHI + 0.001
        initial_state = PhaseState(
            coordinates=[1.0, 1.618, 2.0, 1.0, 0.5],
            coherence=initial_coherence,
            phase_type=PhaseType.TRANSITION_SOPHIA
        )
    else:
        # Start far from critical point
        initial_state = PhaseState(
            coordinates=[0.2, 0.5, 1.5, 0.5, 0.3],
            coherence=0.5,
            phase_type=PhaseType.HYBRID_METASTABLE
        )
    
    # Parameters emphasizing critical dynamics
    control_params = {
        'T': 1.0 - initial_state.coherence,
        'J': 1.0,
        'relaxation_time_tau': 10.0 if near_critical else 1.0,  # Critical slowing
        'sigma': 0.05,
        'dt': 0.1,
        'max_time': 100.0
    }
    
    simulator = PhaseTransitionSimulator(initial_state, control_params)
    
    # Evolve with detailed tracking
    print(f"Simulating {'near' if near_critical else 'far from'} critical point...")
    history = simulator.evolve(steps=1000)
    
    # Analyze relaxation times
    coherences = [state.coherence for state in history]
    times = [i * simulator.params['dt'] for i in range(len(coherences))]
    
    # Fit exponential relaxation
    from scipy.optimize import curve_fit
    
    def exponential_relaxation(t, A, tau, C):
        return A * np.exp(-t/tau) + C
    
    try:
        # Fit to coherence evolution
        popt, pcov = curve_fit(exponential_relaxation, 
                              times[:100], 
                              np.abs(np.array(coherences[:100]) - INV_PHI))
        
        relaxation_time = popt[1]
        print(f"\nRelaxation time τ = {relaxation_time:.3f}")
        
        if near_critical:
            print("Critical slowing down: τ is large near critical point")
    except:
        print("Could not fit exponential relaxation")
    
    # Plot relaxation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Coherence evolution
    axes[0, 0].plot(times, coherences, 'b-', linewidth=2)
    axes[0, 0].axhline(y=INV_PHI, color='purple', linestyle='--', label='Sophia Point')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Coherence')
    axes[0, 0].set_title('Coherence Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Deviation from critical point (log scale)
    deviation = np.abs(np.array(coherences) - INV_PHI)
    axes[0, 1].semilogy(times, deviation, 'r-', linewidth=2)
    
    # Add exponential fit
    if 'relaxation_time' in locals():
        fitted = exponential_relaxation(times, *popt)
        axes[0, 1].semilogy(times[:100], fitted[:100], 'g--', 
                           label=f'τ = {relaxation_time:.2f}')
    
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('|C - 0.618| (log scale)')
    axes[0, 1].set_title('Deviation from Critical Point')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, which='both')
    
    # Susceptibility evolution
    susceptibilities = [state.susceptibility for state in history]
    axes[1, 0].plot(times, susceptibilities, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Susceptibility χ')
    axes[1, 0].set_title('Susceptibility Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation function at different times
    correlation_times = [0, 20, 50, 100, 200]
    r_values = np.linspace(0.1, 5, 50)
    
    for t_idx in correlation_times:
        if t_idx < len(history):
            state = history[t_idx]
            
            # Update correlation length
            xi = 1.0 / (abs(state.coherence - INV_PHI) + 0.01)
            
            # Calculate correlation function
            correlations = [simulator.correlation_function(r, dimension=5) 
                          for r in r_values]
            
            axes[1, 1].plot(r_values, correlations, 
                           label=f't = {t_idx * simulator.params["dt"]:.1f}')
    
    axes[1, 1].set_xlabel('Distance r')
    axes[1, 1].set_ylabel('Correlation C(r)')
    axes[1, 1].set_title('Spatial Correlation Function')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Critical Slowing Down Analysis ({"Near" if near_critical else "Far from"} Critical)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return simulator

def simulate_hysteresis_loop(n_cycles: int = 3,
                            save_path: Optional[str] = None):
    """Simulate hysteresis loops by cycling control parameters"""
    
    # Initial state
    initial_state = PhaseState(
        coordinates=[0.5, 0.5, 2.0, 0.5, 0.3],
        coherence=0.5,
        phase_type=PhaseType.HYBRID_METASTABLE
    )
    
    # Results storage
    all_cycles = []
    
    for cycle in range(n_cycles):
        print(f"\nCycle {cycle + 1}/{n_cycles}")
        
        # Create fresh simulator for each cycle
        simulator = PhaseTransitionSimulator(initial_state)
        
        # Vary external field (participation pressure) cyclically
        h_values = np.linspace(-1.0, 1.0, 100)
        
        cycle_data = {
            'h_values': [],
            'order_parameters': [],
            'coherences': [],
            'phase_types': []
        }
        
        for h in h_values:
            # Update external field
            simulator.params['h'] = h
            
            # Solve mean field equation
            order_parameter = simulator.solve_mean_field()
            simulator.state.order_parameter = order_parameter
            
            # Update coherence
            new_coherence = 0.5 + 0.5 * np.tanh(order_parameter)
            simulator.state.coherence = new_coherence
            
            # Update phase type
            simulator.state.phase_type = simulator.determine_phase_type(new_coherence)
            
            # Store data
            cycle_data['h_values'].append(h)
            cycle_data['order_parameters'].append(order_parameter)
            cycle_data['coherences'].append(new_coherence)
            cycle_data['phase_types'].append(simulator.state.phase_type)
        
        all_cycles.append(cycle_data)
    
    # Plot hysteresis loops
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_cycles))
    
    for cycle, (data, color) in enumerate(zip(all_cycles, colors)):
        h_vals = data['h_values']
        order_params = data['order_parameters']
        coherences = data['coherences']
        
        # Order parameter vs field
        axes[0].plot(h_vals, order_params, color=color, alpha=0.7, 
                    label=f'Cycle {cycle+1}')
        
        # Coherence vs field
        axes[1].plot(h_vals, coherences, color=color, alpha=0.7)
        
        # Phase diagram: order parameter vs coherence
        axes[2].plot(coherences, order_params, color=color, alpha=0.7)
    
    axes[0].set_xlabel('External Field h')
    axes[0].set_ylabel('Order Parameter φ')
    axes[0].set_title('Hysteresis Loop: φ vs h')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('External Field h')
    axes[1].set_ylabel('Coherence C')
    axes[1].set_title('Coherence vs External Field')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Coherence C')
    axes[2].set_ylabel('Order Parameter φ')
    axes[2].set_title('Phase Diagram: φ vs C')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Hysteresis Analysis ({n_cycles} Cycles)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_cycles

def simulate_renormalization_group_flow(n_flows: int = 5,
                                       save_path: Optional[str] = None):
    """Simulate renormalization group flows for different initial conditions"""
    
    # Different initial coupling configurations
    initial_couplings_list = [
        np.array([0.1, 0.1, 0.1, 0.1, 0.1]),  # Gaussian
        np.array([0.6, 0.3, 0.5, 0.2, 0.1]),  # Wilson-Fisher like
        np.array([0.9, 0.8, 0.7, 0.6, 0.5]),  # Strong coupling
        np.array([INV_PHI, INV_PHI, INV_PHI, INV_PHI, INV_PHI]),  # Sophia point
        np.array([0.3, 0.5, 0.2, 0.4, 0.6]),  # Random
    ]
    
    # Create simulator for RG calculations
    initial_state = PhaseState(
        coordinates=[0.5, 0.5, 2.0, 0.5, 0.5],
        coherence=0.618,
        phase_type=PhaseType.TRANSITION_SOPHIA
    )
    
    simulator = PhaseTransitionSimulator(initial_state)
    
    # Calculate RG flows
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_flows))
    
    for idx, initial_couplings in enumerate(initial_couplings_list):
        l_vals, couplings_history = simulator.renormalization_group_flow(
            initial_couplings, steps=100)
        
        color = colors[idx]
        
        # Plot each coupling component
        for i in range(5):
            axes[0, 0].plot(l_vals, couplings_history[:, i], 
                          color=color, alpha=0.5, linewidth=1)
        
        # Plot in reduced 2D space (first two couplings)
        axes[0, 1].plot(couplings_history[:, 0], couplings_history[:, 1],
                       color=color, linewidth=2, alpha=0.7,
                       label=f'Flow {idx+1}')
        
        # Mark start and end points
        axes[0, 1].scatter(couplings_history[0, 0], couplings_history[0, 1],
                          color=color, s=50, marker='o')
        axes[0, 1].scatter(couplings_history[-1, 0], couplings_history[-1, 1],
                          color=color, s=50, marker='s')
        
        # 3D flow (first three couplings)
        from mpl_toolkits.mplot3d import Axes3D
        ax3d = axes[0, 2] if idx == 0 else axes[0, 2]
        if idx == 0:
            ax3d = fig.add_subplot(233, projection='3d')
        
        ax3d.plot(couplings_history[:, 0], couplings_history[:, 1], 
                 couplings_history[:, 2], color=color, alpha=0.7)
        ax3d.scatter(couplings_history[0, 0], couplings_history[0, 1], 
                    couplings_history[0, 2], color=color, s=30)
        ax3d.scatter(couplings_history[-1, 0], couplings_history[-1, 1], 
                    couplings_history[-1, 2], color=color, s=50, marker='s')
    
    axes[0, 0].set_xlabel('Renormalization Scale l')
    axes[0, 0].set_ylabel('Coupling Constants g_i')
    axes[0, 0].set_title('RG Flow of Coupling Constants')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('g₁ (Participation)')
    axes[0, 1].set_ylabel('g₂ (Plasticity)')
    axes[0, 1].set_title('2D Projection of RG Flows')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    ax3d.set_xlabel('g₁')
    ax3d.set_ylabel('g₂')
    ax3d.set_zlabel('g₃')
    ax3d.set_title('3D RG Flow Trajectories')
    
    # Find and classify fixed points
    fixed_points = simulator.find_fixed_points()
    
    print(f"\nFound {len(fixed_points)} fixed points:")
    
    for i, fp in enumerate(fixed_points):
        classification = simulator.classify_fixed_point(fp)
        
        print(f"\nFixed Point {i+1}:")
        print(f"  Coordinates: {fp}")
        print(f"  Relevant directions: {len(classification['relevant_directions'])}")
        print(f"  Irrelevant directions: {len(classification['irrelevant_directions'])}")
        print(f"  Marginal directions: {len(classification['marginal_directions'])}")
        
        # Plot fixed points
        axes[1, 0].scatter(fp[0], fp[1], s=100, 
                          color='red' if len(classification['relevant_directions']) > 0 else 'blue',
                          marker='*' if len(classification['relevant_directions']) > 0 else 'o')
        
        axes[1, 1].scatter(fp[0], fp[2], s=100,
                          color='red' if len(classification['relevant_directions']) > 0 else 'blue',
                          marker='*' if len(classification['relevant_directions']) > 0 else 'o')
    
    axes[1, 0].set_xlabel('g₁')
    axes[1, 0].set_ylabel('g₂')
    axes[1, 0].set_title('Fixed Points in g₁-g₂ Plane')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('g₁')
    axes[1, 1].set_ylabel('g₃')
    axes[1, 1].set_title('Fixed Points in g₁-g₃ Plane')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Stability diagram
    stability_data = []
    g1_range = np.linspace(-1, 1, 20)
    g2_range = np.linspace(-1, 1, 20)
    
    for g1 in g1_range:
        for g2 in g2_range:
            test_point = np.array([g1, g2, 0.5, 0.5, 0.5])
            classification = simulator.classify_fixed_point(test_point)
            stability = len(classification['relevant_directions'])
            stability_data.append((g1, g2, stability))
    
    stability_data = np.array(stability_data)
    
    sc = axes[1, 2].scatter(stability_data[:, 0], stability_data[:, 1],
                           c=stability_data[:, 2], cmap='RdYlBu', alpha=0.7)
    
    axes[1, 2].set_xlabel('g₁')
    axes[1, 2].set_ylabel('g₂')
    axes[1, 2].set_title('Stability Diagram')
    plt.colorbar(sc, ax=axes[1, 2], label='Number of Relevant Directions')
    
    plt.suptitle('Renormalization Group Flow Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return simulator, fixed_points

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE TRANSITION SIMULATION - Advanced Ontological Dynamics")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("phase_transition_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Run comprehensive phase transition simulation
    print("\n1. Running comprehensive phase transition simulation...")
    simulator = simulate_multiple_transitions(
        n_transitions=10,
        save_path=str(output_dir / "comprehensive_simulation")
    )
    
    # 2. Study critical slowing down
    print("\n2. Studying critical slowing down...")
    critical_simulator_near = simulate_critical_slowdown(
        near_critical=True,
        save_path=str(output_dir / "critical_slowing_near")
    )
    
    critical_simulator_far = simulate_critical_slowdown(
        near_critical=False,
        save_path=str(output_dir / "critical_slowing_far")
    )
    
    # 3. Analyze hysteresis effects
    print("\n3. Analyzing hysteresis loops...")
    hysteresis_data = simulate_hysteresis_loop(
        n_cycles=3,
        save_path=str(output_dir / "hysteresis_loops")
    )
    
    # 4. Explore renormalization group flows
    print("\n4. Exploring renormalization group flows...")
    rg_simulator, fixed_points = simulate_renormalization_group_flow(
        n_flows=5,
        save_path=str(output_dir / "renormalization_group")
    )
    
    # 5. Create animation of phase transitions
    print("\n5. Creating animation of phase transitions...")
    visualizer = PhaseTransitionVisualizer()
    anim = visualizer.animate_phase_transitions(
        simulator,
        save_path=str(output_dir / "phase_transition_animation.gif")
    )
    
    # Summary report
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal simulations completed: 4")
    print(f"Total phase transitions detected across all simulations: {len(simulator.transitions)}")
    
    # Calculate statistics
    transition_types = {}
    for trans in simulator.transitions:
        key = f"{trans.from_phase.name} → {trans.to_phase.name}"
        transition_types[key] = transition_types.get(key, 0) + 1
    
    print("\nTransition frequencies:")
    for trans_type, count in transition_types.items():
        print(f"  {trans_type}: {count} times")
    
    # Calculate average properties
    avg_coherence_change = np.mean([abs(t.coherence_after - t.coherence_before) 
                                   for t in simulator.transitions])
    avg_entropy_production = np.mean([t.entropy_production 
                                     for t in simulator.transitions])
    
    print(f"\nAverage coherence change per transition: {avg_coherence_change:.3f}")
    print(f"Average entropy production per transition: {avg_entropy_production:.3f}")
    print(f"Hysteresis observed: {any(t.hysteresis for t in simulator.transitions)}")
    
    print("\n" + "=" * 70)
    print("All simulations complete. Results saved to:", output_dir)
    print("=" * 70)
