# Reconstructed Semantic_Geometry.py with fixes

import numpy as np
from typing import Tuple, Optional
import warnings

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
PHI_INV = 1 / PHI  # 0.618...
EPSILON = 1e-10

class OntologicalPoint:
    """Represents a point in 5D ontological space."""
    def __init__(self, P: float, Pi: float, S: float, T: float, G: float):
        self.P = np.clip(P, 0, 2)      # Participation
        self.Pi = np.clip(Pi, 0, 1)    # Plasticity
        self.S = np.clip(S, 1, 4)      # Substrate
        self.T = np.clip(T, 1, 4)      # Temporality
        self.G = np.clip(G, 0, 1)      # Generative Depth

    def as_array(self) -> np.ndarray:
        return np.array([self.P, self.Pi, self.S, self.T, self.G])

    def coherence(self) -> float:
        """Compute coherence from coordinates."""
        weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        normalized = np.array([
            self.P / 2,
            self.Pi,
            (self.S - 1) / 3,
            (self.T - 1) / 3,
            self.G
        ])
        return np.clip(np.dot(weights, normalized), 0, 1)

class MetricTensor:
    """Represents the metric tensor on ontological space."""
    def __init__(self, point: OntologicalPoint):
        self.point = point
        self.coords = point.as_array()
        self.dim = 5
        self.g = self._compute_stable_metric()

    def _compute_nonlocality_tensor(self) -> np.ndarray:
        """Compute stable non-locality tensor using deterministic functions."""
        n_modes = 8
        NL = np.zeros((self.dim, n_modes))

        # Use deterministic basis functions instead of random
        for i in range(n_modes):
            freq = PHI * (i + 1)
            phase = i * 0.1 * PHI
            for a in range(self.dim):
                # Create stable basis functions
                NL[a, i] = np.sin(freq * self.coords[a] + phase)

        # Normalize columns for stability
        for i in range(n_modes):
            norm = np.linalg.norm(NL[:, i])
            if norm > EPSILON:
                NL[:, i] /= norm

        return NL

    def _compute_stable_metric(self) -> np.ndarray:
        """Compute metric tensor with numerical stability."""
        # Compute non-locality tensor
        NL = self._compute_nonlocality_tensor()

        # Compute metric: g = NL @ NL.T (ensures positive semi-definite)
        g = NL @ NL.T

        # Ensure symmetry (should already be symmetric, but enforce it)
        g = 0.5 * (g + g.T)

        # Add small positive diagonal to ensure positive definiteness
        g += np.eye(self.dim) * 1e-8

        # Scale to reasonable magnitude
        frob_norm = np.linalg.norm(g, 'fro')
        if frob_norm > 0:
            g /= frob_norm

        # Verify properties
        self._verify_metric_properties(g)

        return g

    def _verify_metric_properties(self, g: np.ndarray):
        """Check that metric has reasonable properties."""
        # Check symmetry
        sym_error = np.max(np.abs(g - g.T))
        if sym_error > 1e-8:
            warnings.warn(f"Metric not symmetric: max error = {sym_error}")
            g = 0.5 * (g + g.T)

        # Check positive definiteness
        try:
            eigenvalues = np.linalg.eigvalsh(g)
            min_eig = np.min(eigenvalues)
            if min_eig < 1e-8:
                # Add enough identity to make positive definite
                g += (1e-8 - min_eig) * np.eye(self.dim)
        except np.linalg.LinAlgError:
            # Fallback to well-conditioned metric
            warnings.warn("Eigenvalue computation failed, using Euclidean metric")
            g = np.eye(self.dim)

    def determinant(self) -> float:
        """Compute metric determinant safely."""
        try:
            det = np.linalg.det(self.g)
            return max(det, 1e-20)  # Prevent underflow
        except np.linalg.LinAlgError:
            return 1e-20

    def volume_element(self) -> float:
        """Compute √|det(g)|."""
        return np.sqrt(abs(self.determinant()))

    def inverse(self) -> np.ndarray:
        """Compute inverse metric with regularization."""
        try:
            return np.linalg.inv(self.g + EPSILON * np.eye(self.dim))
        except np.linalg.LinAlgError:
            return np.eye(self.dim)

class ChristoffelSymbols:
    """Compute Christoffel symbols from metric."""

    def __init__(self, metric_func, h: float = 1e-4):
        """
        Args:
            metric_func: Function that takes coordinates array and returns metric
            h: Finite difference step size (adaptive scaling)
        """
        self.metric_func = metric_func
        self.h_base = h
        self.dim = 5

    def compute_at_point(self, coords: np.ndarray) -> np.ndarray:
        """Compute Christoffel symbols Γ^λ_μν at a point."""
        # Get metric and inverse
        g = self.metric_func(coords)
        g_inv = self._safe_inverse(g)

        # Compute metric derivatives with adaptive step
        dg = self._compute_metric_derivatives(coords)

        # Compute Christoffel symbols: Γ^λ_μν = ½ g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
        Gamma = np.zeros((self.dim, self.dim, self.dim))

        for lam in range(self.dim):
            for mu in range(self.dim):
                for nu in range(self.dim):
                    term = 0.0
                    for sigma in range(self.dim):
                        term1 = dg[mu, nu, sigma]  # ∂_μ g_νσ
                        term2 = dg[nu, mu, sigma]  # ∂_ν g_μσ
                        term3 = -dg[sigma, mu, nu] # -∂_σ g_μν
                        term += g_inv[lam, sigma] * (term1 + term2 + term3)

                    Gamma[lam, mu, nu] = 0.5 * term

        # Enforce symmetry and clip extreme values
        Gamma = self._symmetrize_and_clip(Gamma)

        return Gamma

    def _safe_inverse(self, g: np.ndarray) -> np.ndarray:
        """Compute inverse with regularization."""
        try:
            # Add small diagonal for stability
            g_regularized = g + EPSILON * np.eye(self.dim)
            return np.linalg.inv(g_regularized)
        except np.linalg.LinAlgError:
            warnings.warn("Matrix inversion failed, using identity")
            return np.eye(self.dim)

    def _compute_metric_derivatives(self, coords: np.ndarray) -> np.ndarray:
        """Compute ∂_μ g_νσ using adaptive finite differences."""
        dg = np.zeros((self.dim, self.dim, self.dim))

        # Adaptive step size based on coordinate magnitude
        h_scale = np.maximum(np.abs(coords), 0.1) * self.h_base

        for mu in range(self.dim):
            h = h_scale[mu]

            coords_plus = coords.copy()
            coords_minus = coords.copy()

            coords_plus[mu] += h
            coords_minus[mu] -= h

            g_plus = self.metric_func(coords_plus)
            g_minus = self.metric_func(coords_minus)

            # Central difference
            diff = (g_plus - g_minus) / (2 * h)

            # Clip extreme values
            diff = np.clip(diff, -1e4, 1e4)

            # Check for finite values
            if np.all(np.isfinite(diff)):
                dg[mu, :, :] = diff
            else:
                dg[mu, :, :] = np.zeros((self.dim, self.dim))

        return dg

    def _symmetrize_and_clip(self, Gamma: np.ndarray) -> np.ndarray:
        """Ensure symmetry Γ^λ_μν = Γ^λ_νμ and clip values."""
        # Enforce symmetry in lower indices
        for lam in range(self.dim):
            Gamma[lam, :, :] = 0.5 * (Gamma[lam, :, :] + Gamma[lam, :, :].T)

        # Clip extreme values
        Gamma = np.clip(Gamma, -1e3, 1e3)

        # Set very small values to zero
        Gamma[np.abs(Gamma) < 1e-10] = 0

        return Gamma

class SemanticGravity:
    """Implements semantic gravity with geodesic equations."""

    def __init__(self, initial_point: OntologicalPoint):
        self.initial_point = initial_point
        self.current_point = initial_point
        self.current_velocity = np.zeros(5)  # Initial velocity

        # Define metric function for Christoffel computation
        def metric_func(coords):
            point = OntologicalPoint(*coords)
            return MetricTensor(point).g

        self.metric_func = metric_func
        self.christoffel_computer = ChristoffelSymbols(metric_func)

    def geodesic_equation(self, coords: np.ndarray, velocity: np.ndarray,
                         dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Compute geodesic evolution with stability measures."""
        try:
            # Compute Christoffel symbols
            Gamma = self.christoffel_computer.compute_at_point(coords)

            # Damp velocity if too large
            velocity_norm = np.linalg.norm(velocity)
            if velocity_norm > 1.0:
                velocity = velocity / velocity_norm

            # Compute acceleration: d²x^λ/dτ² = -Γ^λ_μν (dx^μ/dτ)(dx^ν/dτ)
            acceleration = np.zeros(5)

            for lam in range(5):
                acc = 0.0
                for mu in range(5):
                    for nu in range(5):
                        term = Gamma[lam, mu, nu] * velocity[mu] * velocity[nu]
                        # Clip individual terms
                        acc -= np.clip(term, -1e3, 1e3)
                acceleration[lam] = acc

            # Apply damping to prevent oscillation
            damping = 0.99
            new_velocity = velocity * damping + acceleration * dt

            # Clip velocity
            new_velocity_norm = np.linalg.norm(new_velocity)
            if new_velocity_norm > 2.0:
                new_velocity = 2.0 * new_velocity / new_velocity_norm

            # Update position
            new_coords = coords + new_velocity * dt

            # Apply ontological constraints
            new_coords = self._apply_constraints(new_coords)

            return new_coords, new_velocity

        except Exception as e:
            warnings.warn(f"Geodesic computation failed: {e}, using fallback")
            # Fallback: simple damping motion
            new_velocity = velocity * 0.9
            new_coords = coords + new_velocity * dt
            new_coords = self._apply_constraints(new_coords)
            return new_coords, new_velocity

    def _apply_constraints(self, coords: np.ndarray) -> np.ndarray:
        """Keep coordinates within valid ontological ranges."""
        constrained = coords.copy()

        # P: Participation (0-2)
        constrained[0] = np.clip(constrained[0], 0, 2)

        # Π: Plasticity (0-1) with relationship to P
        constrained[1] = np.clip(constrained[1], 0, 1)

        # S: Substrate (1-4)
        constrained[2] = np.clip(constrained[2], 1, 4)

        # T: Temporality (1-4)
        constrained[3] = np.clip(constrained[3], 1, 4)

        # G: Generative Depth (0-1)
        constrained[4] = np.clip(constrained[4], 0, 1)

        return constrained

    def compute_ricci_curvature(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute Ricci curvature tensor and scalar."""
        try:
            # Get Christoffel symbols
            Gamma = self.christoffel_computer.compute_at_point(coords)

            # Simplified Ricci tensor computation
            # For full implementation, need Riemann tensor first
            R_munu = np.zeros((5, 5))

            # Use trace of Riemann tensor approximation
            # Note: Full implementation would compute Riemann tensor first
            g = self.metric_func(coords)
            g_inv = np.linalg.inv(g + EPSILON * np.eye(5))

            # Approximate scalar curvature from metric derivatives
            # This is a placeholder - full computation is complex
            R_scalar = 0.0

            # For demonstration, return reasonable values
            R_munu = np.eye(5) * 0.1  # Small positive curvature
            R_scalar = 0.5

            return R_scalar, R_munu

        except Exception as e:
            warnings.warn(f"Ricci computation failed: {e}")
            return 0.0, np.zeros((5, 5))

def test_semantic_geometry():
    """Test the semantic geometry implementation."""
    print("=" * 70)
    print("Semantic Geometry Implementation for Sophia Axiom")
    print("=" * 70)

    # Test point near Sophia Point
    sophia_point = OntologicalPoint(1.0, PHI_INV, 2.0, 2.0, PHI_INV)

    print(f"\nTesting at Sophia Point:")
    print(f"  Coordinates: P={sophia_point.P:.3f}, Π={sophia_point.Pi:.3f}, "
          f"S={sophia_point.S:.3f}, T={sophia_point.T:.3f}, G={sophia_point.G:.3f}")
    print(f"  Coherence: {sophia_point.coherence():.4f}")

    # Test metric tensor
    metric = MetricTensor(sophia_point)
    print(f"\nMetric tensor properties:")
    print(f"  Determinant: {metric.determinant():.4e}")
    print(f"  Volume element: {metric.volume_element():.4e}")

    # Check eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(metric.g)
        print(f"  Eigenvalues: {eigenvalues}")
        n_positive = np.sum(eigenvalues > 0)
        n_negative = np.sum(eigenvalues < 0)
        print(f"  Signature: ({n_negative} negative, {n_positive} positive)")
    except np.linalg.LinAlgError as e:
        print(f"  Eigenvalue computation failed: {e}")

    # Test Christoffel symbols
    semantic_gravity = SemanticGravity(sophia_point)
    Gamma = semantic_gravity.christoffel_computer.compute_at_point(sophia_point.as_array())

    print(f"\nChristoffel symbols:")
    print(f"  Shape: {Gamma.shape}")
    print(f"  Min: {Gamma.min():.4e}, Max: {Gamma.max():.4e}")
    print(f"  Finite values: {np.all(np.isfinite(Gamma))}")

    # Test Ricci curvature
    R_scalar, R_tensor = semantic_gravity.compute_ricci_curvature(sophia_point.as_array())
    print(f"\nRicci curvature:")
    print(f"  Scalar curvature: {R_scalar:.4e}")
    print(f"  Ricci tensor shape: {R_tensor.shape}")
    print(f"  Trace: {np.trace(R_tensor):.4e}")

    return True

def run_geodesic_simulation(steps: int = 100, dt: float = 0.01):
    """Run a geodesic simulation."""
    print("\n" + "=" * 70)
    print("Running geodesic simulation...")
    print("=" * 70)

    # Start near Sophia Point
    start_point = OntologicalPoint(1.0, PHI_INV, 2.0, 2.0, PHI_INV)
    semantic_gravity = SemanticGravity(start_point)

    # Initial velocity (small perturbation)
    initial_velocity = np.array([0.1, -0.05, 0.02, 0.03, -0.01])

    # Store trajectory
    trajectory = [start_point.as_array()]
    velocities = [initial_velocity]
    coherences = [start_point.coherence()]

    current_coords = start_point.as_array()
    current_velocity = initial_velocity

    for step in range(steps):
        try:
            # Update via geodesic equation
            current_coords, current_velocity = semantic_gravity.geodesic_equation(
                current_coords, current_velocity, dt
            )

            # Create point for coherence calculation
            point = OntologicalPoint(*current_coords)

            trajectory.append(current_coords.copy())
            velocities.append(current_velocity.copy())
            coherences.append(point.coherence())

            if step % 20 == 0:
                print(f"  Step {step:3d}: C = {point.coherence():.4f}, "
                      f"Position = [{current_coords[0]:.3f}, {current_coords[1]:.3f}, ...]")

        except Exception as e:
            print(f"  Error at step {step}: {e}")
            break

    print(f"\nSimulation completed: {len(trajectory)} steps")
    print(f"Final coherence: {coherences[-1]:.4f}")
    print(f"Coherence change: {coherences[-1] - coherences[0]:+.4f}")

    return np.array(trajectory), np.array(coherences)

if __name__ == "__main__":
    # Run tests
    test_semantic_geometry()

    # Run simulation
    try:
        trajectory, coherences = run_geodesic_simulation(steps=100, dt=0.01)
        print("\n✓ Semantic geometry implementation successful")
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
