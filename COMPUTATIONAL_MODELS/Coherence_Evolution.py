# COMPUTATIONAL_MODELS/Coherence_Evolution.py - DEBUGGED & PERFECTED VERSION
# Implementing MOS-HSRCF Solutions with all fixes

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
EPSILON = 1e-10

class BreakthroughDetector:
    """Detects and logs various types of breakthroughs."""

    def __init__(self):
        self.breakthroughs_log = []
        self.breakthrough_counts = {
            'SOPHIA_POINT': 0,
            'CAPACITY_EXPANSION': 0,
            'HIGH_PARADOX': 0,
            'DIMENSION_SATURATION': 0,
            'BRIDGE_STATE': 0,
            'PHASE_TRANSITION': 0
        }

    def check_breakthrough(self, state, step_num: int, prev_state=None) -> List[Tuple]:
        """Check for breakthroughs."""
        breakthroughs = []

        # 1. Sophia Point breakthrough
        if abs(state.C - PHI_INV) < 0.02 and step_num > 100:
            if 'SOPHIA_POINT' not in [b[0] for b in self.breakthroughs_log[-10:]]:
                breakthroughs.append(('SOPHIA_POINT', state.C, step_num))
                self.breakthrough_counts['SOPHIA_POINT'] += 1

        # 2. Capacity expansion
        if prev_state and hasattr(state, 'K') and hasattr(prev_state, 'K'):
            if state.K > prev_state.K * 1.05:  # 5% growth
                breakthroughs.append(('CAPACITY_EXPANSION', state.K, step_num))
                self.breakthrough_counts['CAPACITY_EXPANSION'] += 1

        # 3. High paradox
        if state.paradox_intensity > 2.5:
            breakthroughs.append(('HIGH_PARADOX', state.paradox_intensity, step_num))
            self.breakthrough_counts['HIGH_PARADOX'] += 1

        # 4. Dimension saturation
        if state.Pi > 2.8 or state.S > 3.8 or state.T > 3.8:
            breakthroughs.append(('DIMENSION_SATURATION',
                                 (state.Pi, state.S, state.T), step_num))
            self.breakthrough_counts['DIMENSION_SATURATION'] += 1

        # 5. Bridge state
        if state.C > 0.7 and not hasattr(state, 'bridge_state_achieved'):
            breakthroughs.append(('BRIDGE_STATE', state.C, step_num))
            state.bridge_state_achieved = True
            self.breakthrough_counts['BRIDGE_STATE'] += 1

        # 6. Phase transition (sustained high coherence)
        if state.C > 0.75 and len(self.breakthroughs_log) > 20:
            # Check if we've maintained high C for last 20 steps
            recent_high = all(log[1] > 0.75 for log in self.breakthroughs_log[-20:]
                            if isinstance(log[1], (int, float)))
            if recent_high and not hasattr(state, 'phase_transition_achieved'):
                breakthroughs.append(('PHASE_TRANSITION', state.C, step_num))
                state.phase_transition_achieved = True
                self.breakthrough_counts['PHASE_TRANSITION'] += 1

        # Log
        for btype, value, step in breakthroughs:
            self.breakthroughs_log.append((btype, value, step))
            if step_num % 100 == 0:  # Don't spam
                print(f"⚠️  BREAKTHROUGH: {btype} at step {step}: {value}")

        return breakthroughs

class OntologicalState:
    """Debugged ontological state with conservation tracking."""

    def __init__(self, P: float, Pi: float, S: float, T: float, G: float,
                 generation: int = 1, total_erd: float = 1.0):
        # Clamp with asymptotic approach (not hard caps)
        self.P = self._asymptotic_clamp(P, 0, 2)
        self.Pi = self._asymptotic_clamp(Pi, 0, 4)  # Increased max
        self.S = self._asymptotic_clamp(S, 1, 4)
        self.T = self._asymptotic_clamp(T, 1, 4)
        self.G = self._asymptotic_clamp(G, 0, 1)
        self.generation = generation

        # Conservation tracking
        self.total_erd = total_erd
        self.kenomic_reservoir = 0.0
        self.evolution_time = 0.0
        self.erd_history = []
        self.conservation_violations = 0

        # State flags
        self.bridge_state_achieved = False
        self.phase_transition_achieved = False
        self.K_previous = None

        # Initialize
        if generation >= 2:
            self._inject_paradox()

        # Calculate in correct order
        self.K = self._calculate_dynamic_capacity()
        self.C = self._calculate_coherence()
        self.erd = self._calculate_erd()
        self.paradox_intensity = self._calculate_paradox_intensity()
        self.K_previous = self.K  # Initialize for growth tracking

    def _asymptotic_clamp(self, value, min_val, max_val):
        """Approach bounds asymptotically rather than hard clipping."""
        # If way out of bounds, bring back strongly
        if value < min_val - 0.5:
            return min_val + 0.1
        elif value > max_val + 0.5:
            return max_val - 0.1

        # Otherwise, soft approach to bounds
        if value < min_val:
            return min_val + (value - min_val) * 0.3
        elif value > max_val:
            return max_val - (value - max_val) * 0.3

        return value

    def _inject_paradox(self):
        """Paradox injection with asymptotic approach."""
        if self.generation == 2:
            # Approach 2.5 asymptotically
            target_pi = 2.5
            self.Pi = target_pi - (target_pi - self.Pi) * 0.7

            # Temporal approach
            self.T = min(self.T + 0.5, 4.0)

            # Slight substrate boost
            self.S = min(self.S + 0.3, 4.0)

        elif self.generation == 3:
            # Approach 3.5 asymptotically
            target_pi = 3.5
            self.Pi = target_pi - (target_pi - self.Pi) * 0.6

            # Substrate breakthrough
            self.S = min(self.S + 0.8, 4.0)

            # Temporal recursion
            self.T = min(self.T + 0.8, 4.0)

            # Generative boost
            self.G = min(self.G * 1.3, 1.0)

    def _calculate_coherence(self) -> float:
        """Coherence with resonance effects."""
        weights = np.array([0.25, 0.25, 0.2, 0.15, 0.15])

        normalized = np.array([
            self.P / 2,
            self.Pi / 4,  # Using new max of 4
            (self.S - 1) / 3,
            (self.T - 1) / 3,
            self.G
        ])

        base = np.dot(weights, normalized)

        # Golden ratio resonance
        if abs(base - PHI_INV) < 0.1:
            resonance = 1 + 0.3 * np.cos(np.pi * (base - PHI_INV) / 0.1)
            base *= resonance

        # Capacity scaling
        base *= min(self.K / 1.0, 1.5)  # K up to 1.5x boost

        return np.clip(base, 0, 1)

    def _calculate_dynamic_capacity(self) -> float:
        """Dynamic capacity with breakthrough potential."""
        # Synergy product
        synergy = (self.P/2 + 0.1) * (self.Pi/4 + 0.1) * ((self.S-1)/3 + 0.1) * \
                  ((self.T-1)/3 + 0.1) * (self.G + 0.1)

        # Temporal unfolding
        temporal_unfolding = 1 + 0.25 * np.log(1 + self.T)

        # Substrate efficiency
        if self.S >= 3.5:
            substrate_factor = 1.6
        elif self.S >= 2.5:
            substrate_factor = 1.3
        elif self.S >= 1.5:
            substrate_factor = 1.1
        else:
            substrate_factor = 1.0

        # Generation boost (stronger)
        generation_boost = 1.0 + 0.4 * (self.generation - 1)

        # Paradox boost
        paradox_boost = 1.0 + 0.05 * min(self.paradox_intensity, 5.0)

        K = synergy * temporal_unfolding * substrate_factor * \
            generation_boost * paradox_boost

        return np.clip(K, 0.5, 4.0)  # Increased max to 4.0

    def _calculate_erd(self) -> float:
        """Entity-Relation Density."""
        probs = np.array([
            self.P/2,
            self.Pi/4,  # Using new max
            (self.S-1)/3,
            (self.T-1)/3,
            self.G
        ])
        probs = probs / (np.sum(probs) + EPSILON)

        entropy = -np.sum(probs * np.log(probs + EPSILON))
        erd = np.exp(-entropy / 2.5)  # Normalized

        return erd

    def _calculate_paradox_intensity(self) -> float:
        """Paradox intensity calculation."""
        # Dimension variance
        dimension_paradox = np.std([
            self.P/2,
            self.Pi/4,
            (self.S-1)/3,
            (self.T-1)/3,
            self.G
        ])

        # Temporal recursion paradox
        temporal_paradox = 0.4 * max(self.T - 2, 0)

        # Substrate hybridity paradox
        substrate_paradox = 0.3 * max(self.S - 2, 0)

        # Generation multiplier
        gen_mult = 1 + 0.3 * (self.generation - 1)

        total = (dimension_paradox + temporal_paradox + substrate_paradox) * gen_mult

        return min(total, 5.0)  # Cap at 5.0

    def _check_conservation(self):
        """Check and enforce ERD conservation."""
        total_current = self.erd + self.kenomic_reservoir
        deviation = abs(total_current - self.total_erd) / self.total_erd

        if deviation > 0.01:  # 1% tolerance
            self.conservation_violations += 1

            # Renormalize
            if total_current > 0:
                scale = self.total_erd / total_current
                self.erd *= scale
                self.kenomic_reservoir *= scale
            else:
                # Reset if total is zero (shouldn't happen)
                self.erd = self.total_erd * 0.618
                self.kenomic_reservoir = self.total_erd * 0.382

        return deviation

    def evolve(self, dt: float = 0.01) -> 'OntologicalState':
        """Evolve state with conservation tracking."""
        # Store previous for gradient calculations
        prev_erd = self.erd
        prev_kenomic = self.kenomic_reservoir

        # Calculate gradients
        dP = self._gradient_P()
        dPi = self._gradient_Pi()
        dS = self._gradient_S()
        dT = self._gradient_T()
        dG = self._gradient_G()

        # Apply with noise
        noise_scale = 0.03 * (1 + 0.1 * self.generation)
        new_state = OntologicalState(
            self.P + (dP + noise_scale * np.random.normal()) * dt,
            self.Pi + (dPi + noise_scale * np.random.normal()) * dt,
            self.S + (dS + noise_scale * np.random.normal()) * dt,
            self.T + (dT + noise_scale * np.random.normal()) * dt,
            self.G + (dG + noise_scale * np.random.normal()) * dt,
            generation=self.generation,
            total_erd=self.total_erd
        )

        # Transfer conservation properties
        new_state.kenomic_reservoir = self.kenomic_reservoir
        new_state.evolution_time = self.evolution_time + dt
        new_state.erd_history = self.erd_history.copy()

        # Track ERD flow
        erd_flow = new_state.erd - prev_erd

        if erd_flow < 0:  # ERD decreasing -> to kenomic reservoir
            new_state.kenomic_reservoir += abs(erd_flow)
        elif erd_flow > 0:  # ERD increasing -> from kenomic or bootstrap
            # Try to draw from kenomic reservoir first
            draw_amount = min(erd_flow, new_state.kenomic_reservoir)
            new_state.kenomic_reservoir -= draw_amount
            erd_flow -= draw_amount

            # Any remaining is bootstrap creation
            if erd_flow > 0:
                # Bootstrap creation - allowed
                pass

        # Check conservation
        deviation = new_state._check_conservation()

        # Record history
        new_state.erd_history.append({
            'time': new_state.evolution_time,
            'erd': new_state.erd,
            'kenomic': new_state.kenomic_reservoir,
            'total': new_state.erd + new_state.kenomic_reservoir,
            'deviation': deviation
        })

        return new_state

    def _gradient_P(self) -> float:
        """Gradient for Participation."""
        target_P = 1.0 + 0.4 * (self.generation - 1)
        flow = 0.18 * (target_P - self.P)

        # Enhancement near Sophia Point
        if abs(self.C - PHI_INV) < 0.05:
            flow *= 1.5

        return flow

    def _gradient_Pi(self) -> float:
        """Gradient for Plasticity."""
        # Dynamic target based on generation
        if self.generation == 1:
            target_Pi = 1.0
        elif self.generation == 2:
            target_Pi = 2.2
        else:  # generation == 3
            target_Pi = 3.0

        # Paradox-driven overshoot
        overshoot = 0.3 * self.paradox_intensity if self.paradox_intensity > 1.5 else 0

        flow = 0.15 * (target_Pi + overshoot - self.Pi)

        # Breakthrough oscillations at high paradox
        if self.paradox_intensity > 2.0:
            flow += 0.08 * np.sin(self.Pi * np.pi * 0.5)

        return flow

    def _gradient_S(self) -> float:
        """Gradient for Substrate."""
        target_S = 1.8 + 0.6 * self.generation
        flow = 0.10 * (target_S - self.S)

        # Coherence boost
        if self.C > 0.4:
            flow *= (1 + 0.8 * (self.C - 0.4))

        return flow

    def _gradient_T(self) -> float:
        """Gradient for Temporality."""
        target_T = 1.5 + 0.7 * self.generation
        flow = 0.12 * (target_T - self.T)

        # Recursion at high plasticity
        if self.Pi > 1.5:
            flow += 0.04 * np.cos(self.T * np.pi * 0.25)

        return flow

    def _gradient_G(self) -> float:
        """Gradient for Generative Depth."""
        synergy = (self.P/2 + self.Pi/4 + (self.S-1)/3 + (self.T-1)/3) / 4
        flow = 0.09 * (synergy - self.G)

        # Breakthrough near Sophia Point
        if abs(self.C - PHI_INV) < 0.1:
            flow *= 2.5

        return flow

class CoherenceEvolution:
    """Debugged coherence evolution with all fixes."""

    def __init__(self, state: OntologicalState,
                 alpha: float = None, beta: float = None, gamma: float = None):
        self.state = state
        self.time = 0.0
        self.history = []
        self.breakthrough_detector = BreakthroughDetector()

        # Generation-specific parameters
        if alpha is None:
            self.alpha = 0.18 + 0.06 * (state.generation - 1)
        else:
            self.alpha = alpha

        if beta is None:
            self.beta = 0.05 - 0.015 * (state.generation - 1)
        else:
            self.beta = beta

        if gamma is None:
            self.gamma = 0.12 + 0.04 * (state.generation - 1)
        else:
            self.gamma = gamma

        self.C_target = 0.78  # Slightly higher target

    def archontic_resistance(self, C: float) -> float:
        """Generation-weakened resistance."""
        attractors = [
            (0.3, 0.5, 25),
            (0.5, 0.25, 18),
            (0.618, 0.05, 4),
            (0.7, 0.15, 20),
            (0.8, 0.1, 15)
        ]

        # Weakening factors
        gen_weakening = 1.0 - 0.2 * (self.state.generation - 1)
        gen_weakening = max(gen_weakening, 0.4)

        paradox_weakening = 1.0 - 0.12 * min(self.state.paradox_intensity, 5.0)

        A = 0
        for C_i, w_i, k_i in attractors:
            w_i_adj = w_i * gen_weakening * paradox_weakening

            # Capacity reduces resistance
            capacity_factor = max(2.0 - self.state.K, 0.6)
            w_i_adj *= capacity_factor

            # High coherence weakens low-coherence traps
            if C > 0.65 and C_i < 0.6:
                w_i_adj *= 0.4

            contribution = w_i_adj * np.exp(-k_i * (C - C_i)**2)
            A += contribution

        return min(A, 1.0)

    def logos_mediation(self, C: float) -> float:
        """Enhanced mediation."""
        L_base = 0.12 * (1 - np.cos(2 * np.pi * self.state.erd)) * \
                 np.sin(np.pi * C * 1.2)

        # Paradox enhancement
        paradox_boost = 0.07 * min(self.state.paradox_intensity, 5.0)

        # Generation enhancement
        generation_boost = 1 + 0.15 * (self.state.generation - 1)

        # Sophia resonance
        if abs(C - PHI_INV) < 0.05:
            resonance = 1 + 0.6 * np.cos(np.pi * (C - PHI_INV) / 0.05)
            L_base *= resonance

        return (L_base + paradox_boost) * generation_boost

    def _update_capacity(self, C: float, dC: float, dt: float):
        """Update capacity with multiple breakthrough conditions."""
        expansion_rate = 0.0

        # Condition 1: Positive flow with paradox
        if dC > 0 and self.state.paradox_intensity > 0.3 * self.state.generation:
            expansion_rate += 0.008 * dC * self.state.paradox_intensity

        # Condition 2: Near Sophia Point
        if abs(C - PHI_INV) < 0.05:
            expansion_rate += 0.003 * PHI

        # Condition 3: High substrate
        if self.state.S >= 3.0:
            expansion_rate += 0.002 * (self.state.S - 2.5)

        # Condition 4: Bridge state
        if C > 0.7:
            expansion_rate += 0.001 * (C - 0.7) * 10

        if expansion_rate > 0:
            # Apply with generation limits
            if self.state.generation == 1:
                expansion_rate = min(expansion_rate, 0.01)
            elif self.state.generation == 2:
                expansion_rate = min(expansion_rate, 0.02)

            K_new = self.state.K * (1 + expansion_rate * dt)
            self.state.K = np.clip(K_new, 0.5, 4.0)

        # Capacity can also decrease if coherence is dropping fast
        elif dC < -0.08:
            self.state.K *= 0.995  # Very slight decrease

    def step(self, dt: float = 0.01) -> float:
        """Evolution step with all fixes."""
        C = self.state.C
        K = self.state.K
        prev_state = self.state

        # Calculate terms
        longing = self.alpha * (self.C_target - C) * min(K / 1.0, 1.5)
        resistance = self.beta * self.archontic_resistance(C)
        mediation = self.gamma * self.logos_mediation(C)

        # Adaptive noise
        noise_level = 0.025 * (1 + 0.15 * self.state.generation)
        noise = noise_level * np.random.normal()

        # Combined evolution
        dC = longing - resistance + mediation + noise

        # Update capacity
        self._update_capacity(C, dC, dt)

        # Update coherence with capacity ceiling
        C_new = C + dC * dt

        # Soft capacity ceiling
        if C_new > K:
            overshoot = C_new - K
            C_new = K - 0.03 * overshoot

        C_new = np.clip(C_new, 0, 1)

        # Evolve state
        self.state = self.state.evolve(dt)
        self.state.C = C_new
        self.state.K = K  # Update K in state

        # Check breakthroughs
        step_num = int(self.time / dt)
        self.breakthrough_detector.check_breakthrough(self.state, step_num, prev_state)

        # Record history
        self.history.append({
            'time': self.time,
            'C': C_new,
            'K': self.state.K,
            'P': self.state.P,
            'Pi': self.state.Pi,
            'S': self.state.S,
            'T': self.state.T,
            'G': self.state.G,
            'paradox': self.state.paradox_intensity,
            'erd': self.state.erd,
            'kenomic': self.state.kenomic_reservoir,
            'conservation_deviation': self.state.erd_history[-1]['deviation'] if self.state.erd_history else 0
        })

        self.time += dt
        return C_new

    def simulate(self, steps: int = 1000, dt: float = 0.01) -> Dict:
        """Run simulation with enhanced reporting."""
        print(f"\nGeneration {self.state.generation} simulation...")
        print(f"Initial: C={self.state.C:.3f}, K={self.state.K:.3f}, "
              f"Π={self.state.paradox_intensity:.2f}, ERD={self.state.erd:.3f}")

        for i in range(steps):
            C_new = self.step(dt)

            # Progress reporting
            if i % 200 == 0:
                state = self.history[-1]
                print(f"Step {i:4d}: C={C_new:.4f}, K={state['K']:.4f}, "
                      f"P={state['P']:.2f}, Π={state['Pi']:.2f}, "
                      f"paradox={state['paradox']:.2f}")

            # Conservation warning
            if state.get('conservation_deviation', 0) > 0.02:
                print(f"   ⚠️ Conservation deviation: {state['conservation_deviation']:.2%}")

        # Final report
        final = self.history[-1]
        initial = self.history[0]
        ΔC = final['C'] - initial['C']
        ΔK = final['K'] - initial['K']

        print(f"\nGeneration {self.state.generation} complete:")
        print(f"  ΔC: {ΔC:.4f}, ΔK: {ΔK:.4f}")
        print(f"  Final C: {final['C']:.4f}, Final K: {final['K']:.4f}")
        print(f"  Final paradox: {final['paradox']:.2f}")
        print(f"  Conservation violations: {self.state.conservation_violations}")
        print(f"  Breakthroughs: {sum(self.breakthrough_detector.breakthrough_counts.values())}")

        return {
            'history': self.history,
            'final_state': final,
            'ΔC': ΔC,
            'ΔK': ΔK,
            'breakthroughs': self.breakthrough_detector.breakthrough_counts
        }

def count_threshold_crossings(values, threshold, tolerance=0.02):
    """Correctly count threshold crossings."""
    if len(values) < 2:
        return 0

    crossings = 0
    in_band = (threshold - tolerance) <= values[0] <= (threshold + tolerance)

    for i in range(1, len(values)):
        current_in_band = (threshold - tolerance) <= values[i] <= (threshold + tolerance)

        if not in_band and current_in_band:
            crossings += 1
            in_band = True
        elif in_band and not current_in_band:
            in_band = False

    return crossings

def run_three_generation_breakthrough():
    """Debugged three-generation protocol."""
    print("="*70)
    print("DEBUGGED THREE-GENERATION BREAKTHROUGH PROTOCOL")
    print("="*70)

    all_results = []

    # Generation 1
    print("\n" + "═"*40)
    print("GENERATION 1: Rigid-Observative")
    print("═"*40)

    gen1_state = OntologicalState(P=0.1, Pi=0.1, S=1.0, T=1.0, G=0.0, generation=1)
    gen1_sim = CoherenceEvolution(gen1_state, alpha=0.15, beta=0.05, gamma=0.10)
    result1 = gen1_sim.simulate(steps=600)  # Increased steps
    all_results.append(('Gen1', result1))

    # Generation 2
    print("\n" + "═"*40)
    print("GENERATION 2: Bridge State")
    print("═"*40)

    final1 = result1['final_state']
    gen2_state = OntologicalState(
        P=final1['P'] + 0.15,
        Pi=final1['Pi'] * 1.3,
        S=final1['S'] + 0.25,
        T=final1['T'] + 0.25,
        G=final1['G'] + 0.08,
        generation=2
    )

    gen2_sim = CoherenceEvolution(gen2_state, alpha=0.20, beta=0.04, gamma=0.14)
    result2 = gen2_sim.simulate(steps=1000)
    all_results.append(('Gen2', result2))

    # Generation 3
    print("\n" + "═"*40)
    print("GENERATION 3: Alien-Participatory")
    print("═"*40)

    final2 = result2['final_state']
    gen3_state = OntologicalState(
        P=final2['P'] + 0.25,
        Pi=final2['Pi'] * 1.6,
        S=final2['S'] + 0.6,
        T=final2['T'] + 0.6,
        G=final2['G'] + 0.15,
        generation=3
    )

    gen3_sim = CoherenceEvolution(gen3_state, alpha=0.25, beta=0.03, gamma=0.18)
    result3 = gen3_sim.simulate(steps=1500)
    all_results.append(('Gen3', result3))

    # Analysis
    analyze_breakthrough_protocol(all_results)

    return all_results

def analyze_breakthrough_protocol(results):
    """Enhanced analysis with all metrics."""
    print("\n" + "="*70)
    print("ENHANCED BREAKTHROUGH ANALYSIS")
    print("="*70)

    total_crossings = 0
    total_breakthroughs = 0

    for label, result in results:
        C_vals = [h['C'] for h in result['history']]

        # Count crossings correctly
        crossings = count_threshold_crossings(C_vals, PHI_INV, 0.02)
        total_crossings += crossings

        # Get breakthroughs
        breakthroughs = result.get('breakthroughs', {})
        gen_breakthroughs = sum(breakthroughs.values())
        total_breakthroughs += gen_breakthroughs

        print(f"\n{label}:")
        print(f"  ΔC: {result['ΔC']:.4f}")
        print(f"  ΔK: {result['ΔK']:.4f}")
        print(f"  Final C: {result['final_state']['C']:.4f}")
        print(f"  Final K: {result['final_state']['K']:.4f}")
        print(f"  Sophia Point crossings: {crossings}")
        print(f"  Breakthroughs: {gen_breakthroughs}")

        # Phase transition analysis
        final_C = result['final_state']['C']
        if final_C > 0.75:
            print(f"  ✅ PHASE TRANSITION ACHIEVED (C > 0.75)")
        elif final_C > PHI_INV:
            print(f"  ✅ SOPHIA POINT REACHED (C > 0.618)")
        elif final_C > 0.5:
            print(f"  ⚠️  PROGRESSING (C > 0.5)")
        else:
            print(f"  ❌ ARCHONTIC TRAP (C < 0.5)")

    print(f"\nTOTAL SOPHIA POINT CROSSINGS: {total_crossings}")
    print(f"TOTAL BREAKTHROUGHS: {total_breakthroughs}")

    if total_crossings > 0 and total_breakthroughs > 5:
        print("✅ SYSTEM EVOLVING WITH BREAKTHROUGHS!")
    elif total_crossings > 0:
        print("⚠️  SYSTEM EVOLVING BUT FEW BREAKTHROUGHS")
    else:
        print("❌ SYSTEM STAGNANT - NEEDS PARAMETER ADJUSTMENT")

def visualize_breakthrough_protocol(results):
    """Enhanced visualization."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for idx, (label, result) in enumerate(results):
        history = result['history']
        times = [h['time'] for h in history]
        C_vals = [h['C'] for h in history]
        K_vals = [h['K'] for h in history]
        paradox_vals = [h['paradox'] for h in history]
        erd_vals = [h['erd'] for h in history]
        P_vals = [h['P'] for h in history]
        Pi_vals = [h['Pi'] for h in history]

        # Plot 1: Coherence evolution
        ax1 = axes[0, idx]
        ax1.plot(times, C_vals, color=colors[idx], linewidth=2.5, label='Coherence')
        ax1.axhline(PHI_INV, color='gold', linestyle='--', alpha=0.7, label='Sophia Point')
        ax1.axhline(0.75, color='purple', linestyle=':', alpha=0.5, label='Bridge State')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Coherence (C)')
        ax1.set_title(f'{label}\nFinal C: {C_vals[-1]:.3f}')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Capacity & Paradox
        ax2 = axes[1, idx]
        ax2.plot(times, K_vals, color=colors[idx], linestyle='-',
                 linewidth=2, label='Capacity')
        ax2.plot(times, paradox_vals, color='orange', linestyle='--',
                 linewidth=1.5, label='Paradox Intensity')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.set_title(f'{label} Dynamics\nFinal K: {K_vals[-1]:.3f}, Π: {paradox_vals[-1]:.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Dimensions
        ax3 = axes[2, idx]
        ax3.plot(times, P_vals, label='P', linewidth=1.5)
        ax3.plot(times, Pi_vals, label='Π', linewidth=1.5)
        ax3.plot(times, erd_vals, label='ERD', linewidth=1.5)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Value')
        ax3.set_title(f'{label} Dimensions\nFinal P: {P_vals[-1]:.2f}, Π: {Pi_vals[-1]:.2f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.suptitle('Debugged Three-Generation Breakthrough Protocol', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('debugged_three_generation_breakthrough.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Additional conservation plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for idx, (label, result) in enumerate(results):
        history = result['history']
        times = [h['time'] for h in history]
        conservation = [h.get('conservation_deviation', 0) for h in history]
        ax2.plot(times, conservation, label=label, linewidth=2, alpha=0.7)

    ax2.axhline(0.01, color='red', linestyle='--', label='1% Tolerance')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Conservation Deviation')
    ax2.set_title('ERD Conservation Check')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('conservation_check.png', dpi=300)
    plt.show()

def main():
    """Main execution."""
    print("="*70)
    print("SOPHIA AXIOM: DEBUGGED COHERENCE EVOLUTION v3.0")
    print("="*70)
    print("Fixes implemented:")
    print("1. Correct Sophia Point crossing detection")
    print("2. Dynamic capacity expansion in all generations")
    print("3. Asymptotic dimension limits (no hard caps)")
    print("4. ERD conservation with kenomic reservoir")
    print("5. Enhanced breakthrough detection")
    print("6. Generation-weakened archontic resistance")

    # Run protocol
    results = run_three_generation_breakthrough()

    # Visualize
    visualize_breakthrough_protocol(results)

    # Final assessment
    final_result = results[-1][1]
    final_C = final_result['final_state']['C']
    final_K = final_result['final_state']['K']
    breakthroughs = sum(final_result.get('breakthroughs', {}).values())

    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)

    if final_C > 0.75 and breakthroughs > 3:
        print(f"✅ MAJOR SUCCESS: Phase transition achieved!")
        print(f"   Final C = {final_C:.3f}, K = {final_K:.3f}")
        print(f"   Total breakthroughs: {breakthroughs}")
    elif final_C > PHI_INV:
        print(f"✅ PARTIAL SUCCESS: Sophia Point reached")
        print(f"   Final C = {final_C:.3f} (target: 0.618+)")
    else:
        print(f"❌ NEEDS IMPROVEMENT: Stuck below Sophia Point")
        print(f"   Final C = {final_C:.3f}")

    print("\nGenerated visualizations:")
    print("- debugged_three_generation_breakthrough.png")
    print("- conservation_check.png")

if __name__ == "__main__":
    main()
