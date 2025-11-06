import numpy as np
from scipy.integrate import solve_ivp
from src.geometry import SpiralParameters, spiral_radius_time, spiral_velocity_components, opening_factor

def external_pressure(t: float, params: SpiralParameters, ρ: float, p_0: float) -> float:
    """
    Computes external pressure at time t along the spiral path using Bernoulli approximation.
    """
    r = spiral_radius_time(params, t)
    v_total = spiral_velocity_components(params, r)[2]
    return p_0 - 0.5 * ρ * v_total**2

def rayleigh_plesset(t, y, params: SpiralParameters, ρ: float, μ: float, σ: float, p_vap: float, p_0: float):
    """
    Rayleigh–Plesset equation for bubble radius R(t) and its derivative R_dot(t)
    """
    R, R_dot = y
    k = opening_factor(params.α)
    p_inf = external_pressure(t, params, ρ, p_0)
    p_g = p_vap  # assuming constant internal gas pressure
    term1 = (p_g - p_inf - 2 * σ / R - 4 * μ * R_dot / R) / (ρ * R)
    term2 = 1.5 * (R_dot**2) / R
    R_ddot = term1 - term2
    return [R_dot, R_ddot]

def run_simulation(params: SpiralParameters, ρ: float, μ: float, σ: float, p_vap: float, p_0: float,
                   R_0: float = 1e-6, R_dot_0: float = 0.0, t_max: float = 0.1, n_points: int = 1000):
    """
    Solves the Rayleigh–Plesset equation numerically over time
    """
    t_eval = np.linspace(0, t_max, n_points)
    y0 = [R_0, R_dot_0]
    sol = solve_ivp(
        rayleigh_plesset,
        [0, t_max],
        y0,
        args=(params, ρ, μ, σ, p_vap, p_0),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    return t_eval, sol
from src.simulation import run_simulation
from src.geometry import SpiralParameters

params = SpiralParameters(r_0=0.01, α=np.radians(20), ω=500)
t, sol = run_simulation(params, ρ=998.0, μ=0.001, σ=0.0728, p_vap=2339.0, p_0=101325.0)

# Plot
import matplotlib.pyplot as plt
plt.plot(t, sol.y[0])
plt.xlabel("Time [s]")
plt.ylabel("Bubble radius R(t) [m]")
plt.title("Rayleigh–Plesset Bubble Dynamics")
plt.grid(True)
plt.show()
