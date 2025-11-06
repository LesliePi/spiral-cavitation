import numpy as np
from src.geometry import SpiralParameters, spiral_radius_time, opening_factor

def cavitation_threshold(params: SpiralParameters, ρ: float, p_0: float, p_vap: float) -> float:
    """
    Computes the cavitation threshold Δp = p_0 - p_vap
    """
    return p_0 - p_vap

def critical_angular_velocity(params: SpiralParameters, ρ: float, Δp: float) -> float:
    """
    Computes the critical angular velocity ω_c(r) to avoid cavitation
    """
    r = spiral_radius_time(params, t=0)
    k = opening_factor(params.α)
    ω_c = (1 / (k * r)) * np.sqrt(2 * Δp / (ρ * (1 + k**2)))
    return ω_c

def critical_radius(params: SpiralParameters, ρ: float, Δp: float) -> float:
    """
    Computes the critical radius r_crit where cavitation may begin
    """
    k = opening_factor(params.α)
    r_crit = (1 / (params.ω * np.sqrt(1 + k**2))) * np.sqrt(2 * Δp / ρ)
    return r_crit

def critical_time(params: SpiralParameters, r_crit: float) -> float:
    """
    Computes the critical time t_crit when radius reaches r_crit
    """
    k = opening_factor(params.α)
    t_crit = (1 / (k * params.ω)) * np.log(r_crit / params.r_0)
    return t_crit

def compute_critical_conditions(params: SpiralParameters, ρ: float, p_0: float, p_vap: float) -> dict:
    """
    Returns all critical cavitation parameters in a dictionary
    """
    Δp = cavitation_threshold(params, ρ, p_0, p_vap)
    ω_c = critical_angular_velocity(params, ρ, Δp)
    r_crit = critical_radius(params, ρ, Δp)
    t_crit = critical_time(params, r_crit)
    return {
        "Δp": Δp,
        "ω_c": ω_c,
        "r_crit": r_crit,
        "t_crit": t_crit
    }
from src.geometry import SpiralParameters
from src.cavitation import compute_critical_conditions

params = SpiralParameters(r_0=0.01, α=np.radians(20), ω=500)
results = compute_critical_conditions(params, ρ=998.0, p_0=101325.0, p_vap=2339.0)

print(f"Critical ω_c: {results['ω_c']:.2f} rad/s")
print(f"Critical radius: {results['r_crit']:.4f} m")
print(f"Critical time: {results['t_crit']:.4f} s")
