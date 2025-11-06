import numpy as np
from dataclasses import dataclass

@dataclass
class SpiralParameters:
    r_0: float       # Initial radius [m]
    α: float         # Spiral angle [rad]
    ω: float         # Angular velocity [rad/s]

def opening_factor(α: float) -> float:
    """
    Computes the opening factor k = cot(α)
    """
    return 1 / np.tan(α)

def spiral_radius_theta(params: SpiralParameters, θ: float) -> float:
    """
    Computes radius as a function of angle θ: r(θ) = r_0 * exp(k * θ)
    """
    k = opening_factor(params.α)
    return params.r_0 * np.exp(k * θ)

def spiral_radius_time(params: SpiralParameters, t: float) -> float:
    """
    Computes radius as a function of time: r(t) = r_0 * exp(k * ω * t)
    """
    k = opening_factor(params.α)
    θ = params.ω * t
    return params.r_0 * np.exp(k * θ)

def spiral_velocity_components(params: SpiralParameters, r: float) -> tuple:
    """
    Computes radial, tangential and total velocity at radius r
    """
    k = opening_factor(params.α)
    v_r = k * params.ω * r
    v_θ = params.ω * r
    v_total = np.sqrt(v_r**2 + v_θ**2)
    return v_r, v_θ, v_total
