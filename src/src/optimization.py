import numpy as np
import pandas as pd
from src.geometry import SpiralParameters
from src.cavitation import compute_critical_conditions

def sweep_opening_angles(ω: float, r_0: float, ρ: float, p_0: float, p_vap: float,
                         α_range=(5, 80), n_points=100):
    """
    Sweeps spiral opening angles (α) and computes critical conditions for each.
    Returns a DataFrame with α, ω_c, r_crit, t_crit.
    """
    α_values = np.linspace(α_range[0], α_range[1], n_points)
    results = []

    for α_deg in α_values:
        α_rad = np.radians(α_deg)
        params = SpiralParameters(r_0=r_0, α=α_rad, ω=ω)
        crit = compute_critical_conditions(params, ρ, p_0, p_vap)
        results.append({
            "α": α_deg,
            "ω_c": crit["ω_c"],
            "r_crit": crit["r_crit"],
            "t_crit": crit["t_crit"]
        })

    return pd.DataFrame(results)

def find_optimal_alpha(df, objective="max_ω_c"):
    """
    Finds the optimal α based on the objective:
    - 'max_ω_c': maximize critical angular velocity
    - 'min_t_crit': minimize time to cavitation
    """
    if objective == "max_ω_c":
        idx = df["ω_c"].idxmax()
    elif objective == "min_t_crit":
        idx = df["t_crit"].idxmin()
    else:
        raise ValueError("Unsupported objective")
    return df.loc[idx]
from src.optimization import sweep_opening_angles, find_optimal_alpha

df = sweep_opening_angles(
    ω=500, r_0=0.01, ρ=998.0, p_0=101325.0, p_vap=2339.0,
    α_range=(5, 80), n_points=100
)

opt = find_optimal_alpha(df, objective="max_ω_c")
print(f"Optimal α for max ω_c: {opt['α']:.2f}° → ω_c = {opt['ω_c']:.2f} rad/s")
