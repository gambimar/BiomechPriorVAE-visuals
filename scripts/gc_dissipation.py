"""One-off: ground-contact-model energy dissipation, SIPP (bare, linear
spring-damper) vs SIPP SmoothSphere (Hertz + Hunt-Crossley), at ~4.5 m/s.

Not a plotting script -- a scratch computation answering a specific question
(see chat). Run: `python scripts/gc_dissipation.py`.

Both contact laws' 6 state variables per sphere (Fx,Fy,Fz,xc,yc,zc, in that
column order -- see BioMAC-Sim-Toolbox's contact_3d.al) are raw OCP decision
variables, so no forward-kinematics/OpenSim pipeline is needed here: yc/xc/zc
(global contact-point position) and Fy (already bodyweight-normalized, both
contact laws divide by bodyweight internally/by convention) are read directly
off `row['X']` via `gait_loading._contact_sphere_layout`. Contact-point
velocity (needed for the damping/friction terms) isn't separately stored, so
it's central-differenced from the position trajectory (`np.gradient`) --
adequate for a one-off energy estimate, not litigated further.

Both contact laws, read directly from BioMAC-Sim-Toolbox's
c_files/contact/contact_3d.al (bare) and generate_contact_3d.py (smoothsphere):

  bare SIPP (linear spring-damper):
    Fy = k*p*(1 + c*vn),  k=100 BW/m, c=0.75 s/m
    friction: Fx = -Fy*xcdot/sqrt(xcdot^2+1e-4)  (same form for z)

  SIPP SmoothSphere (Hertz + Hunt-Crossley):
    Fy = C_hertz*p^1.5*(1 + 1.5*c*vn) / bw,  C_hertz from E*=1e6 N/m^2,
    R=0.032m (Simbody Hertz coefficient), c=2.0 (dissipation_val)
    friction: Stribeck+viscous, mu = gate*dynamic_friction + viscous*v_slide
    (static=dynamic=0.8 here so the Stribeck peak term cancels)

Hunt-Crossley decomposition: for F = F_elastic*(1 + c_bracket*vn), the
elastic part's own power (d/dt of the stored spring PE) is F_elastic*vn, so
the REMAINDER, F_elastic*c_bracket*vn^2, is what's actually dissipated (not
returned on rebound) -- this is the standard Hunt-Crossley damping-loss
identity, not an approximation of the contact law itself.

Reported in J/(kg*m) (matches metabolicCost's own units elsewhere in this
project): BW*m of dissipated work, times g, IS J/kg/m directly (BW-normalized
force = physical force / (mass*g); multiplying by distance gives energy per
unit mass per g... concretely: E_BWm/distance * bodyweight_N / bodymass =
E_BWm/distance * g).
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gait_loading as gl

G = 9.81  # m/s^2

# Bare SIPP: linear spring-damper (contact_3d.al, "stiffness 100 BW per meter")
LINEAR_K = 100.0     # BW/m
LINEAR_C = 0.75      # s/m (bracket coefficient, i.e. Fy = k*p*(1+c*vn))

# SIPP SmoothSphere: Hertz + Hunt-Crossley (generate_contact_3d.py defaults)
HERTZ_E_STAR = 1.0e6     # N/m^2, plain-strain modulus
HERTZ_RADIUS = 0.032     # m
HERTZ_DISSIPATION = 2.0  # s/m
HERTZ_C_BRACKET = 1.5 * HERTZ_DISSIPATION  # 3.0

FRICTION_DYNAMIC = 0.8
FRICTION_VISCOUS = 0.5
FRICTION_TRANSITION_V = 0.2  # m/s


def _hertz_C(bodymass_kg):
    """Hertz force coefficient C [BW/m^1.5] -- physical C_hertz [N/m^1.5]
    divided by this trial's own bodyweight, matching the contact law's own
    bodyweight-normalization convention (see get_bodyweight() in the C code)."""
    k_hertz = 0.5 * HERTZ_E_STAR ** (2.0 / 3.0)
    C_hertz_phys = (4.0 / 3.0) * k_hertz * np.sqrt(HERTZ_RADIUS * k_hertz)
    bw_N = bodymass_kg * G
    return C_hertz_phys / bw_N


def _sphere_block(states, idx):
    """(Fx, Fy, Fz, xc, yc, zc) time series for one contact sphere, column
    order per contact_3d.al's x1..x6 -- idx is that sphere's Fx column."""
    return (states[:, idx], states[:, idx + 1], states[:, idx + 2],
            states[:, idx + 3], states[:, idx + 4], states[:, idx + 5])


def _dissipated_power_linear(Fy, xc, yc, zc, dt):
    p = np.maximum(-yc, 0.0)
    vn = -np.gradient(yc, dt)
    xcdot = np.gradient(xc, dt)
    zcdot = np.gradient(zc, dt)

    F_elastic = LINEAR_K * p
    P_normal = F_elastic * LINEAR_C * vn ** 2
    P_friction = Fy * (xcdot ** 2 / np.sqrt(xcdot ** 2 + 1e-4)
                       + zcdot ** 2 / np.sqrt(zcdot ** 2 + 1e-4))
    return P_normal, P_friction


def _dissipated_power_hertz(Fy, xc, yc, zc, dt, bodymass_kg):
    # yc here is the SPHERE CENTRE's y-coordinate (CP marker sits at the
    # centre, per generate_contact_3d.py's own comment: indentation
    # a = radius - yc) -- NOT the ground-contact depth directly, unlike the
    # bare linear model's yc. Confirmed by validating the reconstructed force
    # against each model's own actual solved Fy state (see chat / git log):
    # using p=max(-yc,0) here gave ~100% error against actual Fy, because yc
    # sits around +radius during normal stance, never near 0.
    p = np.maximum(HERTZ_RADIUS - yc, 0.0)
    vn = -np.gradient(yc, dt)
    xcdot = np.gradient(xc, dt)
    zcdot = np.gradient(zc, dt)

    C_hertz_BW = _hertz_C(bodymass_kg)
    F_elastic = C_hertz_BW * p ** 1.5
    P_normal = F_elastic * HERTZ_C_BRACKET * vn ** 2

    v_slide = np.sqrt(xcdot ** 2 + zcdot ** 2 + 1e-5)
    vrel = v_slide / FRICTION_TRANSITION_V
    gate = 0.5 * (vrel + 1 - np.sqrt((vrel - 1) ** 2 + 1e-6))  # smooth min(vrel,1)
    mu = gate * FRICTION_DYNAMIC + FRICTION_VISCOUS * v_slide  # static==dynamic here
    P_friction = mu * Fy * v_slide
    return P_normal, P_friction


def dissipation_per_cycle(row, bodymass_kg, contact_law):
    """(normal_J_per_kg_per_m, friction_J_per_kg_per_m, total) for one full
    gait cycle at this row's achieved speed -- both legs' spheres, one
    half-cycle solve (the model's own symmetry means the mirrored second half
    dissipates the same total, so the full-cycle total is just 2x)."""
    states, grf_r_idx, grf_l_idx, _ = gl._contact_sphere_layout(row['X'])
    n = states.shape[0]
    dt = row['dur'] / (n - 1)

    P_normal_total = np.zeros(n)
    P_friction_total = np.zeros(n)
    for idx in list(grf_r_idx) + list(grf_l_idx):
        Fx, Fy, Fz, xc, yc, zc = _sphere_block(states, idx)
        if contact_law == 'linear':
            P_n, P_f = _dissipated_power_linear(Fy, xc, yc, zc, dt)
        else:
            P_n, P_f = _dissipated_power_hertz(Fy, xc, yc, zc, dt, bodymass_kg)
        P_normal_total += P_n
        P_friction_total += P_f

    E_normal_half = np.trapezoid(P_normal_total, dx=dt)   # BW*m
    E_friction_half = np.trapezoid(P_friction_total, dx=dt)  # BW*m
    E_normal_full = 2 * E_normal_half
    E_friction_full = 2 * E_friction_half

    distance_full = row['speed'] * 2 * row['dur']  # m per full gait cycle
    to_J_per_kg_per_m = G / distance_full  # BW*m -> J/kg/m (see module docstring)
    return (E_normal_full * to_J_per_kg_per_m,
            E_friction_full * to_J_per_kg_per_m,
            (E_normal_full + E_friction_full) * to_J_per_kg_per_m)


def main():
    BODYMASS = {'sipp': 75.164620, 'sipp_generic_runmad_smoothsphere': 74.495823}
    CONTACT_LAW = {'sipp': 'linear', 'sipp_generic_runmad_smoothsphere': 'hertz'}

    for model in ('sipp', 'sipp_generic_runmad_smoothsphere'):
        df = gl.load_results(models=(model,), indices=range(1, 30))
        df = df[df['converged'] == True]
        sub = df[np.isclose(df['speed'], 4.53, atol=0.02)]
        results = [dissipation_per_cycle(row, BODYMASS[model], CONTACT_LAW[model])
                  for _, row in sub.iterrows()]
        e_n, e_f, e_tot = np.array(results).T
        print(f'{model} (speed=4.53 m/s, n={len(sub)} trials):')
        print(f'  normal (spring-damper): {e_n.mean():.4f} +/- {e_n.std():.4f} J/kg/m')
        print(f'  friction:               {e_f.mean():.4f} +/- {e_f.std():.4f} J/kg/m')
        print(f'  total GC dissipation:   {e_tot.mean():.4f} +/- {e_tot.std():.4f} J/kg/m')


if __name__ == '__main__':
    main()
