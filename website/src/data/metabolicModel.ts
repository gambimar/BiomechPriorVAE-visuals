// Pulled from BioMAC-Sim-Toolbox/src/model/@Model/getEratec_bhargavaact.m on
// the lab workstation — the smoothed, activation-based continuous variant of
// Bhargava et al.'s metabolic energy model (J Biomech, 2004) used as
// `--metmodel=bhargavaact` in runGait.sh. "Continuous" and "activation-based"
// (vs. the original's stimulation-based, non-differentiable heat-rate terms)
// make it usable as a smooth OCP cost term. LaTeX mirrors the code's own
// variable names; the epsilon-relaxations the code adds purely for
// differentiability (smoothed max()/piecewise functions) are described in
// prose rather than typeset, since they're a numerical technique, not part
// of the physical model.
export const bhargavaEquations = [
  {
    label: 'Activation heat rate',
    latex: 'A = \\phi \\, m \\left( FT \\cdot A_f \\cdot u_f + ST \\cdot A_s \\cdot u_s \\right)',
    note: 'phi = 0.2, A_f = 133 W/kg, A_s = 40 W/kg. u_f = 1 - cos(pi/2 * act), u_s = sin(pi/2 * act), smoothly extended past act=1 to avoid a kink.',
  },
  {
    label: 'Maintenance heat rate',
    latex: 'M = l_M \\, m \\left( FT \\cdot M_f \\cdot u_f + ST \\cdot M_s \\cdot u_s \\right)',
    note: 'M_f = 111 W/kg, M_s = 74 W/kg. l_M is a smoothed version of the piecewise-linear function of normalized fiber length l_ce.',
  },
  {
    label: 'Shortening / lengthening heat rate',
    latex:
      'S = -\\alpha_l v_{ce,l} - \\alpha_s v_{ce,s}, \\quad ' +
      '\\alpha_l = 0.16\\, act \\cdot F_{iso} F_{max} + 0.18\\, F_{ce}, \\quad ' +
      '\\alpha_s = 0.157\\, F_{ce}',
    note: 'v_ce_l, v_ce_s smoothly split contractile velocity into its lengthening/shortening parts.',
  },
  {
    label: 'Mechanical work rate',
    latex: '\\dot{W} = \\max\\!\\left(-F_{ce}\\, v_{ce},\\ 0\\right)',
    note: 'Positive (concentric) work only, via a smooth max().',
  },
  {
    label: 'Total',
    latex: '\\dot{E} = A + M + S + \\dot{W} \\quad [\\mathrm{W/kg\\ muscle\\ mass}]',
    note: 'Summed over all muscles and normalized by body mass to score cost of transport across the speed sweep.',
  },
];

export const metabolicModelMeta = {
  source: 'BioMAC-Sim-Toolbox/src/model/@Model/getEratec_bhargavaact.m',
  reference: 'Bhargava, Pandy & Anderson (2004), J Biomech 37(1):81-88',
  referenceUrl: 'https://doi.org/10.1016/s0021-9290(03)00239-2',
  massDensity: '1059.7 kg/m^3 muscle density',
  maxStress: '250 kPa max muscle stress',
};

export const activationSubstitutionRef = {
  citation: 'van den Bogert (2025), "Strange Effects of Activation Dynamics on Musculoskeletal Trajectory Optimization"',
  url: 'https://doi.org/10.1101/2025.01.30.635759',
};
