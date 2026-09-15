export interface LayerSpec {
  name: string;
  inputDim: number | string;
  outputDim: number | string;
  activation?: string;
}

// Cross-checked against the manuscript's own Methods section and
// Supplementary Table S1 (Publications/biomechpriorvae/manuscript.tex,
// supplementary.tex) rather than only the training code, so numbers here are
// the ones actually reported/citable, not just code defaults. x is 50-D
// (23 q + 23 q_dot + 4 foot GRF components), z is 24-D. hidden_dim=512.
export const inputDim = 50;
export const latentDim = 24;
const hiddenDim = 512;

export const encoderLayers: LayerSpec[] = [
  { name: 'LayerNorm', inputDim, outputDim: inputDim },
  { name: 'Linear', inputDim, outputDim: hiddenDim, activation: 'GELU' },
  { name: 'Linear', inputDim: hiddenDim, outputDim: hiddenDim, activation: 'GELU' },
  { name: 'RMSNorm', inputDim: hiddenDim, outputDim: hiddenDim },
  { name: 'Linear', inputDim: hiddenDim, outputDim: hiddenDim },
  { name: 'fc_mu / fc_logvar (parallel)', inputDim: hiddenDim, outputDim: latentDim },
];

export const decoderLayers: LayerSpec[] = [
  { name: 'Linear', inputDim: latentDim, outputDim: hiddenDim, activation: 'GELU' },
  { name: 'RMSNorm', inputDim: hiddenDim, outputDim: hiddenDim },
  { name: 'Linear', inputDim: hiddenDim, outputDim: hiddenDim, activation: 'GELU' },
  { name: 'Linear (mu_theta, log-var, split)', inputDim: hiddenDim, outputDim: 2 * inputDim },
];

// Manuscript's own form: L = (1/beta) * GNLL + KL(Enc_phi(x) || N(0,I)), with
// beta=10 up-weighting the KL term because GNLL takes systematically larger
// values than a conventional MSE loss would. Equivalent to Supplementary
// Table S1's "reconstruction weight 0.1, KL weight 1.0" (1/10 = 0.1) -- both
// forms describe the exact same loss.
export const trainingObjective = {
  summary: 'Standard VAE evidence lower bound (ELBO), Gaussian-NLL reconstruction term',
  formula: 'L = (1/beta) * GNLL(Dec_theta(Enc_phi(x)), x) + KL(Enc_phi(x) || N(0, I)),  beta = 10',
  description:
    'The decoder outputs a per-dimension Gaussian (mu_theta, sigma_theta^2); ' +
    'log-variance is tanh-clamped to [-8, 8] before exponentiating, so the ' +
    'reconstruction term is a Gaussian negative log-likelihood (GNLL) rather than MSE, ' +
    'which makes the loss unitless and penalizes noisy measurement channels (e.g. joint ' +
    'velocities) less than precise ones (e.g. joint coordinates). Equivalently: ' +
    '0.1 x GNLL + 1.0 x KL.',
};

export const hyperparameters: Record<string, string | number> = {
  latentDim,
  hiddenDim,
  reconstructionWeight: 0.1,
  klWeight_beta: 1.0,
  batchSize: 256,
  optimizer: 'AdamW',
  learningRate: '1e-3 (ReduceLROnPlateau, factor 0.8, patience 10)',
  epochs: 100,
  gradClipNorm: 1.0,
  trainValSplit: '0.8 / 0.2 (random)',
  dataset: 'AddBiomechanics 1.0, ca. 58 h of mocap',
  hardware: 'NVIDIA RTX 4090, AMD Ryzen 9 9950X',
  trainingTime: '15.2 h to best checkpoint',
};

// The OCP's own use of the prior at simulation time -- distinct from the
// training objective above: evaluated per collocation node using the
// CURRENT state as input to both encoder and decoder (not a decoder-only
// sample from a latent z, as a typical generative-model use would be),
// averaged over all N nodes in the trajectory.
export const priorInferenceObjective = {
  formula: 'J_prior = (1/N) * sum_{i=1}^{N} GNLL( Dec_theta(Enc_phi(x_i)) , x_i )',
  description:
    'Because trajectory optimization requires the state to satisfy the ' +
    'musculoskeletal dynamics at every node, the prior is queried at each node’s ' +
    'current state directly, not sampled from a latent code the way a generative model would.',
};
