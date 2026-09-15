export interface ComparisonEntry {
  title: string;
  blurb: string;
  /** Phase-normalized across conditions (20fps, looped) -- comparable cycles
   * side by side, but NOT actual playback speed. */
  slowmo: string;
  /** Actual elapsed-time playback (25fps, single pass) -- conditions with
   * different cadence visibly drift apart, exactly as they do in reality. */
  realtime: string;
}

// Blurbs cross-checked against manuscript.tex's "Experiment 2: Hypothesis
// testing" section (Publications/biomechpriorvae/manuscript.tex).
export const costComparison: ComparisonEntry = {
  title: 'Effort vs. Metabolic',
  blurb:
    'Which objective best explains human gait: energy-cost minimization (Bhargava metabolic ' +
    'cost) or fatigue avoidance (effort, the cubic sum of muscle activations)? Compared by ' +
    'walk-to-run transition speed and ground reaction forces; shown here at 1.6 m/s walking.',
  slowmo: '/media/videos/cost-slowmo.mp4',
  realtime: '/media/videos/cost-realtime.mp4',
};

export const metabolicModelsComparison: ComparisonEntry = {
  title: 'Metabolic Models',
  blurb:
    'Which metabolic cost model best explains gait? Self-chosen (free) speed under four ' +
    'widely-used models -- Bhargava, Umberger, Houdijk, and Lichtwark -- each finding its ' +
    'own emergent cadence.',
  slowmo: '/media/videos/metabolic-models-slowmo.mp4',
  realtime: '/media/videos/metabolic-models-realtime.mp4',
};

export const contactComparison: ComparisonEntry = {
  title: 'Ground Contact Model',
  blurb:
    'Effect of ground-contact stiffness and damping on running gait, at 4.5 m/s using ' +
    'effort as the task objective: the default model, a 4x-stiffer model, and a model ' +
    'without damping.',
  slowmo: '/media/videos/contact-slowmo.mp4',
  realtime: '/media/videos/contact-realtime.mp4',
};
