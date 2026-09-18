import { asset } from '../lib/asset';

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
  slowmo: asset('media/videos/cost-slowmo.mp4'),
  realtime: asset('media/videos/cost-realtime.mp4'),
};

export const metabolicModelsComparison: ComparisonEntry = {
  title: 'Metabolic Models',
  blurb:
    'Which metabolic cost model best explains gait? Self-chosen (free) speed under four ' +
    'widely-used models -- Bhargava, Umberger, Houdijk, and Lichtwark -- each finding its ' +
    'own emergent cadence.',
  slowmo: asset('media/videos/metabolic-models-slowmo.mp4'),
  realtime: asset('media/videos/metabolic-models-realtime.mp4'),
};

export const contactComparison: ComparisonEntry = {
  title: 'Ground Contact Model',
  blurb:
    'Effect of ground-contact stiffness and damping on running gait, at 4.5 m/s using ' +
    'effort as the task objective: the default model, a 4x-stiffer model, and a model ' +
    'without damping.',
  slowmo: asset('media/videos/contact-slowmo.mp4'),
  realtime: asset('media/videos/contact-realtime.mp4'),
};

// Bilateral hip-abductor weakness sweep (figure04b panel b), filmed from
// BEHIND so the frontal-plane signs -- contralateral pelvic drop, lateral
// trunk lean -- are what the viewer sees. Levels are REMAINING abductor
// strength in %, listed strongest first so the slider reads "more impaired"
// rightward, like the figure's inverted x-axis. One movie per level: that
// level's lowest-metabolic-cost free-speed solve after figure04b's own
// solve-quality filters (scripts/fig04b_trendelenburg_videos.py; the
// slow-mo/real-time pair mirrors the speed sweep's). Weakness is bilateral
// because the OCP's half-cycle mirror constraint cannot express a one-sided
// deficit -- see figure04b.py's module docstring.
export const WEAKNESS_LEVELS = [100, 75, 50, 30, 25, 20, 15, 10, 5] as const;

export interface WeaknessVideoEntry {
  /** Remaining abductor strength, % of nominal. */
  remaining: (typeof WEAKNESS_LEVELS)[number];
  slowmo: string;
  realtime: string;
}

export const weaknessVideos: WeaknessVideoEntry[] = WEAKNESS_LEVELS.map((remaining) => ({
  remaining,
  slowmo: asset(`media/videos/weakness-${remaining}.mp4`),
  realtime: asset(`media/videos/weakness-${remaining}-realtime.mp4`),
}));

export const weaknessComparison = {
  title: 'Abductor Weakness',
  blurb:
    'How does gait adapt to bilateral hip-abductor weakness? Self-chosen walking speed under ' +
    'Bhargava metabolic cost with the abductors weakened in steps from full strength down to ' +
    '5% remaining, viewed from behind: the classic Trendelenburg signs are a contralateral ' +
    'pelvic drop and a compensatory lateral trunk lean over the stance leg.',
};
