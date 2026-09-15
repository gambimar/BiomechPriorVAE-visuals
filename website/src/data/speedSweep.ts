// Full native speed grid (scripts/fig02_full_sweep_videos.py): 0.73-3.53 m/s
// in 0.10 steps, then 3.73-5.53 in 0.20 steps -- evaluation/gait_loading.py's
// own `load_results` default grid, not the 7 hand-curated TARGET_SPEEDS
// figure02.py's published figure uses. Each speed's representative trial is
// selected uniformly: among converged reps, prefer no double-contact
// artifact, then lowest total OCP objective value.
export const SPEEDS = [
  0.73, 0.83, 0.93, 1.03, 1.13, 1.23, 1.33, 1.43, 1.53, 1.63, 1.73, 1.83,
  1.93, 2.03, 2.13, 2.23, 2.33, 2.43, 2.53, 2.63, 2.73, 2.83, 2.93, 3.03,
  3.13, 3.23, 3.33, 3.43, 3.53, 3.73, 3.93, 4.13, 4.33, 4.53, 4.73, 4.93,
  5.13, 5.33, 5.53,
] as const;

export interface SpeedVideoEntry {
  speed: (typeof SPEEDS)[number];
  slowmo: string;
  realtime: string;
}

export const speedVideos: SpeedVideoEntry[] = SPEEDS.map((speed) => ({
  speed,
  slowmo: `/media/videos/speed-${speed}.mp4`,
  realtime: `/media/videos/speed-${speed}-realtime.mp4`,
}));
