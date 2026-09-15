export interface BoxLayout {
  id: string;
  label: string;
  /** Bounding box in % of image size, framing an existing region of the figure. */
  box: { x0: number; y0: number; x1: number; y1: number };
  color: string;
}

// Every hotspot on the hero figure frames a whole region of the schematic
// directly (the objective/prior blocks, the pose row, the constraint row),
// rather than pointing at it from a floating bubble.
export const boxLayouts: BoxLayout[] = [
  {
    id: 'objective',
    label: 'Task objective',
    box: { x0: 19.5, y0: 7, x1: 47.1, y1: 26.1 },
    color: 'var(--color-predsim)',
  },
  {
    id: 'prior',
    label: 'Prior',
    box: { x0: 50.6, y0: 0, x1: 82.2, y1: 30.1 },
    color: 'var(--color-predsim)',
  },
  {
    id: 'results',
    label: 'Results',
    box: { x0: 16, y0: 38.3, x1: 83.75, y1: 72.2 },
    color: 'var(--color-ours)',
  },
  {
    id: 'simulation',
    label: 'Simulation',
    box: { x0: 16, y0: 72.0, x1: 83.75, y1: 96.2 },
    color: 'var(--color-gaitdynamics)',
  },
];
