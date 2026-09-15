import type { BoxLayout } from '../hero/heroLayout';

export type PriorSection = 'input' | 'encoder' | 'latent' | 'decoder';

// Regions of prior_vae_crop.png, in % of image size.
export const priorHeroLayout: (BoxLayout & { id: PriorSection })[] = [
  {
    id: 'input',
    label: 'Input',
    box: { x0: 22.3, y0: 22.7, x1: 28.2, y1: 82.1 },
    color: 'var(--color-accent)',
  },
  {
    id: 'encoder',
    label: 'Encoder',
    box: { x0: 32, y0: 22.7, x1: 45.75, y1: 82.1 },
    color: 'var(--color-predsim)',
  },
  {
    id: 'decoder',
    label: 'Decoder',
    box: { x0: 59.5, y0: 22.7, x1: 83, y1: 82.1 },
    color: 'var(--color-predsim)',
  },
  {
    id: 'latent',
    label: 'Latent',
    box: { x0: 49.5, y0: 32.8, x1: 55.7, y1: 69.4 },
    color: 'var(--color-ours)',
  },
];
