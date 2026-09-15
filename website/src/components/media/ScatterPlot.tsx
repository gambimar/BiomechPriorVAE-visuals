import { useMemo, useState } from 'react';
import type { LatentPoint } from '../../data/latentSpace';
import styles from './ScatterPlot.module.css';

// A handful of distinguishable colors, cycled if there are more categories
// than colors. trial_id_mapping's category set isn't fixed at build time, so
// colors are assigned dynamically rather than hardcoded per label. Sized
// beyond trial_id_mapping's current 8 categories (walk/run/sit to
// stand/stair/gait_any/static/no_name/other) so distinct labels never wrap
// onto the same color.
const PALETTE = [
  'var(--cividis-25)',
  'var(--cividis-100)',
  'var(--color-gaitnet)',
  'var(--color-gaitdynamics)',
  'var(--color-predsim)',
  'var(--cividis-75)',
  'var(--color-gaitencoder)',
  '#7c9cff',
  '#ff8fa3',
  '#5fd68a',
];

interface ScatterPlotProps {
  points: LatentPoint[];
}

const VIEW = 400;
const PAD = 20;

export function ScatterPlot({ points }: ScatterPlotProps) {
  const [hover, setHover] = useState<{ point: LatentPoint; sx: number; sy: number } | null>(null);

  const labels = useMemo(
    () => Array.from(new Set(points.map((p) => p.label))).sort(),
    [points],
  );
  const colorOf = (label: string) => PALETTE[labels.indexOf(label) % PALETTE.length];

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const scaleX = (x: number) =>
    PAD + ((x - minX) / (maxX - minX || 1)) * (VIEW - 2 * PAD);
  const scaleY = (y: number) =>
    VIEW - PAD - ((y - minY) / (maxY - minY || 1)) * (VIEW - 2 * PAD);

  return (
    <div className={styles.wrap}>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        role="img"
        aria-label="Latent space embedding scatter plot"
      >
        {points.map((p, i) => (
          <circle
            key={i}
            className={styles.point}
            cx={scaleX(p.x)}
            cy={scaleY(p.y)}
            r={hover?.point === p ? 6 : 4}
            fill={colorOf(p.label)}
            fillOpacity={0.85}
            onMouseEnter={() => setHover({ point: p, sx: scaleX(p.x), sy: scaleY(p.y) })}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>
      {hover?.point.thumbnailUrl && (
        <div
          className={styles.tooltip}
          style={{
            left: `${(hover.sx / VIEW) * 100}%`,
            top: `${(hover.sy / VIEW) * 100}%`,
          }}
        >
          <img src={hover.point.thumbnailUrl} alt={`${hover.point.label} pose sample`} />
        </div>
      )}
      <div className={styles.legend}>
        {labels.map((label) => (
          <span key={label}>
            <span className={styles.legendSwatch} style={{ background: colorOf(label) }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
