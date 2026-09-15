import { useEffect, useState } from 'react';
import { LATENT_DATA_URL, type LatentPoint } from '../../data/latentSpace';
import { asset } from '../../lib/asset';
import { ScatterPlot } from '../media/ScatterPlot';

export function LatentSpacePanel() {
  const [points, setPoints] = useState<LatentPoint[] | null>(null);

  useEffect(() => {
    if (!LATENT_DATA_URL) return;
    fetch(LATENT_DATA_URL)
      .then((r) => r.json())
      .then((raw: LatentPoint[]) =>
        setPoints(
          raw.map((p) => ({
            ...p,
            thumbnailUrl: p.thumbnailUrl ? asset(p.thumbnailUrl) : p.thumbnailUrl,
          })),
        ),
      )
      .catch(() => setPoints(null));
  }, []);

  if (!LATENT_DATA_URL) {
    return (
      <div>
        <p style={{ color: 'var(--text-muted)' }}>
          The VAE learns a compact latent space over gait states. This panel will show a
          2D embedding of sampled motions, walking, running, sit-to-stand, stairs, and more,
          with a preview frame on hover.
        </p>
        <div
          style={{
            border: '1px dashed var(--border)',
            borderRadius: 'var(--radius)',
            padding: '2rem',
            textAlign: 'center',
            color: 'var(--text-muted)',
          }}
        >
          Embedding pending export from notebook/latent_space_analysis.ipynb, plus a
          skeleton-frame render pass for the hover previews.
        </div>
      </div>
    );
  }

  if (!points) {
    return <p style={{ color: 'var(--text-muted)' }}>Loading embedding…</p>;
  }

  return <ScatterPlot points={points} />;
}
