import { useState } from 'react';
import {
  decoderLayers,
  encoderLayers,
  hyperparameters,
  inputDim,
  latentDim,
  priorInferenceObjective,
  trainingObjective,
} from '../../data/priorArchitecture';
import { links } from '../../data/links';
import { ContentBox } from './ContentBox';
import { LatentSpacePanel } from './LatentSpacePanel';
import { MathBlock } from './MathBlock';
import { PriorHero } from './PriorHero';
import type { PriorSection } from './priorHeroLayout';

function LayerTable({ title, layers }: { title: string; layers: typeof encoderLayers }) {
  return (
    <>
      <h4 style={{ marginBottom: '0.4rem' }}>{title}</h4>
      <table>
        <thead>
          <tr>
            <th>Layer</th>
            <th>In</th>
            <th>Out</th>
            <th>Activation</th>
          </tr>
        </thead>
        <tbody>
          {layers.map((l, i) => (
            <tr key={i}>
              <td>{l.name}</td>
              <td>{l.inputDim}</td>
              <td>{l.outputDim}</td>
              <td>{l.activation ?? '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

const TITLES: Record<PriorSection, string> = {
  input: 'Input',
  encoder: 'Encoder',
  latent: 'Latent',
  decoder: 'Decoder & training',
};

const COLORS: Record<PriorSection, string> = {
  input: 'var(--color-accent)',
  encoder: 'var(--color-predsim)',
  latent: 'var(--color-ours)',
  decoder: 'var(--color-predsim)',
};

export function PriorPanel() {
  const [section, setSection] = useState<PriorSection | null>(null);

  return (
    <div>
      <p style={{ color: 'var(--text-muted)' }}>
        The learned biomechanics prior used inside the OCP as a state-tracking term: a
        variational autoencoder trained to recognize plausible human movement states, then
        queried at every collocation node as a soft constraint. Click a part below to see
        its details.
      </p>

      <PriorHero active={section} onSelect={setSection} />

      {section && (
        <ContentBox label={TITLES[section]} color={COLORS[section]} style={{ marginTop: '1.5rem' }}>
          {section === 'input' && (
            <>
              <p style={{ margin: '0 0 0.75rem' }}>
                Trained on the{' '}
                <a href={links.addBiomechanicsData} target="_blank" rel="noreferrer">
                  AddBiomechanics
                </a>{' '}
                1.0 dataset, every trial with synchronized force-plate data, ca. 58 hours of
                motion-capture.
              </p>
              <p style={{ color: 'var(--text-muted)' }}>
                Each frame is reduced to a {inputDim}-D state vector x = [q, q&#775;, F]: 23 of
                the 33 DOF of the runMaD musculoskeletal model (excluding the metatarsal and
                subtalar joints, locked in the dataset), 23 velocities, and 4 foot
                ground-reaction-force components. Pelvis DOF are excluded; movements are
                invariant to global position and heading.
              </p>
            </>
          )}

          {section === 'encoder' && <LayerTable title="Encoder layers" layers={encoderLayers} />}

          {section === 'latent' && (
            <>
              <p style={{ color: 'var(--text-muted)' }}>
                2D embedding (t-SNE) of the {latentDim}-dimensional latent space; we show
                ca. 0.01% of the AddBiomechanics dataset here (1,500 of ~19.3M frames).
                Hover over a dot to see the movement!
              </p>
              <LatentSpacePanel />
            </>
          )}

          {section === 'decoder' && (
            <>
              <LayerTable title="Decoder layers" layers={decoderLayers} />

              <h4 style={{ marginTop: '1rem', marginBottom: '0.4rem' }}>Training objective</h4>
              <p>
                <strong>{trainingObjective.summary}</strong>
              </p>
              <div
                style={{
                  background: 'var(--bg)',
                  padding: '0.75rem',
                  borderRadius: 'var(--radius)',
                  overflowX: 'auto',
                }}
              >
                <MathBlock latex="\mathcal{L} = \frac{1}{\beta}\,\mathrm{GNLL}\!\left(\mathrm{Dec}_\theta(\mathrm{Enc}_\phi(x)), x\right) + \mathrm{KL}\!\left(\mathrm{Enc}_\phi(x)\,\|\,\mathcal{N}(0, I)\right),\quad \beta = 10" />
              </div>
              <p style={{ color: 'var(--text-muted)' }}>{trainingObjective.description}</p>

              <h4 style={{ marginBottom: '0.4rem' }}>Evaluated inside the OCP</h4>
              <div
                style={{
                  background: 'var(--bg)',
                  padding: '0.75rem',
                  borderRadius: 'var(--radius)',
                  overflowX: 'auto',
                }}
              >
                <MathBlock latex="\mathcal{J}_{\mathrm{prior}} = \frac{1}{N}\sum_{i=1}^{N} \mathrm{GNLL}\!\left(\mathrm{Dec}_\theta(\mathrm{Enc}_\phi(x_i)), x_i\right)" />
              </div>
              <p style={{ color: 'var(--text-muted)' }}>{priorInferenceObjective.description}</p>

              <h4 style={{ marginBottom: '0.4rem' }}>Hyperparameters</h4>
              <table>
                <tbody>
                  {Object.entries(hyperparameters).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </ContentBox>
      )}
    </div>
  );
}
