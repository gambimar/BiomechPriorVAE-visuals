import { activationSubstitutionRef, bhargavaEquations, metabolicModelMeta } from '../../data/metabolicModel';
import { ContentBox } from './ContentBox';
import { MathBlock } from './MathBlock';

export function ObjectivePanel() {
  return (
    <div>
      <p style={{ color: 'var(--text-muted)' }}>
        Alongside the state prior, the OCP minimizes a metabolic-cost term over the full
        gait cycle.
      </p>

      <ContentBox label="Task objective" color="var(--color-predsim)">
        <div style={{ background: 'var(--bg)', padding: '0.75rem', borderRadius: 'var(--radius)', overflowX: 'auto' }}>
          <MathBlock latex="\mathcal{L}_E = \log \frac{1}{md}\int_0^T \dot{E}(x,u)\,dt" />
        </div>
        <p style={{ color: 'var(--text-muted)', margin: 0 }}>
          m is body mass and d is distance traveled over the cycle, so this is the log of the
          cost of transport, in joules per kilogram per meter. It's logarithmic because
          optimizing cost of transport directly conditions badly at low speeds, since dividing
          by speed creates gradients that are large when slow and vanish when fast; the log
          gives a more uniform gradient across the whole range.
        </p>
      </ContentBox>

      <ContentBox label="bhargavaact model" color="var(--color-gaitdynamics)">
        <p style={{ color: 'var(--text-muted)' }}>
          Edot is a per-timestep muscle energy rate, from{' '}
          <a href={metabolicModelMeta.referenceUrl} target="_blank" rel="noreferrer">
            {metabolicModelMeta.reference}
          </a>
          's model, in <code>{metabolicModelMeta.source}</code>. Uses muscle{' '}
          <em>activations</em> in place of the original model's excitations, since optimizing
          excitations directly leads to non-physiological solutions, per{' '}
          <a href={activationSubstitutionRef.url} target="_blank" rel="noreferrer">
            {activationSubstitutionRef.citation}
          </a>
          .
        </p>
        {bhargavaEquations.map((eq) => (
          <div key={eq.label} style={{ marginBottom: '1.1rem' }}>
            <strong style={{ fontSize: '0.9rem' }}>{eq.label}</strong>
            <div
              style={{
                background: 'var(--bg)',
                padding: '0.6rem 0.75rem',
                borderRadius: 'var(--radius)',
                overflowX: 'auto',
                margin: '0.3rem 0',
              }}
            >
              <MathBlock latex={eq.latex} />
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', margin: 0 }}>
              {eq.note}
            </p>
          </div>
        ))}
      </ContentBox>
    </div>
  );
}
