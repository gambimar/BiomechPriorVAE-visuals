import { useState } from 'react';
import type { ResultsView } from '../../types';
import { ContentBox } from './ContentBox';
import { SpeedSweepPanel } from './SpeedSweepPanel';
import { Exp2Panel } from './Exp2Panel';
import { Exp3Panel } from './Exp3Panel';
import { asset } from '../../lib/asset';
import styles from './ContentBox.module.css';

const EXPERIMENTS: {
  id: ResultsView;
  title: string;
  claim: string;
  image: string;
  color: string;
}[] = [
  {
    id: 'speed',
    title: 'Experiment 1',
    claim:
      'Predictive simulations across walking and running speeds',
    image: asset('media/images/figure01.png'),
    color: 'var(--color-ours)',
  },
  {
    id: 'comparisons',
    title: 'Experiment 2',
    claim:
      'Hypothesis testing: which objective (energy cost vs. effort), which metabolic model, ' +
      'and how ground-contact stiffness shape predicted gait.',
    image: asset('media/images/figure04a.png'),
    color: 'var(--color-predsim)',
  },
  {
    id: 'grid',
    title: 'Experiment 3',
    claim:
      'Sparse tracking: reconstructing full-body kinematics from a small number of markers.',
    image: asset('media/images/figure05.png'),
    color: 'var(--color-gaitdynamics)',
  },
];

export function ResultsPanel() {
  const [view, setView] = useState<ResultsView | null>(null);

  return (
    <div>
      <p style={{ color: 'var(--text-muted)' }}>Click a plot to jump to its videos below.</p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
        {EXPERIMENTS.map((exp) => (
          <ContentBox
            key={exp.id}
            label={exp.title}
            color={exp.color}
            onClick={() => setView(exp.id)}
          >
            <img className={styles.img} src={exp.image} alt={exp.claim} />
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: '0.5rem 0 0' }}>
              {exp.claim}
            </p>
          </ContentBox>
        ))}
      </div>

      {view && (
        <>
          <h3 style={{ marginTop: '2rem' }}>Videos</h3>
          {view === 'speed' && <SpeedSweepPanel />}
          {view === 'comparisons' && <Exp2Panel />}
          {view === 'grid' && <Exp3Panel />}
        </>
      )}
    </div>
  );
}
