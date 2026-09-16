import { useState } from 'react';
import {
  costComparison,
  metabolicModelsComparison,
  contactComparison,
  type ComparisonEntry,
} from '../../data/comparisons';
import { RealtimeToggle, type PlaybackMode } from '../media/RealtimeToggle';
import { Selector } from '../media/Selector';
import { VideoPlayer } from '../media/VideoPlayer';
import { ContentBox } from './ContentBox';

type Mode = 'cost' | 'metabolicModels' | 'contact';

const MODES: { id: Mode; entry: ComparisonEntry; color: string }[] = [
  { id: 'cost', entry: costComparison, color: 'var(--color-ours)' },
  { id: 'metabolicModels', entry: metabolicModelsComparison, color: 'var(--color-accent)' },
  { id: 'contact', entry: contactComparison, color: 'var(--color-gaitdynamics)' },
];

export function Exp2Panel() {
  const [mode, setMode] = useState<Mode>('cost');
  const [playback, setPlayback] = useState<PlaybackMode>('slowmo');
  const active = MODES.find((m) => m.id === mode)!;

  return (
    <ContentBox label={active.entry.title} color={active.color}>
      <Selector
        options={MODES.map((m) => ({ id: m.id, label: m.entry.title }))}
        value={mode}
        onChange={setMode}
      />

      <div style={{ marginTop: '1rem' }}>
        <p style={{ color: 'var(--text-muted)', margin: '0 0 0.75rem' }}>{active.entry.blurb}</p>

        <RealtimeToggle value={playback} onChange={setPlayback} />
        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', margin: '0.4rem 0 0.75rem' }}>
          {playback === 'slowmo'
            ? 'Phase-synchronized: every condition shown at the same point in its gait cycle, side by side (not actual speed).'
            : 'Real-time, single gait cycles, can look chunky for multi-condition comparisons.'}
        </p>
        <VideoPlayer
          src={playback === 'slowmo' ? active.entry.slowmo : active.entry.realtime}
        />
      </div>
      {mode === 'contact' && (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.75rem' }}>
          Three ground-contact-law variants, shown side by side: Nominal, Stiff (4x stiffness),
          and Bouncy (near-zero damping).
        </p>
      )}
    </ContentBox>
  );
}
