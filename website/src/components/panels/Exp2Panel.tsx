import { useState } from 'react';
import {
  costComparison,
  metabolicModelsComparison,
  contactComparison,
  weaknessComparison,
  WEAKNESS_LEVELS,
  weaknessVideos,
  type ComparisonEntry,
} from '../../data/comparisons';
import { RealtimeToggle, type PlaybackMode } from '../media/RealtimeToggle';
import { Selector } from '../media/Selector';
import { SpeedSlider } from '../media/SpeedSlider';
import { VideoPlayer } from '../media/VideoPlayer';
import { ContentBox } from './ContentBox';

type Mode = 'cost' | 'metabolicModels' | 'contact' | 'weakness';

// The first three are single side-by-side movies; 'weakness' is a sweep
// (one movie per strength level, picked with a slider), so its entry only
// carries title + blurb and the videos come from `weaknessVideos`.
const MODES: { id: Mode; entry: Pick<ComparisonEntry, 'title' | 'blurb'>; color: string }[] = [
  { id: 'cost', entry: costComparison, color: 'var(--color-ours)' },
  { id: 'metabolicModels', entry: metabolicModelsComparison, color: 'var(--color-accent)' },
  { id: 'contact', entry: contactComparison, color: 'var(--color-gaitdynamics)' },
  { id: 'weakness', entry: weaknessComparison, color: 'var(--color-predsim)' },
];

export function Exp2Panel() {
  const [mode, setMode] = useState<Mode>('cost');
  const [playback, setPlayback] = useState<PlaybackMode>('slowmo');
  const [levelIndex, setLevelIndex] = useState(0);
  const active = MODES.find((m) => m.id === mode)!;

  const videos =
    mode === 'weakness'
      ? weaknessVideos[levelIndex]
      : (active.entry as ComparisonEntry);

  return (
    <ContentBox label={active.entry.title} color={active.color}>
      <Selector
        options={MODES.map((m) => ({ id: m.id, label: m.entry.title }))}
        value={mode}
        onChange={setMode}
      />

      <div style={{ marginTop: '1rem' }}>
        <p style={{ color: 'var(--text-muted)', margin: '0 0 0.75rem' }}>{active.entry.blurb}</p>

        {mode === 'weakness' && (
          <div style={{ margin: '0 0 1rem' }}>
            <SpeedSlider
              speeds={WEAKNESS_LEVELS}
              index={levelIndex}
              onChange={setLevelIndex}
              unit="% abductor strength"
              ariaLabel="Remaining hip-abductor strength"
            />
          </div>
        )}

        <RealtimeToggle value={playback} onChange={setPlayback} />
        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', margin: '0.4rem 0 0.75rem' }}>
          {mode === 'weakness'
            ? playback === 'slowmo'
              ? 'Phase-normalized single gait cycle, looped (not actual speed).'
              : 'Real-time: one gait cycle at its actual self-chosen duration.'
            : playback === 'slowmo'
              ? 'Phase-synchronized: every condition shown at the same point in its gait cycle, side by side (not actual speed).'
              : 'Real-time, single gait cycles, can look chunky for multi-condition comparisons.'}
        </p>
        <VideoPlayer src={playback === 'slowmo' ? videos.slowmo : videos.realtime} />
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
