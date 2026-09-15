import { useState } from 'react';
import { SPEEDS, speedVideos } from '../../data/speedSweep';
import { SpeedSlider } from '../media/SpeedSlider';
import { RealtimeToggle, type PlaybackMode } from '../media/RealtimeToggle';
import { VideoPlayer } from '../media/VideoPlayer';

export function SpeedSweepPanel() {
  const [index, setIndex] = useState(0);
  const [playback, setPlayback] = useState<PlaybackMode>('slowmo');
  const entry = speedVideos[index];

  return (
    <div>
      <p style={{ color: 'var(--text-muted)' }}>
        Predicted gait across the full speed sweep, from slow walking to running. Drag the
        slider to change speed; the slow-mo/real-time switch stays put as you scroll
        through.
      </p>
      <SpeedSlider speeds={SPEEDS} index={index} onChange={setIndex} />
      <div style={{ margin: '1rem 0' }}>
        <RealtimeToggle value={playback} onChange={setPlayback} />
      </div>
      <VideoPlayer
        src={playback === 'slowmo' ? entry.slowmo : entry.realtime}
        label={`${entry.speed} m/s, rendering pending`}
      />
    </div>
  );
}
