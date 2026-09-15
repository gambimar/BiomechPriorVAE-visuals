import styles from './SpeedSlider.module.css';

interface SpeedSliderProps {
  speeds: readonly number[];
  index: number;
  onChange: (index: number) => void;
}

// With a fine-grained speed grid (dozens of stops), a label per tick is
// unreadable -- show a sparse subset of ticks, always including the two
// ends, and rely on the current-speed readout for the exact value.
const MAX_TICK_LABELS = 8;

export function SpeedSlider({ speeds, index, onChange }: SpeedSliderProps) {
  const step = Math.max(1, Math.round((speeds.length - 1) / (MAX_TICK_LABELS - 1)));
  const tickIndices = new Set<number>();
  for (let i = 0; i < speeds.length; i += step) tickIndices.add(i);
  tickIndices.add(speeds.length - 1);

  return (
    <div className={styles.wrap}>
      <div className={styles.readout}>{speeds[index]} m/s</div>
      <input
        className={styles.track}
        type="range"
        min={0}
        max={speeds.length - 1}
        step={1}
        value={index}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label="Walking/running speed"
      />
      <div className={styles.ticks}>
        {[...tickIndices]
          .sort((a, b) => a - b)
          .map((i) => (
            <span key={i} data-active={i === index} onClick={() => onChange(i)}>
              {speeds[i]}
            </span>
          ))}
      </div>
    </div>
  );
}
