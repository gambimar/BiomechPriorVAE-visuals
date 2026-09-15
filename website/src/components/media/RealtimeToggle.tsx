import styles from './RealtimeToggle.module.css';

export type PlaybackMode = 'slowmo' | 'realtime';

interface RealtimeToggleProps {
  value: PlaybackMode;
  onChange: (mode: PlaybackMode) => void;
}

export function RealtimeToggle({ value, onChange }: RealtimeToggleProps) {
  return (
    <div className={styles.wrap}>
      <button
        type="button"
        className={styles.btn}
        data-active={value === 'slowmo'}
        onClick={() => onChange('slowmo')}
      >
        Slow-mo
      </button>
      <button
        type="button"
        className={styles.btn}
        data-active={value === 'realtime'}
        onClick={() => onChange('realtime')}
      >
        Real-time
      </button>
    </div>
  );
}
