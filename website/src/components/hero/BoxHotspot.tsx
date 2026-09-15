import type { CSSProperties } from 'react';
import type { BoxLayout } from './heroLayout';
import styles from './BoxHotspot.module.css';

interface BoxHotspotProps {
  layout: BoxLayout;
  active: boolean;
  onClick: () => void;
}

export function BoxHotspot({ layout, active, onClick }: BoxHotspotProps) {
  const { x0, y0, x1, y1 } = layout.box;
  const style: CSSProperties & Record<'--box-color', string> = {
    left: `${x0}%`,
    top: `${y0}%`,
    width: `${x1 - x0}%`,
    height: `${y1 - y0}%`,
    '--box-color': layout.color,
  };

  return (
    <button
      type="button"
      className={styles.box}
      style={style}
      data-active={active}
      onClick={onClick}
      aria-expanded={active}
      aria-label={layout.label}
    >
      <span className={styles.tab}>{layout.label}</span>
    </button>
  );
}
