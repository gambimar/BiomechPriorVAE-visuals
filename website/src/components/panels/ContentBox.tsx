import type { CSSProperties, ReactNode } from 'react';
import styles from './ContentBox.module.css';

interface ContentBoxProps {
  label: string;
  color?: string;
  children: ReactNode;
  onClick?: () => void;
  style?: CSSProperties;
}

export function ContentBox({ label, color = 'var(--color-accent)', children, onClick, style }: ContentBoxProps) {
  const boxStyle: CSSProperties & Record<'--box-color', string> = {
    '--box-color': color,
    ...style,
  };

  return (
    <div
      className={`${styles.box} ${onClick ? styles.clickable : ''}`}
      style={boxStyle}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <span className={styles.tab}>{label}</span>
      {children}
    </div>
  );
}
