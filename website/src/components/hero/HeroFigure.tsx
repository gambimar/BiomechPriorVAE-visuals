import type { BubbleId } from '../../types';
import { boxLayouts } from './heroLayout';
import { BoxHotspot } from './BoxHotspot';
import styles from './HeroFigure.module.css';

interface HeroFigureProps {
  activeBubble: BubbleId | null;
  onSelect: (id: BubbleId) => void;
}

export function HeroFigure({ activeBubble, onSelect }: HeroFigureProps) {
  return (
    <div className={styles.wrap}>
      <img
        className={styles.img}
        src="/media/images/figure00.png"
        alt="Method overview: prior-based OCP gait synthesis schematic"
      />
      {boxLayouts.map((layout) => (
        <BoxHotspot
          key={layout.id}
          layout={layout}
          active={activeBubble === layout.id}
          onClick={() => onSelect(layout.id as BubbleId)}
        />
      ))}
    </div>
  );
}
