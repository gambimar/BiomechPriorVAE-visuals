import { BoxHotspot } from '../hero/BoxHotspot';
import { priorHeroLayout, type PriorSection } from './priorHeroLayout';
import { asset } from '../../lib/asset';
import styles from './PriorHero.module.css';

interface PriorHeroProps {
  active: PriorSection | null;
  onSelect: (section: PriorSection) => void;
}

export function PriorHero({ active, onSelect }: PriorHeroProps) {
  return (
    <div className={styles.wrap}>
      <img
        className={styles.img}
        src={asset('media/images/prior_vae_crop.png')}
        alt="VAE encoder/decoder schematic: click a part to see its details"
      />
      {priorHeroLayout.map((layout) => (
        <BoxHotspot
          key={layout.id}
          layout={layout}
          active={active === layout.id}
          onClick={() => onSelect(layout.id)}
        />
      ))}
    </div>
  );
}
