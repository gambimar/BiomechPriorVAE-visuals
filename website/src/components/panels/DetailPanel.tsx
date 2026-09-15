import { useEffect, type ReactNode } from 'react';
import styles from './DetailPanel.module.css';

interface DetailPanelProps {
  title: string;
  onClose: () => void;
  children: ReactNode;
  /** Optional full-bleed image at the top of the panel (e.g. a zoomed crop of
   * the relevant part of the hero schematic), to give the "zooming in on this
   * element" feel when a bubble is clicked. */
  heroImage?: { src: string; alt: string };
  /** Skip the title bar (still keeps a floating close button + ESC). */
  hideHeader?: boolean;
}

export function DetailPanel({ title, onClose, children, heroImage, hideHeader }: DetailPanelProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  return (
    <div className={styles.backdrop} role="dialog" aria-label={title}>
      {heroImage && <img className={styles.hero} src={heroImage.src} alt={heroImage.alt} />}
      {hideHeader && (
        <button type="button" className={styles.floatingClose} onClick={onClose}>
          Close
        </button>
      )}
      <div className={styles.content}>
        {!hideHeader && (
          <div className={styles.header}>
            <h2>{title}</h2>
            <button type="button" className={styles.close} onClick={onClose}>
              Close
            </button>
          </div>
        )}
        {children}
      </div>
    </div>
  );
}
