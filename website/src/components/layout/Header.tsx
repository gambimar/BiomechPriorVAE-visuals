import { links } from '../../data/links';
import styles from './Header.module.css';

export function Header() {
  return (
    <header className={styles.header}>
      <h1 className={styles.title}>BiomechPriorVAE</h1>
      <nav className={styles.nav}>
        <a className={styles.pill} href={links.github} target="_blank" rel="noreferrer">
          Code
        </a>
        <a className={styles.pill} href={links.data} target="_blank" rel="noreferrer">
          Data
        </a>
        <a className={styles.pill} href={links.paper} target="_blank" rel="noreferrer">
          Paper
        </a>
      </nav>
    </header>
  );
}
