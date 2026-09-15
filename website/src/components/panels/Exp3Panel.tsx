import { MARKER_SETS, MOVEMENTS, sparseTrackingVideo } from '../../data/sparseTrackingMatrix';
import { VideoPlayer } from '../media/VideoPlayer';
import { ContentBox } from './ContentBox';
import styles from './Exp3Panel.module.css';

export function Exp3Panel() {
  return (
    <ContentBox label="Movement x marker set" color="var(--color-gaitdynamics)">
      <p style={{ color: 'var(--text-muted)' }}>
        Ground-truth / no-prior / with-prior reconstruction, overlaid, across movement
        types and marker sets (the two sparse sets tested in the paper, plus the full
        reference). 4 of 9 combinations
        are rendered so far; the rest are real future work, not placeholders standing in
        for hidden data.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th />
              {MARKER_SETS.map((m) => (
                <th key={m}>{m}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MOVEMENTS.map((movement) => (
              <tr key={movement}>
                <td className={styles.rowLabel}>{movement}</td>
                {MARKER_SETS.map((markerSet) => {
                  const src = sparseTrackingVideo[`${movement}|${markerSet}`];
                  return (
                    <td key={markerSet} className={styles.cell}>
                      {src ? (
                        <VideoPlayer src={src} />
                      ) : (
                        <div className={styles.placeholder}>not yet rendered</div>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ContentBox>
  );
}
