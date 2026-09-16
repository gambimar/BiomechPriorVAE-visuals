import { simulationReferences } from '../../data/references';
import { links } from '../../data/links';
import { ContentBox } from './ContentBox';

const toolboxBox = {
  role: 'Simulation toolbox',
  citation:
    'BioMAC-Sim-Toolbox, full-body musculoskeletal multibody dynamics enforced at every ' +
    'collocation node, plus half-cycle periodicity.',
  url: links.biomacSimToolbox,
};

const boxes = [toolboxBox, ...[...simulationReferences].reverse()];

const COLORS = [
  'var(--color-accent)',
  'var(--color-gaitdynamics)',
  'var(--color-gaitnet)',
  'var(--color-predsim)',
];

export function SimulationPanel() {
  return (
    <div>
      <p style={{ color: 'var(--text-muted)' }}>
        The gait cycle itself is produced by a direct-collocation, implicit-dynamics optimal
        control problem, solved with IPOPT, initialized from a standing pose that itself
        starts from a randomized joint configuration, so no gait pattern is assumed up front.
        Experiments 1 and 2 use 50 collocation nodes per half-gait-cycle, rerun until 10
        converged solutions are found per condition.
      </p>

      <ContentBox label="Musculoskeletal model" color="var(--color-ours)">
        <p style={{ margin: 0 }}>
          <strong>runMaD-SIPP</strong>: a 3D full-body model with 92 lower-limb muscles and
          torque-actuated arms, an adaptation of the 33-DOF runMaD model, combining a
          symmetrized parameterization of generic body-segment inertial parameters from{' '}
          <strong>SIPP</strong> with the exponential (Hunt-Crossley-type) ground-contact model
          of <strong>Falisse et al. 2022</strong>, in place of runMaD's original linear
          spring-damper contact.
        </p>
      </ContentBox>

      {boxes.map((box, i) => (
        <ContentBox key={box.url} label={box.role} color={COLORS[i % COLORS.length]}>
          <p style={{ margin: '0 0 0.5rem', fontSize: '0.9rem' }}>{box.citation}</p>
          <a href={box.url} target="_blank" rel="noreferrer">
            {box.url}
          </a>
        </ContentBox>
      ))}
    </div>
  );
}
