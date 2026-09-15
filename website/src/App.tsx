import { useEffect, useState } from 'react';
import type { BubbleId } from './types';
import { HeroFigure } from './components/hero/HeroFigure';
import { DetailPanel } from './components/panels/DetailPanel';
import { PriorPanel } from './components/panels/PriorPanel';
import { ObjectivePanel } from './components/panels/ObjectivePanel';
import { ResultsPanel } from './components/panels/ResultsPanel';
import { SimulationPanel } from './components/panels/SimulationPanel';
import { Header } from './components/layout/Header';

const PANEL_TITLES: Record<BubbleId, string> = {
  objective: 'Task objective',
  prior: 'Prior',
  results: 'Results',
  simulation: 'Simulation',
};

// Prior deliberately has no entry here — PriorPanel renders its own
// interactive version of the same image as its hero, with clickable parts.
// Results and objective (the metabolic-model/task-objective page) also have
// none — no zoom-in hero banner for those two, per user request.
const PANEL_HERO_IMAGES: Partial<Record<BubbleId, { src: string; alt: string }>> = {
  simulation: {
    src: '/media/images/simulation_constraints_crop.png',
    alt: 'Musculoskeletal dynamics and periodicity constraints enforced by the OCP',
  },
};

function App() {
  const [activeBubble, setActiveBubble] = useState<BubbleId | null>(null);

  useEffect(() => {
    document.title = activeBubble
      ? `${PANEL_TITLES[activeBubble]} - BiomechPriorVAE`
      : 'BiomechPriorVAE';
  }, [activeBubble]);

  return (
    <div>
      <Header />
      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '1rem 2rem' }}>
        <h2 style={{ maxWidth: 640, fontSize: '1.5rem', margin: '0 0 0.5rem' }}>
          Realistic human gait emerges from predictive simulations with a learned state prior.
        </h2>
        <p style={{ maxWidth: 640, color: 'var(--text-muted)' }}>
          Click a highlighted region below to see how.
        </p>
        <HeroFigure activeBubble={activeBubble} onSelect={setActiveBubble} />
      </main>

      {activeBubble && (
        <DetailPanel
          title={PANEL_TITLES[activeBubble]}
          onClose={() => setActiveBubble(null)}
          heroImage={PANEL_HERO_IMAGES[activeBubble]}
          hideHeader={activeBubble === 'results' || activeBubble === 'objective'}
        >
          {activeBubble === 'objective' && <ObjectivePanel />}
          {activeBubble === 'prior' && <PriorPanel />}
          {activeBubble === 'results' && <ResultsPanel />}
          {activeBubble === 'simulation' && <SimulationPanel />}
        </DetailPanel>
      )}
    </div>
  );
}

export default App;
