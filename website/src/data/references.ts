export interface Reference {
  citation: string;
  role: string;
  url: string;
}

export const simulationReferences: Reference[] = [
  {
    citation:
      'Falisse A, Afschrift M, De Groote F (2022). Modeling toes contributes to realistic ' +
      'stance knee mechanics in three-dimensional predictive simulations of walking. PLOS ONE 17(1): e0256311.',
    role: 'Ground-contact model',
    url: 'https://doi.org/10.1371/journal.pone.0256311',
  },
  {
    citation:
      'Gambietz M, et al. (2026). From body hulls to musculoskeletal models: Personalized ' +
      'inertial parameter estimation. PLOS ONE.',
    role: 'Body segment inertial parameters (BSIP)',
    url: 'https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0349886',
  },
  {
    citation:
      'Nitschke M, Dorschky E, Heinrich D, Schlarb H, Eskofier BM, Koelewijn AD, ' +
      'van den Bogert AJ (2020). Efficient trajectory optimization for curved running using a ' +
      '3D musculoskeletal model with implicit dynamics. Scientific Reports 10, 17655.',
    role: 'Simulation framework (implicit dynamics + direct collocation)',
    url: 'https://doi.org/10.1038/s41598-020-73856-w',
  },
];
