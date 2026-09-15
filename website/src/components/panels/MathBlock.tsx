import katex from 'katex';
import { useMemo } from 'react';

interface MathBlockProps {
  latex: string;
  display?: boolean;
}

export function MathBlock({ latex, display = true }: MathBlockProps) {
  const html = useMemo(
    () =>
      katex.renderToString(latex, {
        displayMode: display,
        throwOnError: false,
      }),
    [latex, display],
  );

  // eslint-disable-next-line react/no-danger
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
