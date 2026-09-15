interface VideoPlayerProps {
  src?: string | null;
  label?: string;
}

export function VideoPlayer({ src, label }: VideoPlayerProps) {
  if (!src) {
    return (
      <div
        style={{
          aspectRatio: '16 / 9',
          borderRadius: 'var(--radius)',
          background: 'var(--bg-card)',
          border: '1px dashed var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.9rem',
        }}
      >
        {label ?? 'video coming soon'}
      </div>
    );
  }

  return (
    // eslint-disable-next-line jsx-a11y/media-has-caption
    <video key={src} controls autoPlay muted loop playsInline>
      <source src={src} type="video/mp4" />
    </video>
  );
}
