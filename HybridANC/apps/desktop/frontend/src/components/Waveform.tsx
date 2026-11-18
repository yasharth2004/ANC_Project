type Props = {
  variant?: 'input' | 'output' | 'residual'
  height?: number
  seed?: number
}

export function Waveform({ variant = 'input', height = 150, seed = 1 }: Props) {
  const points = Array.from({ length: 120 }, (_, index) => {
    const x = index / 119 * 100
    const amplitude =
      variant === 'residual'
        ? 7 + ((index * 7 + seed) % 5)
        : 12 + Math.sin(index * 0.31 + seed) * 8
          + Math.sin(index * 0.93 + seed * 2) * 4
    const y = 50 + (index % 2 === 0 ? amplitude : -amplitude)
    return `${x},${y}`
  }).join(' ')

  return (
    <div className={`waveform waveform-${variant}`} style={{ height }}>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-label={`${variant} waveform mock`}>
        <defs>
          <linearGradient id={`fade-${variant}-${seed}`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.1" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0.9" />
          </linearGradient>
        </defs>
        <path d={`M ${points}`} fill="none" vectorEffect="non-scaling-stroke" stroke={`url(#fade-${variant}-${seed})`} />
      </svg>
      <div className="wave-grid" />
    </div>
  )
}
