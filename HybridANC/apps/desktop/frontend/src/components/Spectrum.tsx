const bands = [18, 28, 46, 38, 62, 74, 55, 82, 68, 48, 36, 58, 44, 28, 20, 13]

export function Spectrum({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? 'spectrum compact' : 'spectrum'}>
      {bands.map((height, index) => (
        <span
          key={index}
          style={{ height: `${height}%`, animationDelay: `${index * 35}ms` }}
        />
      ))}
    </div>
  )
}
