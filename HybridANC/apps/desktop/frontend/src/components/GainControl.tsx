import { useAudio } from '../context/AudioContext'

export function GainControl() {
  const { configuration, updateConfiguration } = useAudio()
  const percent = Math.round(configuration.gain * 100)
  return (
    <label className="range-control gain-control">
      <span>
        <span>Output gain</span>
        <strong>{percent}%</strong>
      </span>
      <input
        type="range"
        min={0}
        max={25}
        step={1}
        value={percent}
        style={{ '--range': `${percent / 25 * 100}%` } as React.CSSProperties}
        onChange={(event) =>
          void updateConfiguration({
            gain: Number(event.target.value) / 100,
          })
        }
      />
      <small>Conservative maximum: 25%</small>
    </label>
  )
}
