import { useAudio } from '../context/AudioContext'

type Props = {
  kind: 'input' | 'output'
  label: string
}

export function DeviceSelector({ kind, label }: Props) {
  const { devices, configuration, updateConfiguration, loading } = useAudio()
  const matches = devices.filter((device) => device.kind === kind)
  const currentId =
    kind === 'input'
      ? configuration.inputDeviceId
      : configuration.outputDeviceId

  return (
    <label className="select-control device-selector">
      <span>{label}</span>
      <select
        value={currentId}
        disabled={loading || matches.length === 0}
        onChange={(event) =>
          void updateConfiguration(
            kind === 'input'
              ? { inputDeviceId: event.target.value }
              : { outputDeviceId: event.target.value },
          )
        }
      >
        {matches.length === 0 && <option value="">No devices found</option>}
        {matches.map((device) => (
          <option key={device.id} value={device.id} disabled={!device.available}>
            {device.name}{device.disconnected ? ' — disconnected' : ''}
          </option>
        ))}
      </select>
    </label>
  )
}
