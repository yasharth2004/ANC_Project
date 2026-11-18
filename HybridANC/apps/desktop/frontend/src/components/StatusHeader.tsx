import { Icon } from './Icon'
import { useAudio } from '../context/AudioContext'

export function StatusHeader() {
  const { devices, configuration, status, loading } = useAudio()
  const input = devices.find((device) => device.id === configuration.inputDeviceId)
  const output = devices.find((device) => device.id === configuration.outputDeviceId)
  const connected = status === 'running'
  const unavailable = input?.available === false || output?.available === false

  return (
    <header className="status-header">
      <div className="status-device">
        <span className="status-icon"><Icon name="mic" size={16} /></span>
        <div>
          <span>Microphone</span>
          <strong>{loading ? 'Loading…' : input?.name ?? 'No input selected'}</strong>
        </div>
      </div>
      <span className="status-route" aria-hidden="true" />
      <div className="status-device">
        <span className="status-icon"><Icon name="speaker" size={16} /></span>
        <div>
          <span>Output</span>
          <strong>{loading ? 'Loading…' : output?.name ?? 'No output selected'}</strong>
        </div>
      </div>
      <div className="status-meta">
        <div className={unavailable ? 'status-error' : ''}>
          <span className={unavailable ? 'status-dot error' : 'status-dot'} />
          {unavailable ? 'Device unavailable' : connected ? 'Mock connected' : 'Ready'}
        </div>
        <span>{configuration.sampleRate / 1000} kHz</span>
        <span>{configuration.bufferSize} frames</span>
      </div>
    </header>
  )
}
