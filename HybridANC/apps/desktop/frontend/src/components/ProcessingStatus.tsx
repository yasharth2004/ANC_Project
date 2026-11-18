import { useAudio } from '../context/AudioContext'

const labels = {
  idle: 'Idle',
  starting: 'Starting',
  running: 'Running',
  stopping: 'Stopping',
  muted: 'Emergency mute',
  error: 'Error',
}

export function ProcessingStatus() {
  const { status, configuration, error } = useAudio()
  return (
    <div className={`processing-status ${status}`}>
      <div className="processing-status-main">
        <span className="status-led" />
        <div>
          <span>Processing status</span>
          <strong>{labels[status]}</strong>
        </div>
      </div>
      {configuration.physicalANCEnabled ? (
        <span className="physical-warning">Physical ANC requested</span>
      ) : (
        <span>Physical ANC disabled</span>
      )}
      {error && <p className="control-error">{error}</p>}
    </div>
  )
}
