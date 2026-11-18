import { useAudio } from '../context/AudioContext'

export function EmergencyMute() {
  const { emergencyMute, status } = useAudio()
  return (
    <button
      className="stop-button emergency"
      disabled={status === 'muted'}
      onClick={() => void emergencyMute()}
    >
      {status === 'muted' ? 'Muted' : 'Emergency mute'}
    </button>
  )
}
