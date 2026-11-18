import { useAudio } from '../context/AudioContext'

export function AudioStartStop() {
  const { status, start, stop } = useAudio()
  const active = status === 'running' || status === 'starting'

  return (
    <button
      className={active ? 'primary-control stop' : 'primary-control'}
      disabled={status === 'starting' || status === 'stopping'}
      onClick={() => void (active ? stop() : start())}
    >
      {status === 'starting'
        ? 'Starting…'
        : status === 'stopping'
          ? 'Stopping…'
          : active
            ? 'Stop audio'
            : 'Start audio'}
    </button>
  )
}
