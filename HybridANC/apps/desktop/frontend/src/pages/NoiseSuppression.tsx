import { useState } from 'react'
import { Card } from '../components/Card'
import { RangeControl } from '../components/RangeControl'
import { Waveform } from '../components/Waveform'
import { useAudio } from '../context/AudioContext'
import { EmergencyMute } from '../components/EmergencyMute'

export function NoiseSuppression() {
  const [strength, setStrength] = useState(45)
  const { configuration, updateConfiguration } = useAudio()
  const active = configuration.suppressionEnabled

  const toggle = () =>
    void updateConfiguration({ suppressionEnabled: !active })

  return (
    <div className="page">
      <div className="page-heading">
        <div><p className="eyebrow">Microphone signal</p><h1>Noise Suppression</h1></div>
        <button className={active ? 'live-badge running' : 'live-badge'} onClick={toggle}>
          <span /> {active ? 'Preview enabled' : 'BYPASS'}
        </button>
      </div>

      <div className="suppression-layout">
        <div className="comparison-stack">
          <Card title="Original microphone" action={<span className="card-tag">INPUT</span>}>
            <Waveform height={155} seed={8} />
          </Card>
          <Card title="Processed microphone" action={<span className="card-tag">{active ? 'SIMULATED' : 'BYPASS'}</span>}>
            <Waveform variant="residual" height={155} seed={11} />
          </Card>
        </div>
        <Card title="Processing" className="control-card">
          <div className="toggle-row">
            <div><strong>Noise suppression</strong><span>Software microphone process</span></div>
            <button className={active ? 'toggle enabled' : 'toggle'} onClick={toggle} aria-label="Toggle noise suppression">
              <span />
            </button>
          </div>
          <RangeControl label="Suppression strength" value={strength} min={0} max={100} unit="%" onChange={setStrength} />
          <RangeControl label="Spectral floor" value={12} min={0} max={30} unit=" dB" onChange={() => undefined} />
          <RangeControl label="Speech preservation" value={72} min={0} max={100} unit="%" onChange={() => undefined} />
          <div className="stats-list">
            <div><span>Input SNR</span><strong>12.4 dB</strong></div>
          </div>
          <p className="mock-note">Mock comparison only. No live microphone processing is connected.</p>
          <EmergencyMute />
        </Card>
      </div>
    </div>
  )
}
