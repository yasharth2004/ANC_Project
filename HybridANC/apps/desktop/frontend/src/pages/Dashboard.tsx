import { Card } from '../components/Card'
import { Icon } from '../components/Icon'
import { Spectrum } from '../components/Spectrum'
import { Waveform } from '../components/Waveform'
import { mockDevices, mockLevels } from '../data/mockData'
import { useAudio } from '../context/AudioContext'
import { AudioStartStop } from '../components/AudioStartStop'
import { GainControl } from '../components/GainControl'
import { ProcessingStatus } from '../components/ProcessingStatus'
import { EmergencyMute } from '../components/EmergencyMute'

export function Dashboard() {
  const { status } = useAudio()
  return (
    <div className="page">
      <div className="page-heading">
        <div>
          <p className="eyebrow">System overview</p>
          <h1>Dashboard</h1>
        </div>
        <button className="live-badge"><span /> Mock preview</button>
      </div>

      <div className="metric-grid">
        <Card title="Input level" action={<span className="metric-unit">dBFS</span>}>
          <div className="metric-value">{mockLevels.rms}</div>
          <div className="level-track"><span style={{ width: '58%' }} /></div>
          <div className="metric-footer"><span>Peak {mockLevels.peak}</span><span>Floor {mockLevels.noiseFloor}</span></div>
        </Card>
        <Card title="Processing mode" action={<Icon name="shield" size={16} />}>
          <div className="mode-value">
            {status === 'running' ? 'Monitoring' : 'Standby'}
          </div>
          <div className="mode-description">Suppression and ANC outputs are idle.</div>
        </Card>
        <Card title="Measured latency" action={<span className="metric-unit">Mock</span>}>
          <div className="metric-value">18.6 <small>ms</small></div>
          <div className="metric-footer"><span>Not physically calibrated</span></div>
        </Card>
      </div>

      <div className="dashboard-grid">
        <Card title="Live input" action={<span className="card-tag">1 channel</span>} className="wave-card">
          <Waveform height={210} seed={2} />
        </Card>
        <Card title="Frequency profile" action={<span className="card-tag">FFT 2048</span>} className="spectrum-card">
          <Spectrum />
        </Card>
      </div>

      <div className="bottom-grid">
        <Card title="Device connection">
          <div className="connection-row">
            <div className="connection-icon"><Icon name="mic" /></div>
            <div><span>Microphone</span><strong>{mockDevices.input}</strong></div>
            <span className="connected-label">Mock connected</span>
          </div>
          <Card title="Audio control" className="dashboard-control">
            <div className="dashboard-control-grid">
              <AudioStartStop />
              <GainControl />
              <ProcessingStatus />
              <EmergencyMute />
            </div>
          </Card>
          <div className="connection-row">
            <div className="connection-icon"><Icon name="speaker" /></div>
            <div><span>Playback</span><strong>{mockDevices.output}</strong></div>
            <span className="connected-label">Mock connected</span>
          </div>
        </Card>
        <Card title="System status">
          <div className="status-list">
            <div><span>Audio engine</span><strong><i /> Standby</strong></div>
            <div><span>Sample rate</span><strong>{mockDevices.sampleRate}</strong></div>
            <div><span>Processing</span><strong>BYPASS</strong></div>
            <div><span>ANC output</span><strong className="muted">Disabled</strong></div>
          </div>
        </Card>
      </div>
    </div>
  )
}
