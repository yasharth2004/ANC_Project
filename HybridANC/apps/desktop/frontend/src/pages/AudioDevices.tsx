import { Card } from '../components/Card'
import { Icon } from '../components/Icon'
import { useAudio } from '../context/AudioContext'
import { DeviceSelector } from '../components/DeviceSelector'
import { EmergencyMute } from '../components/EmergencyMute'
import { AudioStartStop } from '../components/AudioStartStop'
import { GainControl } from '../components/GainControl'
import { ProcessingStatus } from '../components/ProcessingStatus'

export function AudioDevices() {
  const { devices, refreshDevices, loading, error, status, configuration, updateConfiguration } = useAudio()
  const inputs = devices.filter((device) => device.kind === 'input')
  const outputs = devices.filter((device) => device.kind === 'output')
  return (
    <div className="page">
      <div className="page-heading">
        <div><p className="eyebrow">Routing</p><h1>Audio Devices</h1></div>
        <button className="live-badge" onClick={() => void refreshDevices()} disabled={loading}>
          <span /> {loading ? 'Refreshing…' : 'Mock inventory'}
        </button>
      </div>

      <div className="devices-layout">
        <Card title="Input device" action={<button className="text-button" onClick={() => void refreshDevices()}>Refresh</button>}>
          <div className="device-detail">
            <div className="device-icon"><Icon name="mic" /></div>
            <div><strong>{inputs.find((device) => device.id === 'builtin-mic')?.name ?? 'No input selected'}</strong><span>{loading ? 'Loading…' : `${inputs.length} mock devices`}</span></div>
            <span className="selected-check">{status === 'error' ? 'Unavailable' : 'Selected'}</span>
          </div>
          <DeviceSelector kind="input" label="Microphone" />
        </Card>

        <Card title="Output device" action={<button className="text-button" onClick={() => void refreshDevices()}>Refresh</button>}>
          <div className="device-detail">
            <div className="device-icon"><Icon name="speaker" /></div>
            <div><strong>{outputs.find((device) => device.id === 'wired-output')?.name ?? 'No output selected'}</strong><span>{loading ? 'Loading…' : `${outputs.length} mock devices`}</span></div>
            <span className="selected-check">{status === 'error' ? 'Unavailable' : 'Selected'}</span>
          </div>
          <DeviceSelector kind="output" label="Headphones" />
        </Card>
      </div>

      {error && <Card title="Device error" className="routing-note error-note"><strong>{error}</strong></Card>}
      <Card title="Format and transport" className="transport-card">
        <div className="transport-grid">
          <label className="select-control">
            <span>Sample rate</span>
            <select
              value={configuration.sampleRate}
              disabled={loading}
              onChange={(event) => void updateConfiguration({ sampleRate: Number(event.target.value) })}
            >
              <option value={44100}>44.1 kHz</option>
              <option value={48000}>48 kHz</option>
            </select>
          </label>
          <label className="select-control">
            <span>Buffer size</span>
            <select
              value={configuration.bufferSize}
              disabled={loading}
              onChange={(event) => void updateConfiguration({ bufferSize: Number(event.target.value) })}
            >
              {[128, 256, 512, 1024].map((size) => <option key={size} value={size}>{size} frames</option>)}
            </select>
          </label>
          <GainControl />
          <AudioStartStop />
          <ProcessingStatus />
          <EmergencyMute />
        </div>
      </Card>
      <Card title="Transport safety" className="routing-note">
        <EmergencyMute />
      </Card>

      <Card title="Routing note" className="routing-note">
        <Icon name="shield" />
        <div>
          <strong>Independent device roles are not assumed.</strong>
          <p>The wired earphone microphone is not treated as an ANC error microphone unless a future calibrated measurement configuration proves its role.</p>
        </div>
      </Card>
    </div>
  )
}
