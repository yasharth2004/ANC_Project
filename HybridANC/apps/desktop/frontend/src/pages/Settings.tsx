import { useState } from 'react'
import { Card } from '../components/Card'
import { useAudio } from '../context/AudioContext'
import { EmergencyMute } from '../components/EmergencyMute'

export function Settings() {
  const [theme, setTheme] = useState('Dark')
  const { configuration, updateConfiguration, loading, error } = useAudio()

  return (
    <div className="page">
      <div className="page-heading">
        <div><p className="eyebrow">Preferences</p><h1>Settings</h1></div>
      </div>
      <div className="settings-layout">
        <Card title="Interface">
          <label className="setting-row">
            <div><strong>Appearance</strong><span>Application color theme</span></div>
            <select value={theme} onChange={(event) => setTheme(event.target.value)}>
              <option>Dark</option><option>System</option>
            </select>
          </label>
          <label className="setting-row">
            <div><strong>Audio buffer</strong><span>Planned native-engine setting</span></div>
            <select
              value={configuration.bufferSize}
              disabled={loading}
              onChange={(event) => void updateConfiguration({ bufferSize: Number(event.target.value) })}
            >
              {[128, 256, 512, 1024].map((size) => <option key={size} value={size}>{size}</option>)}
            </select>
          </label>
        </Card>
        <Card title="Safety">
          <div className="setting-row">
            <div><strong>Physical ANC output</strong><span>Always disabled until calibrated hardware is connected</span></div>
            <span className="locked-status">Locked</span>
          </div>
          <div className="setting-row">
            <div><strong>Maximum output gain</strong><span>Conservative prototype ceiling</span></div>
            <strong>-12 dBFS</strong>
          </div>
          <div className="setting-row">
            <div><strong>Emergency stop</strong><span>Immediately mute and close active streams</span></div>
            <EmergencyMute />
          </div>
          {error && <p className="control-error">{error}</p>}
        </Card>
      </div>
    </div>
  )
}
