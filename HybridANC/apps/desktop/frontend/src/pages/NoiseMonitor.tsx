import { Card } from '../components/Card'
import { RangeControl } from '../components/RangeControl'
import { Spectrum } from '../components/Spectrum'
import { Waveform } from '../components/Waveform'

export function NoiseMonitor() {
  return (
    <div className="page">
      <div className="page-heading">
        <div><p className="eyebrow">Analysis</p><h1>Noise Monitor</h1></div>
        <button className="live-badge"><span /> Mock capture</button>
      </div>
      <div className="monitor-layout">
        <div className="monitor-main">
          <Card title="Time domain" action={<span className="card-tag">WAVEFORM</span>} className="monitor-card">
            <Waveform height={190} seed={4} />
          </Card>
          <Card title="Frequency spectrum" action={<span className="card-tag">FFT 2048</span>} className="monitor-card">
            <Spectrum compact />
          </Card>
        </div>
        <Card title="Monitor controls" className="control-card">
          <RangeControl label="FFT size" value={2048} min={256} max={8192} step={256} unit="" onChange={() => undefined} />
          <RangeControl label="Refresh" value={100} min={20} max={500} step={20} unit=" ms" onChange={() => undefined} />
          <RangeControl label="Display range" value={80} min={40} max={120} step={5} unit=" dB" onChange={() => undefined} />
          <div className="meter-stack">
            <div><span>RMS</span><strong>-34.2 dBFS</strong></div>
            <div><span>Peak</span><strong>-12.8 dBFS</strong></div>
            <div><span>Dominant</span><strong>1.0 kHz</strong></div>
          </div>
          <p className="mock-note">RMS values are mock electrical levels, not calibrated SPL.</p>
        </Card>
      </div>
    </div>
  )
}
