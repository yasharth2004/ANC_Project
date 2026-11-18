import { useState } from 'react'
import { Card } from '../components/Card'
import { RangeControl } from '../components/RangeControl'
import { Waveform } from '../components/Waveform'
import { algorithmMetrics } from '../data/mockData'
import type { AlgorithmId } from '../types'
import { useAudio } from '../context/AudioContext'
import { ConfirmationDialog } from '../components/ConfirmationDialog'

const algorithms: AlgorithmId[] = ['Fixed Inversion', 'LMS', 'NLMS', 'FxLMS']

export function ANCLab() {
  const [algorithm, setAlgorithm] = useState<AlgorithmId>('FxLMS')
  const [delay, setDelay] = useState(16)
  const [frequency, setFrequency] = useState(100)
  const [filterLength, setFilterLength] = useState(64)
  const [learningRate, setLearningRate] = useState(1)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [calibrationId, setCalibrationId] = useState('')
  const [confirmBusy, setConfirmBusy] = useState(false)
  const { configuration, updateConfiguration, confirmPhysicalCalibration, error } = useAudio()
  const metrics = algorithmMetrics[algorithm]

  const toggleSimulation = () =>
    void updateConfiguration({ simulationEnabled: !configuration.simulationEnabled })

  const confirmPhysical = async () => {
    if (!calibrationId.trim()) return
    setConfirmBusy(true)
    try {
      await confirmPhysicalCalibration(calibrationId)
      await updateConfiguration({ physicalANCEnabled: true })
    } finally {
      setConfirmBusy(false)
      setConfirmOpen(false)
    }
  }

  return (
    <div className="page">
      <div className="page-heading">
        <div><p className="eyebrow">Offline experiment</p><h1>ANC Lab</h1></div>
        <button
          className={configuration.simulationEnabled ? 'live-badge running' : 'simulation-badge'}
          onClick={toggleSimulation}
        >
          {configuration.simulationEnabled ? 'Simulation running' : 'Simulation only'}
        </button>
      </div>

      <div className="anc-layout">
        <Card title="Experiment setup" className="control-card anc-controls">
          <label className="select-control">
            <span>Algorithm</span>
            <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value as AlgorithmId)}>
              {algorithms.map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label className="select-control">
            <span>Noise type</span>
            <select defaultValue="Constant tone"><option>Constant tone</option><option>White noise</option><option>Multi-tone</option><option>Changing amplitude</option></select>
          </label>
          <RangeControl label="Noise frequency" value={frequency} min={20} max={2000} step={10} unit=" Hz" onChange={setFrequency} />
          <RangeControl label="Filter length" value={filterLength} min={8} max={512} step={8} unit=" taps" onChange={setFilterLength} />
          <RangeControl label="Learning rate" value={learningRate} min={1} max={100} unit="%" onChange={setLearningRate} />
          <RangeControl label="Secondary delay" value={delay} min={0} max={64} unit=" samples" onChange={setDelay} />
          <div className="anc-safety">
            <span>Physical output</span>
            <strong>{configuration.physicalANCEnabled ? 'Enabled — mock only' : 'Disabled'}</strong>
          </div>
          <button
            className="physical-enable"
            disabled={!calibrationId.trim() && !configuration.physicalANCEnabled}
            onClick={() =>
              configuration.physicalANCEnabled
                ? void updateConfiguration({ physicalANCEnabled: false })
                : setConfirmOpen(true)
            }
          >
            {configuration.physicalANCEnabled ? 'Disable physical request' : 'Confirm physical test'}
          </button>
          {!configuration.physicalCalibrationConfirmed && (
            <p className="mock-note">Physical output requires a validated calibration ID.</p>
          )}
          {error && <p className="control-error">{error}</p>}
          <label className="select-control calibration-id">
            <span>Calibration ID</span>
            <input
              value={calibrationId}
              placeholder="CAL-0000"
              onChange={(event) => setCalibrationId(event.target.value)}
            />
          </label>
        </Card>

        <div className="anc-main">
          <Card title="Residual simulation" action={<span className="card-tag">NOT PHYSICAL</span>}>
            <Waveform variant="residual" height={220} seed={algorithm.length} />
          </Card>
          <div className="anc-metrics">
            <Card title="Residual RMS"><strong>{metrics.rms.toFixed(4)}</strong><span>Relative units</span></Card>
            <Card title="Noise reduction"><strong>{metrics.reduction.toFixed(1)} dB</strong><span>Mock result</span></Card>
            <Card title="Convergence"><strong>{metrics.convergence} samples</strong><span>Off-line estimate</span></Card>
            <Card title="Processing cost"><strong>{metrics.cost.toFixed(2)} ms</strong><span>Mock runtime</span></Card>
          </div>
          <ConfirmationDialog
            open={confirmOpen}
            title="Enable experimental physical ANC?"
            description="This mock confirmation does not activate audio output. It requires a validated calibration ID and does not replace a controlled acoustic experiment."
            confirmLabel="Acknowledge and enable"
            busy={confirmBusy}
            onCancel={() => setConfirmOpen(false)}
            onConfirm={() => void confirmPhysical()}
          />
        </div>
      </div>
    </div>
  )
}
