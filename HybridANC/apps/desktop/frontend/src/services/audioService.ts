export type AudioDevice = {
  id: string
  name: string
  kind: 'input' | 'output'
  hostApi: string
  channels: number
  defaultSampleRate: number
  available: boolean
  disconnected?: boolean
}

export type AudioConfiguration = {
  inputDeviceId: string
  outputDeviceId: string
  sampleRate: number
  bufferSize: number
  gain: number
  suppressionEnabled: boolean
  simulationEnabled: boolean
  physicalANCEnabled: boolean
  physicalCalibrationConfirmed: boolean
  physicalCalibrationId: string | null
}

export type ProcessingStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'stopping'
  | 'muted'
  | 'error'

export type AudioService = {
  listDevices: () => Promise<AudioDevice[]>
  getConfiguration: () => Promise<AudioConfiguration>
  setConfiguration: (configuration: AudioConfiguration) => Promise<AudioConfiguration>
  start: () => Promise<AudioConfiguration>
  stop: () => Promise<AudioConfiguration>
  emergencyMute: () => Promise<AudioConfiguration>
  confirmPhysicalANC: (calibrationId: string) => Promise<void>
  startCapture: () => Promise<AudioConfiguration>
  stopCapture: () => Promise<AudioConfiguration>
  startPassthrough: () => Promise<AudioConfiguration>
  stopPassthrough: () => Promise<AudioConfiguration>
  setNoiseSuppression: (enabled: boolean) => Promise<AudioConfiguration>
  getNoiseMetrics: () => Promise<Record<string, number | null>>
  getLatencyMetrics: () => Promise<Record<string, number | null>>
  startANCSimulation: () => Promise<AudioConfiguration>
  stopANCSimulation: () => Promise<AudioConfiguration>
  getANCResults: () => Promise<Record<string, unknown>>
}

const wait = (milliseconds: number) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds))

export const mockDevices: AudioDevice[] = [
  {
    id: 'builtin-mic',
    name: 'MacBook Air Microphone',
    kind: 'input',
    hostApi: 'Core Audio',
    channels: 1,
    defaultSampleRate: 48000,
    available: true,
  },
  {
    id: 'external-mic',
    name: 'External Microphone',
    kind: 'input',
    hostApi: 'Core Audio',
    channels: 1,
    defaultSampleRate: 48000,
    available: true,
  },
  {
    id: 'disconnected-mic',
    name: 'Disconnected USB Microphone',
    kind: 'input',
    hostApi: 'USB',
    channels: 1,
    defaultSampleRate: 48000,
    available: false,
    disconnected: true,
  },
  {
    id: 'wired-output',
    name: 'External Headphones',
    kind: 'output',
    hostApi: 'Core Audio',
    channels: 2,
    defaultSampleRate: 48000,
    available: true,
  },
  {
    id: 'disconnected-output',
    name: 'Unavailable Speakers',
    kind: 'output',
    hostApi: 'Core Audio',
    channels: 2,
    defaultSampleRate: 48000,
    available: false,
    disconnected: true,
  },
]

let configuration: AudioConfiguration = {
  inputDeviceId: 'builtin-mic',
  outputDeviceId: 'wired-output',
  sampleRate: 48000,
  bufferSize: 512,
  gain: 0.05,
  suppressionEnabled: false,
  simulationEnabled: false,
  physicalANCEnabled: false,
  physicalCalibrationConfirmed: false,
  physicalCalibrationId: null,
}

export const mockAudioService: AudioService = {
  async listDevices() {
    await wait(180)
    return mockDevices.map((device) => ({ ...device }))
  },
  async getConfiguration() {
    await wait(120)
    return { ...configuration }
  },
  async setConfiguration(next) {
    await wait(220)
    configuration = { ...configuration, ...next }
    return { ...configuration }
  },
  async start() {
    await wait(500)
    if (
      !mockDevices.find((device) => device.id === configuration.inputDeviceId)?.available ||
      !mockDevices.find((device) => device.id === configuration.outputDeviceId)?.available
    ) {
      throw new Error('Selected input or output device is unavailable.')
    }
    return { ...configuration }
  },
  async stop() {
    await wait(300)
    configuration = {
      ...configuration,
      simulationEnabled: false,
      physicalANCEnabled: false,
    }
    return { ...configuration }
  },
  async emergencyMute() {
    await wait(80)
    configuration = {
      ...configuration,
      gain: 0,
      physicalANCEnabled: false,
      simulationEnabled: false,
    }
    return { ...configuration }
  },
  async confirmPhysicalANC(calibrationId) {
    await wait(160)
    if (!calibrationId.trim()) {
      throw new Error('A validated calibration ID is required.')
    }
    configuration = {
      ...configuration,
      physicalCalibrationConfirmed: true,
      physicalCalibrationId: calibrationId.trim(),
    }
  },
  async startCapture() {
    await wait(180)
    return { ...configuration }
  },
  async stopCapture() {
    await wait(180)
    return { ...configuration }
  },
  async startPassthrough() {
    await wait(180)
    return { ...configuration }
  },
  async stopPassthrough() {
    await wait(180)
    return { ...configuration }
  },
  async setNoiseSuppression(enabled) {
    await wait(180)
    configuration = { ...configuration, suppressionEnabled: enabled }
    return { ...configuration }
  },
  async getNoiseMetrics() {
    await wait(80)
    return { rms: -34.2, peak: -12.8, dbfs: -34.2 }
  },
  async getLatencyMetrics() {
    await wait(80)
    return { callbackIntervalMs: 10.67, captureOverflows: 0, playbackUnderruns: 0 }
  },
  async startANCSimulation() {
    await wait(200)
    configuration = { ...configuration, simulationEnabled: true }
    return { ...configuration }
  },
  async stopANCSimulation() {
    await wait(200)
    configuration = { ...configuration, simulationEnabled: false }
    return { ...configuration }
  },
  async getANCResults() {
    await wait(80)
    return {
      mode: 'mock-simulation',
      physical_output: false,
      algorithm: 'FxLMS',
    }
  },
}

export function createHttpAudioService(baseUrl: string): AudioService {
  const base = baseUrl.replace(/\/$/, '')

  const request = async <T,>(path: string, init?: RequestInit): Promise<T> => {
    const response = await fetch(`${base}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...init?.headers,
      },
    })
    const payload = await response.json()
    if (!response.ok) {
      throw new Error(
        typeof payload.error === 'string'
          ? payload.error
          : `HybridANC backend request failed (${response.status})`,
      )
    }
    return payload as T
  }

  const statusConfiguration = (status: Record<string, unknown>) =>
    (status.configuration as AudioConfiguration)

  return {
    async listDevices() {
      const payload = await request<{
        inputs: AudioDevice[]
        outputs: AudioDevice[]
      }>('/api/v1/devices')
      return [...payload.inputs, ...payload.outputs]
    },
    async getConfiguration() {
      const status = await request<Record<string, unknown>>('/api/v1/status')
      return statusConfiguration(status)
    },
    async setConfiguration(next) {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/configuration',
        {
          method: 'PUT',
          body: JSON.stringify(next),
        },
      )
      return statusConfiguration(payload)
    },
    async start() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/capture/start',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async stop() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/capture/stop',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async emergencyMute() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/emergency-mute',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async confirmPhysicalANC() {
      throw new Error(
        'Physical ANC requires a calibrated native safety workflow and is unavailable.',
      )
    },
    async startCapture() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/capture/start',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async stopCapture() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/capture/stop',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async startPassthrough() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/passthrough/start',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async stopPassthrough() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/passthrough/stop',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async setNoiseSuppression(enabled) {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/suppression',
        { method: 'POST', body: JSON.stringify({ enabled }) },
      )
      return statusConfiguration(payload)
    },
    async getNoiseMetrics() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/metrics/noise',
      )
      return payload.noiseMetrics as Record<string, number | null>
    },
    async getLatencyMetrics() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/metrics/latency',
      )
      return payload.latencyMetrics as Record<string, number | null>
    },
    async startANCSimulation() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/anc/simulation/start',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async stopANCSimulation() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/anc/simulation/stop',
        { method: 'POST', body: '{}' },
      )
      return statusConfiguration(payload)
    },
    async getANCResults() {
      const payload = await request<Record<string, unknown>>(
        '/api/v1/anc/results',
      )
      return payload
    },
  }
}
