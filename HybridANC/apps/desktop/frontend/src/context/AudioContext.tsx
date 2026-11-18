import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  mockAudioService,
  createHttpAudioService,
  type AudioService,
  type AudioConfiguration,
  type AudioDevice,
  type ProcessingStatus,
} from '../services/audioService'

type AudioContextValue = {
  devices: AudioDevice[]
  configuration: AudioConfiguration
  status: ProcessingStatus
  error: string | null
  loading: boolean
  refreshDevices: () => Promise<void>
  updateConfiguration: (partial: Partial<AudioConfiguration>) => Promise<void>
  start: () => Promise<void>
  stop: () => Promise<void>
  emergencyMute: () => Promise<void>
  confirmPhysicalCalibration: (calibrationId: string) => Promise<void>
}

const AudioContext = createContext<AudioContextValue | null>(null)

const apiBaseUrl = import.meta.env.VITE_HYBRIDANC_API_URL as string | undefined
const audioService: AudioService = apiBaseUrl
  ? createHttpAudioService(apiBaseUrl)
  : mockAudioService

export function AudioProvider({ children }: { children: ReactNode }) {
  const [devices, setDevices] = useState<AudioDevice[]>([])
  const [configuration, setConfiguration] = useState<AudioConfiguration>({
    inputDeviceId: '',
    outputDeviceId: '',
    sampleRate: 48000,
    bufferSize: 512,
    gain: 0.05,
    suppressionEnabled: false,
    simulationEnabled: false,
    physicalANCEnabled: false,
    physicalCalibrationConfirmed: false,
    physicalCalibrationId: null,
  })
  const [status, setStatus] = useState<ProcessingStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refreshDevices = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [listed, initial] = await Promise.all([
        audioService.listDevices(),
        audioService.getConfiguration(),
      ])
      setDevices(listed)
      setConfiguration(initial)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to load audio devices.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshDevices()
  }, [refreshDevices])

  const updateConfiguration = useCallback(async (partial: Partial<AudioConfiguration>) => {
    setError(null)
    const next = { ...configuration, ...partial }
    const input = devices.find((device) => device.id === next.inputDeviceId)
    const output = devices.find((device) => device.id === next.outputDeviceId)
    if (input?.kind === 'input' && !input.available) {
      setError('Selected microphone is disconnected or unavailable.')
      return
    }
    if (output?.kind === 'output' && !output.available) {
      setError('Selected output is disconnected or unavailable.')
      return
    }
    if (next.gain < 0 || next.gain > 0.25) {
      setError('Output gain must remain between 0% and 25%.')
      return
    }
    if (![44100, 48000].includes(next.sampleRate)) {
      setError('Only 44.1 kHz and 48 kHz are supported.')
      return
    }
    if (![128, 256, 512, 1024].includes(next.bufferSize)) {
      setError('Select a supported buffer size.')
      return
    }
    setLoading(true)
    try {
      setConfiguration(await audioService.setConfiguration(next))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to update configuration.')
    } finally {
      setLoading(false)
    }
  }, [configuration, devices])

  const start = useCallback(async () => {
    setError(null)
    setStatus('starting')
    try {
      setConfiguration(await audioService.start())
      setStatus('running')
    } catch (caught) {
      setStatus('error')
      setError(caught instanceof Error ? caught.message : 'Unable to start audio.')
    }
  }, [])

  const stop = useCallback(async () => {
    setError(null)
    setStatus('stopping')
    try {
      setConfiguration(await audioService.stop())
      setStatus('idle')
    } catch (caught) {
      setStatus('error')
      setError(caught instanceof Error ? caught.message : 'Unable to stop audio.')
    }
  }, [])

  const emergencyMute = useCallback(async () => {
    setError(null)
    setStatus('muted')
    try {
      setConfiguration(await audioService.emergencyMute())
      setStatus('muted')
    } catch (caught) {
      setStatus('error')
      setError(caught instanceof Error ? caught.message : 'Emergency mute failed.')
    }
  }, [])

  const confirmPhysicalCalibration = useCallback(async (calibrationId: string) => {
    setError(null)
    setLoading(true)
    try {
      await audioService.confirmPhysicalANC(calibrationId)
      setConfiguration(await audioService.getConfiguration())
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Calibration confirmation failed.')
      throw caught
    } finally {
      setLoading(false)
    }
  }, [])

  const value = useMemo(
    () => ({
      devices,
      configuration,
      status,
      error,
      loading,
      refreshDevices,
      updateConfiguration,
      start,
      stop,
      emergencyMute,
      confirmPhysicalCalibration,
    }),
    [
      devices,
      configuration,
      status,
      error,
      loading,
      refreshDevices,
      updateConfiguration,
      start,
      stop,
      emergencyMute,
      confirmPhysicalCalibration,
    ],
  )

  return <AudioContext.Provider value={value}>{children}</AudioContext.Provider>
}

export function useAudio() {
  const value = useContext(AudioContext)
  if (!value) {
    throw new Error('useAudio must be used inside AudioProvider')
  }
  return value
}
