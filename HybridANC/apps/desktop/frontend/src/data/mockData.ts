import type { AlgorithmId } from '../types'

export const mockDevices = {
  input: 'MacBook Air Microphone',
  output: 'External Headphones',
  sampleRate: '48 kHz',
  latency: '18.6 ms',
  connection: 'Connected',
}

export const mockLevels = {
  rms: -34.2,
  peak: -12.8,
  noiseFloor: -52.6,
}

export const algorithmMetrics: Record<
  AlgorithmId,
  { rms: number; reduction: number; convergence: number; cost: number }
> = {
  'Fixed Inversion': { rms: 0.008, reduction: -1.8, convergence: 0, cost: 0.01 },
  LMS: { rms: 0.0051, reduction: -8.4, convergence: 48, cost: 0.12 },
  NLMS: { rms: 0.0042, reduction: -10.7, convergence: 31, cost: 0.14 },
  FxLMS: { rms: 0.0031, reduction: -13.2, convergence: 76, cost: 0.19 },
}
