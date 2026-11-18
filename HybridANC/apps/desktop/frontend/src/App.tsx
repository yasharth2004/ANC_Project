import { useState } from 'react'
import './App.css'
import { Sidebar } from './components/Sidebar'
import { StatusHeader } from './components/StatusHeader'
import { Dashboard } from './pages/Dashboard'
import { NoiseMonitor } from './pages/NoiseMonitor'
import { NoiseSuppression } from './pages/NoiseSuppression'
import { ANCLab } from './pages/ANCLab'
import { AudioDevices } from './pages/AudioDevices'
import { Settings } from './pages/Settings'
import type { PageId } from './types'
import { AudioProvider } from './context/AudioContext'

function AppShell() {
  const [page, setPage] = useState<PageId>('dashboard')

  return (
    <div className="app-shell">
      <Sidebar activePage={page} onNavigate={setPage} />
      <div className="main-shell">
        <StatusHeader />
        <main className="page-content">
          {page === 'dashboard' && <Dashboard />}
          {page === 'noise-monitor' && <NoiseMonitor />}
          {page === 'suppression' && <NoiseSuppression />}
          {page === 'anc-lab' && <ANCLab />}
          {page === 'devices' && <AudioDevices />}
          {page === 'settings' && <Settings />}
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <AudioProvider>
      <AppShell />
    </AudioProvider>
  )
}
