import { Icon } from './Icon'
import type { PageId } from '../types'

type Props = {
  activePage: PageId
  onNavigate: (page: PageId) => void
}

const navigation: Array<{ id: PageId; label: string; icon: Parameters<typeof Icon>[0]['name'] }> = [
  { id: 'dashboard', label: 'Dashboard', icon: 'grid' },
  { id: 'noise-monitor', label: 'Noise Monitor', icon: 'pulse' },
  { id: 'suppression', label: 'Suppression', icon: 'filter' },
  { id: 'anc-lab', label: 'ANC Lab', icon: 'lab' },
  { id: 'devices', label: 'Audio Devices', icon: 'devices' },
]

export function Sidebar({ activePage, onNavigate }: Props) {
  return (
    <aside className="sidebar">
      <button className="brand" onClick={() => onNavigate('dashboard')} aria-label="HybridANC dashboard">
        <span className="brand-mark"><span /></span>
        <span>HYBRID<span>ANC</span></span>
      </button>

      <nav className="navigation" aria-label="Primary navigation">
        {navigation.map((item) => (
          <button
            key={item.id}
            className={activePage === item.id ? 'nav-item active' : 'nav-item'}
            onClick={() => onNavigate(item.id)}
          >
            <Icon name={item.icon} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <button
        className={activePage === 'settings' ? 'nav-item settings-link active' : 'nav-item settings-link'}
        onClick={() => onNavigate('settings')}
      >
        <Icon name="settings" />
        <span>Settings</span>
      </button>
    </aside>
  )
}
