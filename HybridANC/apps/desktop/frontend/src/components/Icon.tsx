import type { ReactNode } from 'react'

type IconName =
  | 'grid'
  | 'pulse'
  | 'filter'
  | 'lab'
  | 'devices'
  | 'settings'
  | 'mic'
  | 'speaker'
  | 'activity'
  | 'shield'
  | 'pause'
  | 'play'

type Props = {
  name: IconName
  size?: number
}

export function Icon({ name, size = 18 }: Props) {
  const paths: Record<IconName, ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    pulse: <path d="M3 12h4l2.2-6 4.1 12 2.4-6H21" />,
    filter: <><path d="M4 5h16M7 12h10M10 19h4" /><circle cx="8" cy="5" r="1" /><circle cx="15" cy="12" r="1" /><circle cx="12" cy="19" r="1" /></>,
    lab: <><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6A2 2 0 0 0 19 18l-5-9V3" /><path d="M8 16h8" /></>,
    devices: <><rect x="3" y="5" width="18" height="12" rx="2" /><path d="M8 21h8M12 17v4" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-1.8 1.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-2.5V20a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L7.7 17l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.6-1H6v-2.5h.5a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L7.7 8l1.8-1.8.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6v-.2H15V5a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 8l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2V14H21a1.7 1.7 0 0 0-1.6 1Z" /></>,
    mic: <><rect x="9" y="2" width="6" height="12" rx="3" /><path d="M5 10a7 7 0 0 0 14 0M12 17v5M8 22h8" /></>,
    speaker: <><rect x="4" y="7" width="4" height="10" rx="1" /><path d="M8 8l7-4v16l-7-4M19 9a5 5 0 0 1 0 6" /></>,
    activity: <path d="M3 12h4l2-5 4 10 3-7h5" />,
    shield: <path d="M12 22s8-3.5 8-10V5l-8-3-8 3v7c0 6.5 8 10 8 10Z" />,
    pause: <><path d="M8 5v14M16 5v14" /></>,
    play: <path d="m7 4 12 8-12 8Z" />,
  }

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  )
}
