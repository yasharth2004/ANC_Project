import type { ReactNode } from 'react'

type Props = {
  title?: string
  action?: ReactNode
  className?: string
  children: ReactNode
}

export function Card({ title, action, className = '', children }: Props) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <header className="card-header">
          {title && <h2>{title}</h2>}
          {action && <div className="card-action">{action}</div>}
        </header>
      )}
      {children}
    </section>
  )
}
