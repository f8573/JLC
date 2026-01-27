import React from 'react'
import { navigate } from '../../utils/navigation'

/**
 * Breadcrumb trail for matrix analysis pages.
 *
 * @param {Object} props
 * @param {Array<{label: string, href?: string}>} props.items
 */
export default function Breadcrumb({ items }) {
  return (
    <div className="flex items-center gap-2 text-xs font-medium text-slate-400">
      {items.map((item, idx) => (
        <React.Fragment key={idx}>
          {idx > 0 && <span className="material-symbols-outlined text-[14px]">chevron_right</span>}
          {item.href ? (
            <a
              className="hover:text-primary"
              href={item.href}
              onClick={(e) => {
                // treat '#' as dashboard
                if (item.href === '#' || item.href === '/') {
                  e.preventDefault()
                  navigate('/')
                }
              }}
            >
              {item.label}
            </a>
          ) : (
            <span className="text-slate-900">{item.label}</span>
          )}
        </React.Fragment>
      ))}
    </div>
  )
}
