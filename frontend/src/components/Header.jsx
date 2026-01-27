import React from 'react'
import Logo from './ui/Logo'
import SearchBar from './ui/SearchBar'
import IconButton from './ui/IconButton'
import UserAvatar from './ui/UserAvatar'

/**
 * Top navigation header for the landing experience.
 */
export default function Header() {
  return (
    <header className="flex items-center justify-between border-b border-border-color bg-white px-6 py-2.5 shrink-0">
      <div className="flex items-center gap-10">
        <div className="flex items-center gap-2.5">
          <Logo />
          <h2 className="text-lg font-bold tracking-tight text-slate-900 uppercase italic">Mathpad</h2>
        </div>
        <nav className="hidden md:flex items-center gap-6">
          <a className="text-primary font-semibold text-sm" href="#">Documentation</a>
        </nav>
      </div>
      <div className="flex-1 max-w-xl mx-8">
        <SearchBar placeholder="Compute matrix determinants or solve systems..." />
      </div>
      <div className="flex items-center gap-4">
        <div className="flex gap-1">
          <IconButton icon="notifications" />
          <IconButton icon="settings" />
        </div>
        <div className="h-8 w-[1px] bg-border-color mx-1"></div>
        <div className="flex items-center gap-3 ml-2">
          <UserAvatar 
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuCiPdSzqA7b7pSM7J5PtlaeVYe_zxTFm0RXF77N-S3d5dGCaW3fS7DahL4Pn4HsWrK75GS9wk9bd3CaUAkuIz2R3QkkfuD5ix0s1bIvSmRSeUbDeT8BWu7YLdnEQlFDOnOueNo8YKH2vZrgtXStHCf4oqpSs8vnqcz_Gmy95Wy8hnvFK-EWdGSvlzkMSl6FdbEJS-10IZ9XcGbctoou9QRCiwT6jH7ihmURoCpBVAHZWlEL04sW6GsA8ANpzjQHAXNrrdry2ACBiGN6"
          />
        </div>
      </div>
    </header>
  )
}
