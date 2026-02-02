import React, { useState, useEffect } from 'react'
import { sendContact } from '../utils/contact'

/**
 * Unified top navigation header across all pages.
 * Accepts an optional inputValue for matrix input and onCompute callback.
 *
 * @param {Object} props
 * @param {string} [props.inputValue]
 * @param {(matrixString: string) => void} [props.onCompute]
 */
export default function Header({ inputValue: initialValue, onCompute }) {
  const [inputValue, setInputValue] = useState(initialValue || '')
  const [isContactOpen, setIsContactOpen] = useState(false)
  const [contactLoading, setContactLoading] = useState({ bug: false, inquiry: false })
  const [contactResult, setContactResult] = useState({ bug: null, inquiry: null })

  useEffect(() => {
    setInputValue(initialValue || '')
  }, [initialValue])

  useEffect(() => {
    if (!isContactOpen) return undefined
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [isContactOpen])

  function handleSubmit() {
    if (inputValue.trim()) {
      if (onCompute) {
        onCompute(inputValue.trim())
      } else {
        window.location.href = `/matrix=${encodeURIComponent(inputValue.trim())}/basic`
      }
    }
  }

  function handleCloseContact() {
    setIsContactOpen(false)
  }

  async function handleContactSubmitBug(event) {
    event.preventDefault()
    const form = new FormData(event.target)
    const payload = {
      formType: 'bug',
      reporterEmail: form.get('bug_email') || null,
      subject: form.get('bug_subject') || null,
      matrixContext: form.get('bug_matrix') || null,
      details: form.get('bug_details') || null,
      timestamp: new Date().toISOString()
    }
    try {
      setContactLoading((s) => ({ ...s, bug: true }))
      setContactResult((r) => ({ ...r, bug: null }))
      await sendContact(payload)
      setContactResult((r) => ({ ...r, bug: 'ok' }))
      event.target.reset()
      handleCloseContact()
    } catch (err) {
      console.error('Contact bug submit failed', err)
      setContactResult((r) => ({ ...r, bug: 'error' }))
    } finally {
      setContactLoading((s) => ({ ...s, bug: false }))
    }
  }

  async function handleContactSubmitInquiry(event) {
    event.preventDefault()
    const form = new FormData(event.target)
    const payload = {
      formType: 'inquiry',
      name: form.get('inquiry_name') || null,
      reporterEmail: form.get('inquiry_email') || null,
      message: form.get('inquiry_message') || null,
      timestamp: new Date().toISOString()
    }
    try {
      setContactLoading((s) => ({ ...s, inquiry: true }))
      setContactResult((r) => ({ ...r, inquiry: null }))
      await sendContact(payload)
      setContactResult((r) => ({ ...r, inquiry: 'ok' }))
      event.target.reset()
      handleCloseContact()
    } catch (err) {
      console.error('Contact inquiry submit failed', err)
      setContactResult((r) => ({ ...r, inquiry: 'error' }))
    } finally {
      setContactLoading((s) => ({ ...s, inquiry: false }))
    }
  }

  return (
    <header className="flex items-center justify-between border-b border-solid border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-6 py-3 sticky top-0 z-50 transition-colors duration-300">
      <div className="flex items-center gap-8">
        <a href="/" className="flex items-center gap-3">
          <div className="size-9 bg-primary rounded-xl flex items-center justify-center text-white shadow-lg shadow-primary/20">
            <span className="text-xl font-bold italic">Λ</span>
          </div>
          <h2 className="text-lg font-bold leading-tight tracking-tight hidden md:block text-slate-800 dark:text-white">ΛCompute</h2>
        </a>
        <div className="flex flex-col min-w-[320px] lg:min-w-[600px]">
          <div className="flex w-full items-stretch rounded-xl h-11 border border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-700 overflow-hidden focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all">
            <div className="text-slate-400 dark:text-slate-500 flex items-center justify-center pl-4">
              <span className="material-symbols-outlined text-[20px]">function</span>
            </div>
            <input
              className="w-full bg-transparent border-none focus:ring-0 text-sm font-medium px-4 placeholder:text-slate-400 dark:placeholder:text-slate-500 text-slate-900 dark:text-slate-100"
              placeholder="Enter Matrix (e.g. [[1,2],[3,4]])"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  handleSubmit()
                }
              }}
            />
            <button
              className="bg-primary text-white px-8 text-xs font-bold uppercase tracking-widest hover:bg-primary/90 transition-all active:scale-95"
              onClick={handleSubmit}
            >
              Compute
            </button>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <nav className="hidden xl:flex items-center gap-6">
          <button
            className="text-sm font-semibold text-slate-600 dark:text-slate-300 hover:text-primary transition-colors"
            type="button"
            onClick={() => setIsContactOpen(true)}
          >
            Contact Me
          </button>
          <a className="text-sm font-semibold text-slate-600 dark:text-slate-300 hover:text-primary transition-colors" href="/documentation">Documentation</a>
        </nav>
        <div className="h-8 w-[1px] bg-slate-200 dark:bg-slate-600 mx-2"></div>
        <a href="/settings" className="p-2 hover:bg-primary/5 dark:hover:bg-primary/20 rounded-lg transition-colors text-slate-500 dark:text-slate-400">
          <span className="material-symbols-outlined">settings</span>
        </a>
      </div>
      {isContactOpen && (
        <div
          className="fixed inset-0 z-[60] bg-slate-900/40 backdrop-blur-sm flex items-center justify-center px-4 py-8"
          role="dialog"
          aria-modal="true"
          aria-label="Contact support"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              handleCloseContact()
            }
          }}
        >
          <div className="w-full max-w-5xl bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/70 dark:border-slate-700 shadow-2xl shadow-slate-900/20 overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-700">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-primary font-semibold">Contact</p>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">Contact Support</h2>
                <p className="text-sm text-slate-500 dark:text-slate-400">Experiencing a computation error or just want to say hi? Select the appropriate channel below.</p>
              </div>
              <button
                className="size-9 rounded-lg border border-slate-200 dark:border-slate-600 text-slate-500 dark:text-slate-300 hover:text-primary hover:border-primary/40 transition-colors"
                type="button"
                onClick={handleCloseContact}
                aria-label="Close contact form"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 max-h-[70vh] overflow-y-auto">
              <div className="group relative flex flex-col bg-slate-50 dark:bg-slate-800/70 rounded-2xl border border-primary/20 hover:border-primary/50 transition-colors shadow-lg shadow-primary/5 p-6 overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary to-violet-400 rounded-t-2xl z-10"></div>
                <div className="flex items-start gap-4 mb-5">
                  <div className="size-11 rounded-xl bg-rose-50 dark:bg-rose-900/30 flex items-center justify-center text-rose-500">
                    <span className="material-symbols-outlined text-2xl">bug_report</span>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-slate-900 dark:text-white">Report a Bug</h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Found a glitch in the matrix? Help us fix it.</p>
                  </div>
                </div>
                <form
                  className="space-y-4 flex-1 flex flex-col"
                  onSubmit={handleContactSubmitBug}
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <label className="block">
                      <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2 block">Your Email</span>
                      <input
                        className="w-full h-11 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/60 focus:border-primary focus:ring-primary text-sm px-3"
                        placeholder="john@example.com"
                        type="email"
                        name="bug_email"
                      />
                    </label>
                    <label className="block">
                      <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2 block">Subject</span>
                      <input
                        className="w-full h-11 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/60 focus:border-primary focus:ring-primary text-sm px-3"
                        placeholder="e.g. Eigenvalue calculation error"
                        type="text"
                        name="bug_subject"
                      />
                    </label>
                  </div>
                  <label className="block">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-semibold text-primary">Matrix Input Context</span>
                      <span className="text-[11px] text-slate-400">Monospace • Syntax Only</span>
                    </div>
                    <div className="relative">
                      <textarea
                        className="w-full min-h-[110px] rounded-lg border border-primary/20 bg-primary/[0.03] dark:bg-primary/[0.07] dark:border-primary/30 focus:border-primary focus:ring-primary font-mono text-xs p-3 placeholder:text-primary/40 resize-y"
                        placeholder="{{1,2,3},{4,5,6},{7,8,9}}"
                        name="bug_matrix"
                      />
                      <div className="absolute right-3 bottom-3 text-primary/40 pointer-events-none">
                        <span className="material-symbols-outlined text-sm">code</span>
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-400 mt-2">Please paste the exact matrix syntax that triggered the error.</p>
                  </label>
                  <label className="block">
                    <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2 block">Additional Details</span>
                    <textarea
                      className="w-full min-h-[90px] rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/60 focus:border-primary focus:ring-primary text-sm p-3 resize-y"
                      placeholder="Describe what happened..."
                      name="bug_details"
                    />
                  </label>
                  <div className="pt-2 mt-auto">
                    <button
                      className="w-full h-11 bg-rose-500 hover:bg-rose-600 text-white text-sm font-semibold rounded-lg shadow-lg shadow-rose-500/20 transition-all flex items-center justify-center gap-2"
                      type="submit"
                      disabled={contactLoading.bug}
                    >
                      <span>{contactLoading.bug ? 'Submitting...' : 'Submit Bug Report'}</span>
                      <span className="material-symbols-outlined text-base">arrow_forward</span>
                    </button>
                  </div>
                </form>
              </div>
              <div className="group relative flex flex-col bg-slate-50 dark:bg-slate-800/70 rounded-2xl border border-primary/20 hover:border-primary/50 transition-colors shadow-lg shadow-primary/5 p-6 overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary/50 to-primary rounded-t-2xl z-10"></div>
                <div className="flex items-start gap-4 mb-5">
                  <div className="size-11 rounded-xl bg-primary/10 flex items-center justify-center text-primary">
                    <span className="material-symbols-outlined text-2xl">forum</span>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-slate-900 dark:text-white">General Inquiry</h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Partnerships, features, or general questions.</p>
                  </div>
                </div>
                <form
                  className="space-y-4 flex-1 flex flex-col"
                  onSubmit={handleContactSubmitInquiry}
                >
                  <label className="block">
                    <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2 block">Full Name</span>
                    <input
                      className="w-full h-11 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/60 focus:border-primary focus:ring-primary text-sm px-3"
                      placeholder="Jane Doe"
                      type="text"
                      name="inquiry_name"
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2 block">Your Email</span>
                    <input
                      className="w-full h-11 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/60 focus:border-primary focus:ring-primary text-sm px-3"
                      placeholder="jane@example.com"
                      type="email"
                      name="inquiry_email"
                    />
                  </label>
                  <label className="block flex-1">
                    <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2 block">Message</span>
                    <textarea
                      className="w-full h-full min-h-[160px] rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/60 focus:border-primary focus:ring-primary text-sm p-3 resize-y"
                      placeholder="How can we help you today?"
                      name="inquiry_message"
                    />
                  </label>
                  <div className="pt-2 mt-auto">
                    <button
                      className="w-full h-11 bg-primary hover:bg-primary/90 text-white text-sm font-semibold rounded-lg shadow-lg shadow-primary/20 transition-all flex items-center justify-center gap-2"
                      type="submit"
                      disabled={contactLoading.inquiry}
                    >
                      <span>{contactLoading.inquiry ? 'Sending...' : 'Send Message'}</span>
                      <span className="material-symbols-outlined text-base">send</span>
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}
    </header>
  )
}
