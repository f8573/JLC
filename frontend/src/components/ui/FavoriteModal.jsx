import React from 'react'

export default function FavoriteModal({ open, defaultName = '', onCancel, onSave }) {
  const [name, setName] = React.useState(defaultName)

  React.useEffect(() => {
    setName(defaultName || '')
  }, [defaultName, open])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onCancel}></div>
      <div className="relative bg-white rounded-lg shadow-lg w-full max-w-md p-6">
        <h3 className="text-lg font-bold mb-2">Save Favorite</h3>
        <p className="text-sm text-slate-500 mb-4">Give this matrix a friendly name for quick access.</p>
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full px-3 py-2 border border-slate-200 rounded mb-4"
          placeholder="Favorite name"
        />
        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="px-4 py-2 rounded bg-slate-100">Cancel</button>
          <button onClick={() => onSave(name)} className="px-4 py-2 rounded bg-primary text-white">Save</button>
        </div>
      </div>
    </div>
  )
}
