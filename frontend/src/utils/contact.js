// Read Vite env var for the contact endpoint. Do NOT put secrets here.
const API_CONTACT = import.meta.env.VITE_API_CONTACT || '/api/contact'

export async function sendContact(payload) {
  const res = await fetch(API_CONTACT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Contact request failed: ${res.status} ${text}`)
  }

  try {
    return await res.json()
  } catch {
    return { status: 'ok' }
  }
}
