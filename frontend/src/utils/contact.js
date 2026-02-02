export async function sendContact(payload) {
  const res = await fetch('/api/contact', {
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
  } catch (e) {
    return { status: 'ok' }
  }
}
