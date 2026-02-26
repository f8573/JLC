const HEAVY_ENDPOINT_TOKEN = (import.meta.env.VITE_HEAVY_ENDPOINT_TOKEN || '').trim()

/**
 * Add the heavy-endpoint API key header when configured.
 */
export function withHeavyAuthHeaders(headers = {}) {
  if (!HEAVY_ENDPOINT_TOKEN) {
    return headers
  }
  return {
    ...headers,
    'X-API-Key': HEAVY_ENDPOINT_TOKEN
  }
}
