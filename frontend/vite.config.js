import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import tailwindcss from 'tailwindcss'
import autoprefixer from 'autoprefixer'

export default defineConfig({
  plugins: [react()],
  css: {
    postcss: {
      plugins: [
        tailwindcss({ config: path.resolve(__dirname, 'tailwind.config.cjs') }),
        autoprefixer()
      ]
    }
  },
  server: {
    port: 5173,
    allowedHosts: ['lambdacompute.org'],
    fs: {
      allow: [path.resolve(__dirname)]
    },
    proxy: {
      '/api': 'http://localhost:8080'
    }
  }
})
