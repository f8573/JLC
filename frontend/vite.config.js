import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    dedupe: ['react', 'react-dom'],
    alias: {
      'react$': path.resolve(__dirname, 'node_modules/react/index.js'),
      'react/jsx-runtime': path.resolve(__dirname, 'node_modules/react/jsx-runtime.js'),
      'react/jsx-dev-runtime': path.resolve(__dirname, 'node_modules/react/jsx-dev-runtime.js'),
      'react-dom$': path.resolve(__dirname, 'node_modules/react-dom/index.js'),
      'react-dom/client': path.resolve(__dirname, 'node_modules/react-dom/client.js')
    }
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react/jsx-runtime', 'react/jsx-dev-runtime'],
    force: true
  },
  server: {
    port: 5173,
    allowedHosts: ['lambdacompute.org'],
    proxy: {
      '/api': 'http://localhost:8080'
    }
  }
})
