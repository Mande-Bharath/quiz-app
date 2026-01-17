import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'https://quiz-app-dleh.onrender.com', // Render URL
        changeOrigin: true,
        secure: true, // Required for HTTPS
      }
    }
  }
})
