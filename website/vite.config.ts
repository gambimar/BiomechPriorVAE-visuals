import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Served from https://<user>.github.io/biomechpriorVAE/ in production;
  // keep local dev/preview at root.
  base: process.env.GITHUB_ACTIONS ? '/biomechpriorVAE/' : '/',
  plugins: [react()],
})
