import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Served from https://<user>.github.io/BiomechPriorVAE-visuals/ in
  // production (the repo's actual current name); keep local dev/preview at
  // root.
  base: process.env.GITHUB_ACTIONS ? '/BiomechPriorVAE-visuals/' : '/',
  plugins: [react()],
})
