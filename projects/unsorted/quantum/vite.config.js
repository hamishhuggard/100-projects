import { defineConfig } from 'vite';

export default defineConfig({
  root: '', // Set this to the root directory to serve any subdirectory directly
  server: {
    fs: {
      allow: ['..'], // Allow accessing the parent directory
    },
  },
});
