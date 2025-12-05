/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        wildfire: {
          red: '#FF4B4B',
          dark: '#1E1E1E',
          light: '#F0F2F6',
        }
      }
    },
  },
  plugins: [],
}
