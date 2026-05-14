/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./app/frontend/**/*.{html,js,ts}",
    "./app/frontend/index.html"
  ],
  theme: {
    extend: {
      colors: {
        ethiopian: {
          green: "#078930",
          yellow: "#FCDD09",
          red: "#DA121A",
          blue: "#0F47AF"
        },
        slate: {
          850: "#151e2e",
          950: "#0b0f19"
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        amharic: ['"Noto Sans Ethiopic"', 'Nyala', '"Abyssinica SIL"', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
