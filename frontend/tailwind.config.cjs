/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#7c3aed',
        'primary-hover': '#6d28d9',
        'purple-light': '#f5f3ff',
        'slate-gray': '#64748b',
        'border-color': '#e2e8f0',
        'primary-light': '#f5f3ff',
        'background-light': '#fcfaff',
        'surface-white': '#ffffff',
        'accent-purple': '#a855f7',
        'surface-accent': '#faf9ff',
        'border-elegant': '#e9e4ff'
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Space Grotesk', 'sans-serif'],
        math: ['Lora', 'serif']
      },
      borderRadius: {
        DEFAULT: '0.375rem',
        lg: '0.5rem',
        xl: '1rem',
        full: '9999px'
      }
    }
  },
  plugins: []
}