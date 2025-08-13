/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'x-men-gold': '#ffd700',
        'x-men-blue': '#1e3a8a',
        'x-men-red': '#dc2626',
        'apocalypse-red': '#8b0000',
        'bg-primary': '#0a0a0a',
        'bg-secondary': 'rgba(20, 0, 0, 0.9)',
        'bg-tertiary': 'rgba(40, 0, 0, 0.8)',
        'text-muted': '#b0b0ff',
        'accent': '#ff6b35',
        'accent-secondary': '#4ecdc4',
        'success': '#45b7d1',
        'warning': '#ffd700',
        'error': '#ff4757',
      },
      fontFamily: {
        'apocalypse': ['Creepster', 'cursive'],
        'display': ['Special Elite', 'cursive'],
        'tech': ['Orbitron', 'monospace'],
      },
      boxShadow: {
        'comic': '4px 4px 0px rgba(0, 0, 0, 0.8)',
        'glow': '0 0 20px rgba(255, 215, 0, 0.6)',
        'dark': '0 8px 32px rgba(0, 0, 0, 0.9)',
      },
      animation: {
        'comic-slide': 'comicSlide 0.5s ease-out',
        'power-pulse': 'powerPulse 1.4s infinite',
        'mutant-glow': 'mutantGlow 3s ease-in-out infinite',
        'power-flow': 'powerFlow 3s ease-in-out infinite',
        'apocalypse-shift': 'apocalypseShift 20s ease-in-out infinite',
      },
      keyframes: {
        comicSlide: {
          'from': { opacity: '0', transform: 'translateY(20px) scale(0.95)' },
          'to': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        powerPulse: {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.7', transform: 'scale(1.05)' },
        },
        mutantGlow: {
          '0%, 100%': { boxShadow: '4px 4px 0px rgba(0, 0, 0, 0.8)' },
          '50%': { boxShadow: '0 0 20px rgba(255, 215, 0, 0.6)' },
        },
        powerFlow: {
          '0%, 100%': { opacity: '0.6' },
          '50%': { opacity: '1' },
        },
        apocalypseShift: {
          '0%, 100%': { transform: 'scale(1) rotate(0deg)', filter: 'hue-rotate(0deg)' },
          '50%': { transform: 'scale(1.05) rotate(1deg)', filter: 'hue-rotate(10deg)' },
        },
      },
    },
  },
  plugins: [],
}
