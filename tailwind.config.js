// tailwind.config.js
module.exports = {
  content: [
    './frontend/templates/**/*.html',
    './frontend/static/js/**/*.js',
    './frontend/static/css/**/*.css',
  ],
  theme: {
    extend: {
      fontFamily: {
        outfit: ['Outfit', 'sans-serif'],
        heading: ['Open Sans', 'sans-serif'],
      },
      colors: {
        'primary-dark': '#0a2342',
        'primary': '#155eab',
        'primary-light': '#38bdf8',
        'highlight': '#00b8f4',
        'surface': 'rgba(255,255,255,0.08)',
        'border': '#1e3a5b',
        'text-high': '#ffffff',
        'text-med': '#f0f0f0',
        'text-muted': '#cfd8e3',
        'success': '#03c988',
        'warning': '#ff9800',
        'danger': '#e94f37',
      },
      boxShadow: {
        DEFAULT: '0 8px 20px rgba(0, 0, 0, 0.25)',
        md: '0 8px 20px rgba(0, 0, 0, 0.25)',
        lg: '0 20px 40px rgba(0, 0, 0, 0.35)',
      },
    },
  },
  plugins: [],
};
