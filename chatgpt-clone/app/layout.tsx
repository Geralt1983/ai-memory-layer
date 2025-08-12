import './globals.css'
import { type ReactNode } from 'react'

export const metadata = {
  title: 'Personal ChatGPT',
  description: 'Single-user ChatGPT-like app',
}

// Pre-hydration theme fix: sets html.dark before React mounts
const ThemeScript = () => (
  <script
    dangerouslySetInnerHTML={{
      __html: `
(function(){
  try {
    var saved = localStorage.getItem('theme');        // 'dark' | 'light' | null
    var dark = saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    var el = document.documentElement;
    if (dark) el.classList.add('dark'); else el.classList.remove('dark');
    el.style.colorScheme = dark ? 'dark' : 'light';
  } catch (e) {
    document.documentElement.classList.add('dark');
    document.documentElement.style.colorScheme = 'dark';
  }
})();`,
    }}
  />
)

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <ThemeScript />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  )
}