import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Cerebro Apocalypse - X-Men Interface',
  description: 'Advanced AI system designed to assist the X-Men in their missions',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
