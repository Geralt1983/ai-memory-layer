'use client'

import { useState, useEffect, useRef } from 'react'
import { Zap, Menu, Shield, Loader2 } from 'lucide-react'
import CerebroChat from '../components/CerebroChat'

export default function Home() {
  return (
    <main className="h-screen overflow-hidden">
      <CerebroChat />
    </main>
  )
}
