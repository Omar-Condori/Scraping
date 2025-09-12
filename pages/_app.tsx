import type { AppProps } from 'next/app'
import '../styles/globals.css'
import { useEffect } from 'react'

export default function App({ Component, pageProps }: AppProps) {
  useEffect(() => {
    // Inicializar cron jobs solo en el servidor
    if (typeof window === 'undefined') {
      const { startCronJobs } = require('../lib/cron')
      startCronJobs()
    }
  }, [])

  return <Component {...pageProps} />
}
