import { useState, useEffect } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import { 
  Newspaper, 
  BarChart3, 
  RefreshCw,
  Search,
  Filter,
  Settings,
  Globe
} from 'lucide-react'

interface Stats {
  totals: {
    articles: number
  }
  recent: {
    articles: number
  }
}

export default function Home() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [scraping, setScraping] = useState(false)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/stats')
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Error fetching stats:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleScraping = async () => {
    setScraping(true)
    try {
      const response = await fetch('/api/scrape', { method: 'POST' })
      if (response.ok) {
        await fetchStats() // Actualizar estadísticas
        alert('Scraping completado exitosamente!')
      } else {
        alert('Error durante el scraping')
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Error durante el scraping')
    } finally {
      setScraping(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>Scraping Dashboard - Next.js</title>
        <meta name="description" content="Dashboard de scraping con Next.js" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  Scraping Dashboard
                </h1>
                <p className="text-gray-600 mt-1">
                  Recolección automática de datos de múltiples fuentes
                </p>
              </div>
              <button
                onClick={handleScraping}
                disabled={scraping}
                className="btn-primary flex items-center gap-2 disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${scraping ? 'animate-spin' : ''}`} />
                {scraping ? 'Scraping...' : 'Ejecutar Scraping'}
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="card">
              <div className="flex items-center">
                <div className="p-3 bg-blue-100 rounded-lg">
                  <Newspaper className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Artículos</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {stats?.totals.articles || 0}
                  </p>
                  <p className="text-xs text-green-600">
                    +{stats?.recent.articles || 0} esta semana
                  </p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center">
                <div className="p-3 bg-green-100 rounded-lg">
                  <Globe className="w-6 h-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Fuentes Activas</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {stats?.totals.articles || 0}
                  </p>
                  <p className="text-xs text-blue-600">
                    Configurables
                  </p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center">
                <div className="p-3 bg-purple-100 rounded-lg">
                  <Settings className="w-6 h-6 text-purple-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Categorías</p>
                  <p className="text-2xl font-bold text-gray-900">
                    8
                  </p>
                  <p className="text-xs text-purple-600">
                    Disponibles
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Link href="/articles" className="group">
              <div className="card hover:shadow-md transition-shadow duration-200">
                <div className="flex items-center mb-4">
                  <Newspaper className="w-8 h-8 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-900 ml-3">
                    Artículos
                  </h3>
                </div>
                <p className="text-gray-600 text-sm">
                  Explora los últimos artículos de Hacker News
                </p>
                <div className="mt-4 flex items-center text-primary-600 group-hover:text-primary-700">
                  <span className="text-sm font-medium">Ver artículos</span>
                  <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </div>
            </Link>


            <Link href="/dashboard" className="group">
              <div className="card hover:shadow-md transition-shadow duration-200">
                <div className="flex items-center mb-4">
                  <BarChart3 className="w-8 h-8 text-orange-600" />
                  <h3 className="text-lg font-semibold text-gray-900 ml-3">
                    Dashboard
                  </h3>
                </div>
                <p className="text-gray-600 text-sm">
                  Estadísticas detalladas y análisis de datos
                </p>
                <div className="mt-4 flex items-center text-primary-600 group-hover:text-primary-700">
                  <span className="text-sm font-medium">Ver dashboard</span>
                  <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </div>
            </Link>

            <Link href="/admin" className="group">
              <div className="card hover:shadow-md transition-shadow duration-200">
                <div className="flex items-center mb-4">
                  <Settings className="w-8 h-8 text-indigo-600" />
                  <h3 className="text-lg font-semibold text-gray-900 ml-3">
                    Administración
                  </h3>
                </div>
                <p className="text-gray-600 text-sm">
                  Gestiona fuentes de scraping y categorías
                </p>
                <div className="mt-4 flex items-center text-primary-600 group-hover:text-primary-700">
                  <span className="text-sm font-medium">Configurar fuentes</span>
                  <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </div>
            </Link>
          </div>
        </main>
      </div>
    </>
  )
}
