import { useState, useEffect } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import { 
  ArrowLeft, 
  Plus, 
  Edit, 
  Trash2, 
  Play, 
  Pause,
  Globe,
  Settings,
  Filter
} from 'lucide-react'

interface ScrapingSource {
  id: string
  name: string
  url: string
  category: string
  description?: string
  isActive: boolean
  selector?: string
  maxItems: number
  createdAt: string
  updatedAt: string
}

interface Category {
  id: string
  name: string
  description?: string
  color?: string
  isActive: boolean
}

export default function AdminPage() {
  const [sources, setSources] = useState<ScrapingSource[]>([])
  const [categories, setCategories] = useState<Category[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [showCategoryForm, setShowCategoryForm] = useState(false)
  const [scraping, setScraping] = useState(false)

  // Form states
  const [formData, setFormData] = useState({
    name: '',
    url: '',
    category: '',
    description: '',
    selector: '',
    maxItems: 50
  })

  const [categoryFormData, setCategoryFormData] = useState({
    name: '',
    description: '',
    color: '#3b82f6'
  })

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [sourcesRes, categoriesRes] = await Promise.all([
        fetch('/api/sources'),
        fetch('/api/categories')
      ])
      
      const sourcesData = await sourcesRes.json()
      const categoriesData = await categoriesRes.json()
      
      setSources(sourcesData.sources || [])
      setCategories(categoriesData.categories || [])
    } catch (error) {
      console.error('Error fetching data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleAddSource = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const response = await fetch('/api/sources', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })

      if (response.ok) {
        await fetchData()
        setShowAddForm(false)
        setFormData({ name: '', url: '', category: '', description: '', selector: '', maxItems: 50 })
        alert('Fuente agregada exitosamente!')
      } else {
        const error = await response.json()
        alert(`Error: ${error.error}`)
      }
    } catch (error) {
      console.error('Error adding source:', error)
      alert('Error al agregar la fuente')
    }
  }

  const handleAddCategory = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const response = await fetch('/api/categories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(categoryFormData)
      })

      if (response.ok) {
        await fetchData()
        setShowCategoryForm(false)
        setCategoryFormData({ name: '', description: '', color: '#3b82f6' })
        alert('Categoría agregada exitosamente!')
      } else {
        const error = await response.json()
        alert(`Error: ${error.error}`)
      }
    } catch (error) {
      console.error('Error adding category:', error)
      alert('Error al agregar la categoría')
    }
  }

  const toggleSourceStatus = async (id: string, isActive: boolean) => {
    try {
      const response = await fetch(`/api/sources/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ isActive: !isActive })
      })

      if (response.ok) {
        await fetchData()
      }
    } catch (error) {
      console.error('Error toggling source:', error)
    }
  }

  const deleteSource = async (id: string) => {
    if (!confirm('¿Estás seguro de que quieres eliminar esta fuente?')) return

    try {
      const response = await fetch(`/api/sources/${id}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        await fetchData()
        alert('Fuente eliminada exitosamente!')
      }
    } catch (error) {
      console.error('Error deleting source:', error)
      alert('Error al eliminar la fuente')
    }
  }

  const handleDynamicScraping = async () => {
    setScraping(true)
    try {
      const response = await fetch('/api/scrape-dynamic', { method: 'POST' })
      if (response.ok) {
        alert('Scraping dinámico completado exitosamente!')
        await fetchData()
      } else {
        alert('Error durante el scraping dinámico')
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Error durante el scraping dinámico')
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
        <title>Administración - Scraping Dashboard</title>
        <meta name="description" content="Panel de administración para gestionar fuentes de scraping" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between py-6">
              <div className="flex items-center">
                <Link href="/" className="mr-4">
                  <ArrowLeft className="w-6 h-6 text-gray-600 hover:text-gray-900" />
                </Link>
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">Administración</h1>
                  <p className="text-gray-600 mt-1">
                    Gestiona fuentes de scraping y categorías
                  </p>
                </div>
              </div>
              <button
                onClick={handleDynamicScraping}
                disabled={scraping}
                className="btn-primary flex items-center gap-2 disabled:opacity-50"
              >
                <Play className={`w-4 h-4 ${scraping ? 'animate-spin' : ''}`} />
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
                  <Globe className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Fuentes Activas</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {sources.filter(s => s.isActive).length}
                  </p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center">
                <div className="p-3 bg-green-100 rounded-lg">
                  <Filter className="w-6 h-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Categorías</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {categories.length}
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
                  <p className="text-sm font-medium text-gray-600">Total Fuentes</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {sources.length}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 mb-8">
            <button
              onClick={() => setShowAddForm(true)}
              className="btn-primary flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Agregar Fuente
            </button>
            <button
              onClick={() => setShowCategoryForm(true)}
              className="btn-secondary flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Agregar Categoría
            </button>
          </div>

          {/* Sources Table */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Fuentes de Scraping
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Nombre
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      URL
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Categoría
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Estado
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Acciones
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {sources.map((source) => (
                    <tr key={source.id}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {source.name}
                          </div>
                          {source.description && (
                            <div className="text-sm text-gray-500">
                              {source.description}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-blue-600 hover:text-blue-800 truncate block max-w-xs"
                        >
                          {source.url}
                        </a>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="badge badge-primary">
                          {source.category}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button
                          onClick={() => toggleSourceStatus(source.id, source.isActive)}
                          className={`badge ${source.isActive ? 'badge-success' : 'badge-error'}`}
                        >
                          {source.isActive ? 'Activa' : 'Inactiva'}
                        </button>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <div className="flex gap-2">
                          <button
                            onClick={() => deleteSource(source.id)}
                            className="text-red-600 hover:text-red-900"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Add Source Modal */}
          {showAddForm && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg p-6 w-full max-w-md">
                <h3 className="text-lg font-semibold mb-4">Agregar Nueva Fuente</h3>
                <form onSubmit={handleAddSource}>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Nombre
                      </label>
                      <input
                        type="text"
                        required
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        className="input-field"
                        placeholder="Ej: Noticias Deportes"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        URL
                      </label>
                      <input
                        type="url"
                        required
                        value={formData.url}
                        onChange={(e) => setFormData({ ...formData, url: e.target.value })}
                        className="input-field"
                        placeholder="https://ejemplo.com/noticias"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Categoría
                      </label>
                      <select
                        required
                        value={formData.category}
                        onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                        className="input-field"
                      >
                        <option value="">Seleccionar categoría</option>
                        {categories.map((cat) => (
                          <option key={cat.id} value={cat.name}>
                            {cat.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Descripción
                      </label>
                      <textarea
                        value={formData.description}
                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                        className="input-field"
                        rows={3}
                        placeholder="Descripción opcional"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Máximo de artículos
                      </label>
                      <input
                        type="number"
                        value={formData.maxItems}
                        onChange={(e) => setFormData({ ...formData, maxItems: parseInt(e.target.value) })}
                        className="input-field"
                        min="1"
                        max="100"
                      />
                    </div>
                  </div>
                  <div className="flex gap-3 mt-6">
                    <button type="submit" className="btn-primary flex-1">
                      Agregar Fuente
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowAddForm(false)}
                      className="btn-secondary flex-1"
                    >
                      Cancelar
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {/* Add Category Modal */}
          {showCategoryForm && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg p-6 w-full max-w-md">
                <h3 className="text-lg font-semibold mb-4">Agregar Nueva Categoría</h3>
                <form onSubmit={handleAddCategory}>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Nombre
                      </label>
                      <input
                        type="text"
                        required
                        value={categoryFormData.name}
                        onChange={(e) => setCategoryFormData({ ...categoryFormData, name: e.target.value })}
                        className="input-field"
                        placeholder="Ej: Deportes, Farándula, Noticias"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Descripción
                      </label>
                      <textarea
                        value={categoryFormData.description}
                        onChange={(e) => setCategoryFormData({ ...categoryFormData, description: e.target.value })}
                        className="input-field"
                        rows={3}
                        placeholder="Descripción opcional"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Color
                      </label>
                      <input
                        type="color"
                        value={categoryFormData.color}
                        onChange={(e) => setCategoryFormData({ ...categoryFormData, color: e.target.value })}
                        className="w-full h-10 rounded border border-gray-300"
                      />
                    </div>
                  </div>
                  <div className="flex gap-3 mt-6">
                    <button type="submit" className="btn-primary flex-1">
                      Agregar Categoría
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowCategoryForm(false)}
                      className="btn-secondary flex-1"
                    >
                      Cancelar
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}
        </main>
      </div>
    </>
  )
}
