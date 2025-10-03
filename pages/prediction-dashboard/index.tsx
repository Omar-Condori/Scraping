import React, { useState, useEffect } from 'react'
import { 
  Brain, 
  Zap, 
  BarChart3, 
  Clock, 
  Target, 
  TrendingUp,
  CheckCircle,
  XCircle,
  Activity,
  Database
} from 'lucide-react'

interface PredictionResult {
  predicted_category: string
  confidence: number
  prediction_time: number
  timestamp: string
  probabilities?: { [key: string]: number }
}

interface ComparisonResult {
  article_id: number
  title: string
  actual_category: string
  predicted_category: string
  confidence: number
  is_correct: boolean
  prediction_time: number
}

interface PredictionStats {
  total_predictions: number
  average_confidence: number
  average_prediction_time: number
  category_distribution: { [key: string]: number }
}

export default function RealTimePredictionDashboard() {
  const [activeTab, setActiveTab] = useState('single')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>([])
  const [stats, setStats] = useState<PredictionStats | null>(null)
  const [accuracy, setAccuracy] = useState<number>(0)
  
  // Formulario para predicción individual
  const [formData, setFormData] = useState({
    title: '',
    content: '',
    description: '',
    return_probabilities: false
  })

  // Cargar estadísticas al montar el componente
  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stats' })
      })
      
      if (response.ok) {
        const data = await response.json()
        setStats(data.data)
      }
    } catch (error) {
      console.error('Error cargando estadísticas:', error)
    }
  }

  const handleSinglePrediction = async () => {
    if (!formData.title || !formData.content) {
      alert('Por favor, ingresa título y contenido')
      return
    }

    setLoading(true)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'predict',
          ...formData
        })
      })

      if (response.ok) {
        const data = await response.json()
        setResult(data.data)
        loadStats() // Recargar estadísticas
      } else {
        alert('Error en la predicción')
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Error en la predicción')
    } finally {
      setLoading(false)
    }
  }

  const handleComparePredictions = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'compare',
          limit: 20
        })
      })

      if (response.ok) {
        const data = await response.json()
        setComparisonResults(data.data.results)
        setAccuracy(data.data.accuracy)
        loadStats() // Recargar estadísticas
      } else {
        alert('Error en la comparación')
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Error en la comparación')
    } finally {
      setLoading(false)
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getConfidenceBg = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100'
    if (confidence >= 0.6) return 'bg-yellow-100'
    return 'bg-red-100'
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Brain className="w-12 h-12 text-purple-600 mr-4" />
            <h1 className="text-4xl font-bold text-gray-900">
              Sistema de Predicción en Tiempo Real
            </h1>
          </div>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Clasificación automática de categorías de noticias usando Red Neuronal Profunda
          </p>
        </div>

        {/* Tabs */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'single', name: 'Predicción Individual', icon: Target },
                { id: 'compare', name: 'Comparar Predicciones', icon: BarChart3 },
                { id: 'stats', name: 'Estadísticas', icon: TrendingUp }
              ].map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.id
                        ? 'border-purple-500 text-purple-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <Icon className="w-5 h-5 mr-2" />
                    {tab.name}
                  </button>
                )
              })}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'single' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Formulario */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                <Target className="w-6 h-6 mr-2 text-purple-600" />
                Predicción Individual
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Título del Artículo *
                  </label>
                  <input
                    type="text"
                    value={formData.title}
                    onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="Ingresa el título del artículo..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Contenido del Artículo *
                  </label>
                  <textarea
                    value={formData.content}
                    onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                    rows={6}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="Ingresa el contenido del artículo..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Descripción (Opcional)
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="Ingresa una descripción del artículo..."
                  />
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="return_probabilities"
                    checked={formData.return_probabilities}
                    onChange={(e) => setFormData({ ...formData, return_probabilities: e.target.checked })}
                    className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                  />
                  <label htmlFor="return_probabilities" className="ml-2 block text-sm text-gray-700">
                    Mostrar probabilidades para todas las categorías
                  </label>
                </div>

                <button
                  onClick={handleSinglePrediction}
                  disabled={loading}
                  className="w-full bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <Activity className="w-5 h-5 mr-2 animate-spin" />
                      Prediciendo...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5 mr-2" />
                      Predecir Categoría
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Resultado */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                <BarChart3 className="w-6 h-6 mr-2 text-green-600" />
                Resultado de la Predicción
              </h2>
              
              {result ? (
                <div className="space-y-4">
                  <div className={`p-4 rounded-lg ${getConfidenceBg(result.confidence)}`}>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">Categoría Predicha:</span>
                      <span className={`font-bold ${getConfidenceColor(result.confidence)}`}>
                        {result.predicted_category}
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-sm font-medium text-gray-700">Confianza:</span>
                      <span className={`font-bold ${getConfidenceColor(result.confidence)}`}>
                        {(result.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-sm font-medium text-gray-700">Tiempo de Predicción:</span>
                      <span className="font-bold text-gray-900">
                        {(result.prediction_time * 1000).toFixed(2)}ms
                      </span>
                    </div>
                  </div>

                  {result.probabilities && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-3">Probabilidades por Categoría:</h3>
                      <div className="space-y-2">
                        {Object.entries(result.probabilities)
                          .sort(([,a], [,b]) => b - a)
                          .map(([category, probability]) => (
                            <div key={category} className="flex items-center justify-between">
                              <span className="text-sm text-gray-700">{category}</span>
                              <div className="flex items-center">
                                <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                                  <div
                                    className="bg-purple-600 h-2 rounded-full"
                                    style={{ width: `${probability * 100}%` }}
                                  />
                                </div>
                                <span className="text-sm font-medium text-gray-900 w-12 text-right">
                                  {(probability * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center text-gray-500 py-8">
                  <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                  <p>Ingresa un artículo para ver la predicción</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'compare' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900 flex items-center">
                <BarChart3 className="w-6 h-6 mr-2 text-blue-600" />
                Comparación con Categorías Reales
              </h2>
              <button
                onClick={handleComparePredictions}
                disabled={loading}
                className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                {loading ? (
                  <>
                    <Activity className="w-5 h-5 mr-2 animate-spin" />
                    Comparando...
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-5 h-5 mr-2" />
                    Comparar Predicciones
                  </>
                )}
              </button>
            </div>

            {accuracy > 0 && (
              <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-lg font-semibold text-blue-900">Precisión del Modelo:</span>
                  <span className="text-2xl font-bold text-blue-600">
                    {(accuracy * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            )}

            {comparisonResults.length > 0 && (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Artículo
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Categoría Real
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Categoría Predicha
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confianza
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Resultado
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {comparisonResults.map((item, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 max-w-xs truncate">
                          {item.title}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {item.actual_category}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {item.predicted_category}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {(item.confidence * 100).toFixed(2)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          {item.is_correct ? (
                            <CheckCircle className="w-5 h-5 text-green-500" />
                          ) : (
                            <XCircle className="w-5 h-5 text-red-500" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats && (
              <>
                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center">
                    <Database className="w-8 h-8 text-blue-600" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-500">Total Predicciones</p>
                      <p className="text-2xl font-bold text-gray-900">{stats.total_predictions}</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center">
                    <TrendingUp className="w-8 h-8 text-green-600" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-500">Confianza Promedio</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {(stats.average_confidence * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center">
                    <Clock className="w-8 h-8 text-purple-600" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-500">Tiempo Promedio</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {(stats.average_prediction_time * 1000).toFixed(2)}ms
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <div className="flex items-center">
                    <BarChart3 className="w-8 h-8 text-orange-600" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-500">Categorías Únicas</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {Object.keys(stats.category_distribution).length}
                      </p>
                    </div>
                  </div>
                </div>
              </>
            )}

            {stats && stats.category_distribution && (
              <div className="md:col-span-2 lg:col-span-4 bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Distribución de Categorías Predichas</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(stats.category_distribution).map(([category, count]) => (
                    <div key={category} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm font-medium text-gray-700">{category}</span>
                      <span className="text-sm font-bold text-gray-900">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

