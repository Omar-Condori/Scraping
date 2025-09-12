import { useState, useEffect } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import { ArrowLeft, ExternalLink, Calendar, User } from 'lucide-react'
import SearchBar from '../../components/SearchBar'
import FilterBar from '../../components/FilterBar'
import Pagination from '../../components/Pagination'
import { format } from 'date-fns'
import { es } from 'date-fns/locale'

interface Article {
  id: string
  title: string
  url: string
  author?: string
  source: string
  category?: string
  description?: string
  imageUrl?: string
  publishedAt?: string
  scrapedAt: string
}

interface ArticlesResponse {
  articles: Article[]
  pagination: {
    page: number
    limit: number
    total: number
    pages: number
  }
}

export default function ArticlesPage() {
  const [articles, setArticles] = useState<Article[]>([])
  const [loading, setLoading] = useState(true)
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 5,
    total: 0,
    pages: 0
  })
  const [searchQuery, setSearchQuery] = useState('')
  const [filters, setFilters] = useState({
    category: '',
    source: ''
  })
  const [categories, setCategories] = useState<Array<{value: string, label: string}>>([])
  const [sources, setSources] = useState<Array<{value: string, label: string}>>([])

  useEffect(() => {
    fetchArticles()
    fetchCategoriesAndSources()
  }, [pagination.page, searchQuery, filters])

  const fetchCategoriesAndSources = async () => {
    try {
      // Obtener categorías
      const categoriesResponse = await fetch('/api/categories')
      const categoriesData = await categoriesResponse.json()
      
      // Obtener fuentes
      const sourcesResponse = await fetch('/api/sources')
      const sourcesData = await sourcesResponse.json()
      
      // Obtener conteos de artículos por categoría
      const articlesResponse = await fetch('/api/articles?limit=1000')
      const articlesData = await articlesResponse.json()
      
      // Contar artículos por categoría
      const categoryCounts: {[key: string]: number} = {}
      articlesData.articles.forEach((article: Article) => {
        if (article.category) {
          categoryCounts[article.category] = (categoryCounts[article.category] || 0) + 1
        }
      })
      
      // Contar artículos por fuente
      const sourceCounts: {[key: string]: number} = {}
      articlesData.articles.forEach((article: Article) => {
        sourceCounts[article.source] = (sourceCounts[article.source] || 0) + 1
      })
      
      // Formatear categorías con conteos
      const formattedCategories = categoriesData.categories
        .filter((cat: any) => categoryCounts[cat.name] > 0)
        .map((cat: any) => ({
          value: cat.name,
          label: `${cat.name} (${categoryCounts[cat.name] || 0})`
        }))
      
      // Formatear fuentes con conteos
      const formattedSources = sourcesData.sources
        .filter((src: any) => sourceCounts[src.name] > 0)
        .map((src: any) => ({
          value: src.name,
          label: `${src.name} (${sourceCounts[src.name] || 0})`
        }))
      
      setCategories(formattedCategories)
      setSources(formattedSources)
    } catch (error) {
      console.error('Error fetching categories and sources:', error)
    }
  }

  const fetchArticles = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({
        page: pagination.page.toString(),
        limit: pagination.limit.toString(),
        ...(searchQuery && { search: searchQuery }),
        ...(filters.category && { category: filters.category }),
        ...(filters.source && { source: filters.source })
      })

      const response = await fetch(`/api/articles?${params}`)
      const data: ArticlesResponse = await response.json()
      
      setArticles(data.articles)
      setPagination(data.pagination)
    } catch (error) {
      console.error('Error fetching articles:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    setPagination(prev => ({ ...prev, page: 1 }))
  }

  const handleFilterChange = (filterKey: string, value: string) => {
    setFilters(prev => ({ ...prev, [filterKey]: value }))
    setPagination(prev => ({ ...prev, page: 1 }))
  }

  const handlePageChange = (page: number) => {
    setPagination(prev => ({ ...prev, page }))
  }

  const filterOptions = {
    category: categories,
    source: sources
  }

  return (
    <>
      <Head>
        <title>Artículos - Scraping Dashboard</title>
        <meta name="description" content="Explora los últimos artículos" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center py-6">
              <Link href="/" className="mr-4">
                <ArrowLeft className="w-6 h-6 text-gray-600 hover:text-gray-900" />
              </Link>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Artículos</h1>
                <p className="text-gray-600 mt-1">
                  {pagination.total} artículos encontrados
                </p>
              </div>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Search and Filters */}
          <div className="mb-8">
            <div className="flex flex-col lg:flex-row gap-4 mb-4">
              <div className="flex-1">
                <SearchBar
                  onSearch={handleSearch}
                  placeholder="Buscar artículos..."
                />
              </div>
              <FilterBar
                filters={filterOptions}
                onFilterChange={handleFilterChange}
              />
            </div>
          </div>

          {/* Articles List */}
          {loading ? (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
            </div>
          ) : articles.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-500 text-lg">No se encontraron artículos</p>
            </div>
          ) : (
            <div className="space-y-6">
              {articles.map((article) => (
                <div key={article.id} className="card">
                  <div className="flex gap-6">
                    {/* Imagen del artículo */}
                    {article.imageUrl && (
                      <div className="flex-shrink-0">
                        <img
                          src={article.imageUrl}
                          alt={article.title}
                          className="w-32 h-24 object-cover rounded-lg"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none'
                          }}
                        />
                      </div>
                    )}
                    
                    {/* Contenido del artículo */}
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold text-gray-900 mb-2 line-clamp-2">
                            {article.title}
                          </h3>
                          
                          {/* Descripción del artículo */}
                          {article.description && (
                            <p className="text-gray-600 mb-4 line-clamp-3">
                              {article.description}
                            </p>
                          )}
                          
                          <div className="flex items-center gap-4 text-sm text-gray-600 mb-4">
                            {article.author && (
                              <div className="flex items-center gap-1">
                                <User className="w-4 h-4" />
                                <span>{article.author}</span>
                              </div>
                            )}
                            <div className="flex items-center gap-1">
                              <Calendar className="w-4 h-4" />
                              <span>
                                {format(new Date(article.scrapedAt), 'dd MMM yyyy', { locale: es })}
                              </span>
                            </div>
                            <span className="badge badge-primary">
                              {article.source}
                            </span>
                            {article.category && (
                              <span className="badge badge-secondary">
                                {article.category}
                              </span>
                            )}
                          </div>
                        </div>
                        
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-4 p-2 text-gray-400 hover:text-primary-600 transition-colors"
                        >
                          <ExternalLink className="w-5 h-5" />
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Pagination */}
          {pagination.pages > 1 && (
            <div className="mt-8">
              <Pagination
                currentPage={pagination.page}
                totalPages={pagination.pages}
                onPageChange={handlePageChange}
              />
            </div>
          )}
        </main>
      </div>
    </>
  )
}
