import * as cheerio from 'cheerio'
import axios from 'axios'
import { prisma } from '../db'

export interface HackerNewsArticle {
  title: string
  url: string
  author?: string
  points?: number
  comments?: number
  description?: string
  imageUrl?: string
}

export async function scrapeHackerNews(): Promise<HackerNewsArticle[]> {
  try {
    const response = await axios.get('https://news.ycombinator.com/')
    const $ = cheerio.load(response.data)
    
    const articles: HackerNewsArticle[] = []
    
    const articlePromises: Promise<void>[] = []
    
    $('.athing').each((index, element) => {
      if (index >= 20) return // Limitar a 20 artículos para evitar sobrecarga
      
      const $element = $(element)
      const titleElement = $element.find('.titleline > a')
      const title = titleElement.text().trim()
      const url = titleElement.attr('href')
      
      if (title && url) {
        const fullUrl = url.startsWith('http') ? url : `https://news.ycombinator.com/${url}`
        
        // Obtener puntos y comentarios del elemento siguiente
        const nextRow = $element.next()
        const pointsText = nextRow.find('.score').text()
        const points = pointsText ? parseInt(pointsText.replace(/\D/g, '')) : 0
        
        const commentsElement = nextRow.find('a[href*="item?id="]').last()
        const commentsText = commentsElement.text()
        const comments = commentsText ? parseInt(commentsText.replace(/\D/g, '')) : 0
        
        // Crear promesa para extraer metadatos del artículo
        const articlePromise = (async () => {
          let description = ''
          let imageUrl = ''
          
          try {
            // Solo hacer scraping del artículo original si no es de Hacker News
            if (!fullUrl.includes('news.ycombinator.com')) {
              const articleResponse = await axios.get(fullUrl, { 
                timeout: 5000,
                headers: {
                  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
              })
              const $article = cheerio.load(articleResponse.data)
              
              // Buscar meta description
              description = $article('meta[name="description"]').attr('content') || 
                           $article('meta[property="og:description"]').attr('content') || ''
              
              // Buscar imagen principal
              imageUrl = $article('meta[property="og:image"]').attr('content') || 
                        $article('meta[name="twitter:image"]').attr('content') || 
                        $article('img').first().attr('src') || ''
              
              // Limpiar URLs relativas
              if (imageUrl && !imageUrl.startsWith('http')) {
                const baseUrl = new URL(fullUrl).origin
                imageUrl = new URL(imageUrl, baseUrl).href
              }
            }
          } catch (error) {
            // Si falla el scraping del artículo, continuar sin imagen/descripción
            console.log(`No se pudo extraer metadatos de: ${fullUrl}`)
          }
          
          articles.push({
            title,
            url: fullUrl,
            points,
            comments,
            description: description.substring(0, 300) + (description.length > 300 ? '...' : ''),
            imageUrl: imageUrl || undefined
          })
        })()
        
        articlePromises.push(articlePromise)
      }
    })
    
    // Esperar a que todos los artículos se procesen
    await Promise.all(articlePromises)
    
    return articles
  } catch (error) {
    console.error('Error scraping Hacker News:', error)
    throw error
  }
}

export async function saveHackerNewsArticles(articles: HackerNewsArticle[]): Promise<void> {
  try {
    for (const article of articles) {
      await prisma.article.upsert({
        where: { url: article.url },
        update: {
          title: article.title,
          author: article.author,
          source: 'Hacker News',
          category: 'Tecnología / Ciencia',
          description: article.description,
          imageUrl: article.imageUrl,
          updatedAt: new Date()
        },
        create: {
          title: article.title,
          url: article.url,
          author: article.author,
          source: 'Hacker News',
          category: 'Tecnología / Ciencia',
          description: article.description,
          imageUrl: article.imageUrl,
          content: `Points: ${article.points || 0}, Comments: ${article.comments || 0}`
        }
      })
    }
    
    // Registrar el scraping
    await prisma.scrapingLog.create({
      data: {
        source: 'Hacker News',
        status: 'success',
        message: `Scraped ${articles.length} articles`,
        count: articles.length
      }
    })
  } catch (error) {
    console.error('Error saving Hacker News articles:', error)
    
    // Registrar error
    await prisma.scrapingLog.create({
      data: {
        source: 'Hacker News',
        status: 'error',
        message: error instanceof Error ? error.message : 'Unknown error',
        count: 0
      }
    })
    
    throw error
  }
}
