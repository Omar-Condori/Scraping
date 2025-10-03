import * as cheerio from 'cheerio'
import axios from 'axios'
import { prisma } from '../db'

export interface DynamicArticle {
  title: string
  url: string
  content?: string
  description?: string
  author?: string
  source: string
  category: string
  imageUrl?: string
  publishedAt?: Date
}

export interface ScrapingSource {
  id: string
  name: string
  url: string
  category: string
  description?: string | null
  isActive: boolean
  selector?: string | null
  maxItems: number
}

export async function scrapeDynamicSource(source: ScrapingSource): Promise<DynamicArticle[]> {
  try {
    console.log(`🕷️ Scraping ${source.name} (${source.url})`)
    
    const response = await axios.get(source.url, {
      timeout: 30000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      }
    })
    
    const $ = cheerio.load(response.data)
    const articles: DynamicArticle[] = []
    
    // Selectores por defecto para diferentes tipos de sitios
    const defaultSelectors = {
      // Para sitios de noticias generales
      news: {
        title: 'h1, h2, h3, .title, .headline, [class*="title"], [class*="headline"]',
        link: 'a[href*="http"]',
        content: 'p, .content, .summary, .excerpt'
      },
      // Para blogs
      blog: {
        title: 'h1, h2, .post-title, .entry-title',
        link: 'a[href*="http"]',
        content: '.post-content, .entry-content, .content'
      },
      // Para agregadores de noticias
      aggregator: {
        title: '.title, .headline, h3, h4',
        link: 'a[href*="http"]',
        content: '.summary, .excerpt, p'
      }
    }
    
    // Selectores específicos para sitios peruanos
    const siteSelectors = {
      'rpp.pe': {
        title: '.story-item__title, .story-item__headline, h2, h3, .title, .news-item__title, .story__title',
        link: '.story-item__link, a[href*="/noticias/"], a[href*="/politica/"], .news-item__link, .story__link',
        content: '.story-item__summary, .story-item__excerpt, p, .news-item__summary, .story__summary',
        image: '.story-item__image img, .news-item__image img, img[src*="rpp"], .story-item img, .story__image img, img[src*="s.rpp-noticias.io"], img[src*="s2.rpp-noticias.io"]'
      },
      'peru21.pe': {
        title: '.story-item__title, .story-item__headline, h2, h3, .title, .news-item__title, .story__title',
        link: '.story-item__link, a[href*="/noticias/"], a[href*="/politica/"], .news-item__link, .story__link',
        content: '.story-item__summary, .story-item__excerpt, p, .news-item__summary, .story__summary',
        image: '.story-item__image img, .news-item__image img, img[src*="peru21"], .story-item img, .story__image img, img[src*="peru21.pe"], img[src*="cdn.peru21.pe"]'
      },
      'elcomercio.pe': {
        title: '.story-item__title, .story-item__headline, h2, h3, .title, .news-item__title',
        link: '.story-item__link, a[href*="/noticias/"], a[href*="/politica/"], .news-item__link',
        content: '.story-item__summary, .story-item__excerpt, p, .news-item__summary',
        image: '.story-item__image img, .news-item__image img, img[src*="elcomercio"], .story-item img'
      },
      'larepublica.pe': {
        title: '.story-item__title, .story-item__headline, h2, h3, .title, .news-item__title',
        link: '.story-item__link, a[href*="/noticias/"], a[href*="/politica/"], .news-item__link',
        content: '.story-item__summary, .story-item__excerpt, p, .news-item__summary',
        image: '.story-item__image img, .news-item__image img, img[src*="larepublica"], .story-item img'
      }
    }
    
    // Determinar el tipo de selector a usar
    let selectors = defaultSelectors.news
    
    // Usar selectores específicos para sitios peruanos
    if (source.url.includes('rpp.pe')) {
      selectors = siteSelectors['rpp.pe']
    } else if (source.url.includes('peru21.pe')) {
      selectors = siteSelectors['peru21.pe']
    } else if (source.url.includes('elcomercio.pe')) {
      selectors = siteSelectors['elcomercio.pe']
    } else if (source.url.includes('larepublica.pe')) {
      selectors = siteSelectors['larepublica.pe']
    } else if (source.url.includes('blog') || source.url.includes('wordpress')) {
      selectors = defaultSelectors.blog
    } else if (source.url.includes('reddit') || source.url.includes('hackernews')) {
      selectors = defaultSelectors.aggregator
    }
    
    // Usar selector personalizado si está definido
    if (source.selector) {
      try {
        const customSelector = JSON.parse(source.selector)
        selectors = { ...selectors, ...customSelector }
      } catch (error) {
        console.warn(`Invalid custom selector for ${source.name}:`, error)
      }
    }
    
    // Extraer artículos
    $(selectors.title).each((index, element) => {
      if (index >= source.maxItems) return
      
      const $element = $(element)
      const title = $element.text().trim()
      
      if (title && title.length > 10) {
        // Buscar el enlace asociado
        let url = ''
        let $link = $element.find('a').first()
        
        if ($link.length === 0) {
          $link = $element.parent().find('a').first()
        }
        
        if ($link.length === 0) {
          $link = $element.closest('a') as any
        }
        
        if ($link.length > 0) {
          url = $link.attr('href') || ''
          if (url && !url.startsWith('http')) {
            if (url.startsWith('/')) {
              url = new URL(source.url).origin + url
            } else {
              url = new URL(url, source.url).href
            }
          }
        }
        
        // Extraer contenido/descripción si está disponible
        let content = ''
        let description = ''
        
        // Buscar descripción en el elemento actual
        const $content = $element.find(selectors.content).first()
        if ($content.length > 0) {
          content = $content.text().trim()
        }
        
        // Si no se encontró, buscar en elementos hermanos
        if (!content) {
          const $siblingContent = $element.siblings().find(selectors.content).first()
          if ($siblingContent.length > 0) {
            content = $siblingContent.text().trim()
          }
        }
        
        // Si no se encontró, buscar en el contenedor padre
        if (!content) {
          const $parentContent = $element.closest('div, article, section').find(selectors.content).first()
          if ($parentContent.length > 0) {
            content = $parentContent.text().trim()
          }
        }
        
        // Buscar descripción en meta tags
        if (!content) {
          const $metaDesc = $('meta[name="description"]').attr('content')
          if ($metaDesc) {
            content = $metaDesc.trim()
          }
        }
        
        // Limitar la longitud y usar como descripción
        if (content) {
          description = content.substring(0, 300)
          content = content.substring(0, 500)
        }
        
        // Extraer imagen si está disponible
        let imageUrl = ''
        if ((selectors as any).image) {
          const $image = $element.find((selectors as any).image).first()
          if ($image.length > 0) {
            imageUrl = $image.attr('src') || $image.attr('data-src') || ''
            if (imageUrl && !imageUrl.startsWith('http')) {
              if (imageUrl.startsWith('/')) {
                imageUrl = new URL(source.url).origin + imageUrl
              } else {
                imageUrl = new URL(imageUrl, source.url).href
              }
            }
          }
        }
        
        // Si no se encontró imagen en el elemento, buscar en el contenedor
        if (!imageUrl && (selectors as any).image) {
          const $containerImage = $element.closest('div, article, section').find((selectors as any).image).first()
          if ($containerImage.length > 0) {
            imageUrl = $containerImage.attr('src') || $containerImage.attr('data-src') || ''
            if (imageUrl && !imageUrl.startsWith('http')) {
              if (imageUrl.startsWith('/')) {
                imageUrl = new URL(source.url).origin + imageUrl
              } else {
                imageUrl = new URL(imageUrl, source.url).href
              }
            }
          }
        }
        
        if (title && url) {
          articles.push({
            title,
            url,
            content: content || undefined,
            description: description || undefined,
            imageUrl: imageUrl || undefined,
            source: source.name,
            category: source.category,
            publishedAt: new Date()
          })
        }
      }
    })
    
    console.log(`✅ ${source.name}: ${articles.length} artículos extraídos`)
    return articles
    
  } catch (error) {
    console.error(`❌ Error scraping ${source.name}:`, error)
    throw error
  }
}

export async function saveDynamicArticles(articles: DynamicArticle[]): Promise<void> {
  try {
    for (const article of articles) {
      await prisma.article.upsert({
        where: { url: article.url },
        update: {
          title: article.title,
          content: article.content,
          description: article.description,
          author: article.author,
          source: article.source,
          category: article.category,
          imageUrl: article.imageUrl,
          publishedAt: article.publishedAt,
          updatedAt: new Date()
        },
        create: {
          title: article.title,
          url: article.url,
          content: article.content,
          description: article.description,
          author: article.author,
          source: article.source,
          category: article.category,
          imageUrl: article.imageUrl,
          publishedAt: article.publishedAt
        }
      })
    }
    
    // Registrar el scraping
    if (articles.length > 0) {
      await prisma.scrapingLog.create({
        data: {
          source: articles[0].source,
          status: 'success',
          message: `Scraped ${articles.length} articles`,
          count: articles.length
        }
      })
    }
  } catch (error) {
    console.error('Error saving dynamic articles:', error)
    
    // Registrar error
    if (articles.length > 0) {
      await prisma.scrapingLog.create({
        data: {
          source: articles[0].source,
          status: 'error',
          message: error instanceof Error ? error.message : 'Unknown error',
          count: 0
        }
      })
    }
    
    throw error
  }
}

export async function scrapeAllDynamicSources(): Promise<void> {
  try {
    console.log('🚀 Iniciando scraping dinámico...')
    
    const sources = await prisma.scrapingSource.findMany({
      where: { isActive: true }
    })
    
    if (sources.length === 0) {
      console.log('⚠️ No hay fuentes activas configuradas')
      return
    }
    
    for (const source of sources) {
      try {
        const articles = await scrapeDynamicSource(source)
        await saveDynamicArticles(articles)
      } catch (error) {
        console.error(`Error processing source ${source.name}:`, error)
        
        // Registrar error individual
        await prisma.scrapingLog.create({
          data: {
            source: source.name,
            status: 'error',
            message: error instanceof Error ? error.message : 'Unknown error',
            count: 0
          }
        })
      }
    }
    
    console.log('🎉 Scraping dinámico completado!')
  } catch (error) {
    console.error('❌ Error en scraping dinámico:', error)
    throw error
  }
}
