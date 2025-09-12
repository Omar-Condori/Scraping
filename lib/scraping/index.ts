import { scrapeHackerNews, saveHackerNewsArticles } from './hackerNews'

export async function runAllScraping(): Promise<void> {
  console.log('🚀 Iniciando scraping de fuentes estáticas...')
  
  try {
    // Scraping de Hacker News
    console.log('📰 Scraping Hacker News...')
    const hackerNewsArticles = await scrapeHackerNews()
    await saveHackerNewsArticles(hackerNewsArticles)
    console.log(`✅ Hacker News: ${hackerNewsArticles.length} artículos`)
    
    console.log('🎉 Scraping estático completado exitosamente!')
  } catch (error) {
    console.error('❌ Error durante el scraping:', error)
    throw error
  }
}

export { scrapeHackerNews, saveHackerNewsArticles } from './hackerNews'
