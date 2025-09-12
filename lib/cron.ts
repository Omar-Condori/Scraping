import cron from 'node-cron'
import { runAllScraping } from './scraping'
import { scrapeAllDynamicSources } from './scraping/dynamicScraper'

let isScrapingRunning = false

export function startCronJobs(): void {
  console.log('🕐 Iniciando cron jobs...')
  
  // Ejecutar scraping cada hora (minuto 0 de cada hora)
  cron.schedule('0 * * * *', async () => {
    if (isScrapingRunning) {
      console.log('⏳ Scraping ya en ejecución, saltando...')
      return
    }
    
    console.log('🔄 Ejecutando scraping programado...')
    isScrapingRunning = true
    
    try {
      // Ejecutar scraping estático (fuentes predefinidas)
      await runAllScraping()
      console.log('✅ Scraping estático completado')
      
      // Ejecutar scraping dinámico (fuentes personalizadas)
      await scrapeAllDynamicSources()
      console.log('✅ Scraping dinámico completado')
      
      console.log('✅ Scraping programado completado')
    } catch (error) {
      console.error('❌ Error en scraping programado:', error)
    } finally {
      isScrapingRunning = false
    }
  }, {
    scheduled: true,
    timezone: "America/Mexico_City"
  })
  
  // Ejecutar scraping cada 6 horas (opcional, para más frecuencia)
  cron.schedule('0 */6 * * *', async () => {
    if (isScrapingRunning) {
      console.log('⏳ Scraping ya en ejecución, saltando...')
      return
    }
    
    console.log('🔄 Ejecutando scraping programado (6h)...')
    isScrapingRunning = true
    
    try {
      // Ejecutar scraping estático (fuentes predefinidas)
      await runAllScraping()
      console.log('✅ Scraping estático (6h) completado')
      
      // Ejecutar scraping dinámico (fuentes personalizadas)
      await scrapeAllDynamicSources()
      console.log('✅ Scraping dinámico (6h) completado')
      
      console.log('✅ Scraping programado (6h) completado')
    } catch (error) {
      console.error('❌ Error en scraping programado (6h):', error)
    } finally {
      isScrapingRunning = false
    }
  }, {
    scheduled: true,
    timezone: "America/Mexico_City"
  })
  
  console.log('✅ Cron jobs iniciados correctamente')
}

export function stopCronJobs(): void {
  console.log('🛑 Deteniendo cron jobs...')
  cron.getTasks().forEach(task => task.destroy())
  console.log('✅ Cron jobs detenidos')
}
