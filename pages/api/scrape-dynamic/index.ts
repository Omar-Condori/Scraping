import { NextApiRequest, NextApiResponse } from 'next'
import { scrapeAllDynamicSources } from '../../../lib/scraping/dynamicScraper'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST') {
    try {
      console.log('🚀 Iniciando scraping dinámico manual...')
      await scrapeAllDynamicSources()
      
      res.status(200).json({ 
        message: 'Scraping dinámico completado exitosamente',
        timestamp: new Date().toISOString()
      })
    } catch (error) {
      console.error('Error en scraping dinámico manual:', error)
      res.status(500).json({ 
        error: 'Error durante el scraping dinámico',
        message: error instanceof Error ? error.message : 'Error desconocido'
      })
    }
  } else {
    res.setHeader('Allow', ['POST'])
    res.status(405).end(`Method ${req.method} Not Allowed`)
  }
}
