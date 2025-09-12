import { NextApiRequest, NextApiResponse } from 'next'
import { runAllScraping } from '../../../lib/scraping'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST') {
    try {
      console.log('🚀 Iniciando scraping manual...')
      await runAllScraping()
      
      res.status(200).json({ 
        message: 'Scraping completado exitosamente',
        timestamp: new Date().toISOString()
      })
    } catch (error) {
      console.error('Error en scraping manual:', error)
      res.status(500).json({ 
        error: 'Error durante el scraping',
        message: error instanceof Error ? error.message : 'Error desconocido'
      })
    }
  } else {
    res.setHeader('Allow', ['POST'])
    res.status(405).end(`Method ${req.method} Not Allowed`)
  }
}
