import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '../../../lib/db'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    try {
      const sources = await prisma.scrapingSource.findMany({
        orderBy: { createdAt: 'desc' }
      })
      
      res.status(200).json({ sources })
    } catch (error) {
      console.error('Error fetching sources:', error)
      res.status(500).json({ error: 'Error interno del servidor' })
    }
  } else if (req.method === 'POST') {
    try {
      const { name, url, category, description, selector, maxItems } = req.body
      
      // Validar URL
      try {
        new URL(url)
      } catch {
        return res.status(400).json({ error: 'URL inválida' })
      }
      
      const source = await prisma.scrapingSource.create({
        data: {
          name,
          url,
          category,
          description,
          selector,
          maxItems: maxItems || 50
        }
      })
      
      res.status(201).json({ source })
    } catch (error) {
      console.error('Error creating source:', error)
      if (error instanceof Error && error.message.includes('Unique constraint')) {
        res.status(400).json({ error: 'Esta URL ya está registrada' })
      } else {
        res.status(500).json({ error: 'Error interno del servidor' })
      }
    }
  } else {
    res.setHeader('Allow', ['GET', 'POST'])
    res.status(405).end(`Method ${req.method} Not Allowed`)
  }
}
