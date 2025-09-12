import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '../../../lib/db'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { id } = req.query

  if (req.method === 'PUT') {
    try {
      const { name, url, category, description, selector, maxItems, isActive } = req.body
      
      // Validar URL si se proporciona
      if (url) {
        try {
          new URL(url)
        } catch {
          return res.status(400).json({ error: 'URL inválida' })
        }
      }
      
      const source = await prisma.scrapingSource.update({
        where: { id: id as string },
        data: {
          ...(name && { name }),
          ...(url && { url }),
          ...(category && { category }),
          ...(description !== undefined && { description }),
          ...(selector !== undefined && { selector }),
          ...(maxItems && { maxItems }),
          ...(isActive !== undefined && { isActive })
        }
      })
      
      res.status(200).json({ source })
    } catch (error) {
      console.error('Error updating source:', error)
      res.status(500).json({ error: 'Error interno del servidor' })
    }
  } else if (req.method === 'DELETE') {
    try {
      await prisma.scrapingSource.delete({
        where: { id: id as string }
      })
      
      res.status(200).json({ message: 'Fuente eliminada exitosamente' })
    } catch (error) {
      console.error('Error deleting source:', error)
      res.status(500).json({ error: 'Error interno del servidor' })
    }
  } else {
    res.setHeader('Allow', ['PUT', 'DELETE'])
    res.status(405).end(`Method ${req.method} Not Allowed`)
  }
}
