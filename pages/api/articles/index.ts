import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '../../../lib/db'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    try {
      const { page = '1', limit = '5', search, category, source } = req.query
      
      const pageNum = parseInt(page as string)
      const limitNum = parseInt(limit as string)
      const skip = (pageNum - 1) * limitNum
      
      // Construir filtros
      const where: any = {}
      
      if (search) {
        where.OR = [
          { title: { contains: search as string, mode: 'insensitive' } },
          { content: { contains: search as string, mode: 'insensitive' } }
        ]
      }
      
      if (category) {
        where.category = category
      }
      
      if (source) {
        where.source = source
      }
      
      // Obtener artículos con paginación
      const [articles, total] = await Promise.all([
        prisma.article.findMany({
          where,
          orderBy: { scrapedAt: 'desc' },
          skip,
          take: limitNum
        }),
        prisma.article.count({ where })
      ])
      
      res.status(200).json({
        articles,
        pagination: {
          page: pageNum,
          limit: limitNum,
          total,
          pages: Math.ceil(total / limitNum)
        }
      })
    } catch (error) {
      console.error('Error fetching articles:', error)
      res.status(500).json({ error: 'Error interno del servidor' })
    }
  } else {
    res.setHeader('Allow', ['GET'])
    res.status(405).end(`Method ${req.method} Not Allowed`)
  }
}
