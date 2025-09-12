import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '../../../lib/db'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    try {
      // Obtener estadísticas generales
      const [
        totalArticles,
        recentScrapingLogs,
        articlesBySource
      ] = await Promise.all([
        prisma.article.count(),
        prisma.scrapingLog.findMany({
          orderBy: { createdAt: 'desc' },
          take: 10
        }),
        prisma.article.groupBy({
          by: ['source'],
          _count: { source: true }
        })
      ])
      
      // Estadísticas de los últimos 7 días
      const sevenDaysAgo = new Date()
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7)
      
      const recentArticles = await prisma.article.count({
        where: { scrapedAt: { gte: sevenDaysAgo } }
      })
      
      res.status(200).json({
        totals: {
          articles: totalArticles
        },
        recent: {
          articles: recentArticles
        },
        breakdown: {
          articlesBySource
        },
        recentScrapingLogs
      })
    } catch (error) {
      console.error('Error fetching stats:', error)
      res.status(500).json({ error: 'Error interno del servidor' })
    }
  } else {
    res.setHeader('Allow', ['GET'])
    res.status(405).end(`Method ${req.method} Not Allowed`)
  }
}
