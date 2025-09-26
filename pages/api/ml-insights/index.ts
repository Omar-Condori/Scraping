import { NextApiRequest, NextApiResponse } from 'next'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Método no permitido' })
  }

  try {
    // Obtener artículos sin categoría para clasificación automática
    const uncategorizedArticles = await prisma.article.findMany({
      where: {
        OR: [
          { category: null },
          { category: 'Noticias' }
        ],
        content: { not: null }
      },
      select: {
        id: true,
        title: true,
        content: true,
        source: true,
        createdAt: true
      },
      take: 50,
      orderBy: { createdAt: 'desc' }
    })

    // Obtener artículos duplicados potenciales (mismos títulos)
    const duplicateCandidates = await prisma.$queryRaw`
      SELECT 
        title,
        COUNT(*)::int as count,
        ARRAY_AGG(id::text) as article_ids
      FROM articles 
      WHERE title IS NOT NULL
      GROUP BY title
      HAVING COUNT(*) > 1
      ORDER BY count DESC
      LIMIT 20
    `

    // Obtener artículos de baja calidad (sin imagen, descripción corta, etc.)
    const lowQualityArticles = await prisma.article.findMany({
      where: {
        AND: [
          { imageUrl: null },
          { description: null },
          { content: { not: null } }
        ]
      },
      select: {
        id: true,
        title: true,
        content: true,
        source: true,
        createdAt: true
      },
      take: 20,
      orderBy: { createdAt: 'desc' }
    })

    res.status(200).json({
      success: true,
      data: {
        uncategorizedArticles,
        duplicateCandidates,
        lowQualityArticles,
        summary: {
          uncategorized: uncategorizedArticles.length,
          duplicates: Array.isArray(duplicateCandidates) ? duplicateCandidates.length : 0,
          lowQuality: lowQualityArticles.length
        }
      }
    })

  } catch (error) {
    console.error('Error obteniendo artículos para análisis:', error)
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Error desconocido'
    })
  }
}
