import { NextApiRequest, NextApiResponse } from 'next'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Método no permitido' })
  }

  try {
    // Obtener estadísticas de ML
    const totalArticles = await prisma.article.count()
    
    const articlesWithImages = await prisma.article.count({
      where: { imageUrl: { not: null } }
    })
    
    const articlesWithDescription = await prisma.article.count({
      where: { description: { not: null } }
    })
    
    const articlesWithAuthor = await prisma.article.count({
      where: { author: { not: null } }
    })

    // Estadísticas por categoría
    const categoryStats = await prisma.article.groupBy({
      by: ['category'],
      _count: { category: true },
      where: { category: { not: null } }
    })

    // Estadísticas por fuente
    const sourceStats = await prisma.article.groupBy({
      by: ['source'],
      _count: { source: true }
    })

    // Artículos por longitud de contenido
    const contentLengthStats = await prisma.$queryRaw`
      SELECT 
        CASE 
          WHEN LENGTH(content) < 200 THEN 'corto'
          WHEN LENGTH(content) < 500 THEN 'medio'
          ELSE 'largo'
        END as length_category,
        COUNT(*)::int as count
      FROM articles 
      WHERE content IS NOT NULL
      GROUP BY length_category
    `

    // Calcular métricas de calidad
    const qualityMetrics = {
      totalArticles,
      articlesWithImages,
      articlesWithDescription,
      articlesWithAuthor,
      imagePercentage: Math.round((articlesWithImages / totalArticles) * 100),
      descriptionPercentage: Math.round((articlesWithDescription / totalArticles) * 100),
      authorPercentage: Math.round((articlesWithAuthor / totalArticles) * 100)
    }

    res.status(200).json({
      success: true,
      data: {
        qualityMetrics,
        categoryStats: categoryStats.map(stat => ({
          category: stat.category,
          count: stat._count.category
        })),
        sourceStats: sourceStats.map(stat => ({
          source: stat.source,
          count: stat._count.source
        })),
        contentLengthStats
      }
    })

  } catch (error) {
    console.error('Error obteniendo estadísticas ML:', error)
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Error desconocido'
    })
  }
}
