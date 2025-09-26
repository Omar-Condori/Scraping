import { NextApiRequest, NextApiResponse } from 'next'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Método no permitido' })
  }

  const { articleId, predictedCategory } = req.body

  try {
    // Actualizar categoría predicha por ML
    const updatedArticle = await prisma.article.update({
      where: { id: articleId },
      data: { 
        category: predictedCategory,
        updatedAt: new Date()
      }
    })

    res.status(200).json({
      success: true,
      data: updatedArticle,
      message: `Categoría actualizada a: ${predictedCategory}`
    })

  } catch (error) {
    console.error('Error actualizando categoría:', error)
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Error desconocido'
    })
  }
}
