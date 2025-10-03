import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '../../../lib/db'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    try {
      const categories = await prisma.category.findMany({
        where: { isActive: true },
        orderBy: { name: 'asc' }
      })
      
      res.status(200).json({ categories })
    } catch (error) {
      console.error('Error fetching categories:', error)
      res.status(500).json({ error: 'Error interno del servidor' })
    }
  } else if (req.method === 'POST') {
    try {
      const { name, description, color } = req.body
      
      // Validar que el nombre no esté vacío
      if (!name || name.trim() === '') {
        return res.status(400).json({ error: 'El nombre de la categoría es requerido' })
      }
      
      const category = await prisma.category.create({
        data: {
          name: name.trim(),
          description: description?.trim() || null,
          color: color || null,
          isActive: true
        }
      })
      
      res.status(201).json({ category })
    } catch (error: any) {
      console.error('Error creating category:', error)
      if (error.code === 'P2002' || (error.message && error.message.includes('Unique constraint'))) {
        res.status(400).json({ error: 'Esta categoría ya existe' })
      } else {
        res.status(500).json({ error: 'Error interno del servidor' })
      }
    }
  } else {
    res.setHeader('Allow', ['GET', 'POST'])
    res.status(405).end(`Method ${req.method} Not Allowed`)
  }
}
