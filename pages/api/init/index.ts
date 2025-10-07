import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '../../lib/db'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Método no permitido' })
  }

  try {
    console.log('🚀 Inicializando proyecto de scraping...')
    
    // 1. Verificar conexión a la base de datos
    console.log('📊 Verificando conexión a la base de datos...')
    await prisma.$connect()
    console.log('✅ Conexión a la base de datos establecida')
    
    // 2. Crear categorías por defecto
    console.log('📂 Creando categorías por defecto...')
    const defaultCategories = [
      { name: 'Política', description: 'Noticias políticas y gubernamentales', color: '#3B82F6' },
      { name: 'Economía / Negocios', description: 'Noticias económicas y empresariales', color: '#10B981' },
      { name: 'Sociedad / Actualidad', description: 'Noticias de sociedad y actualidad', color: '#F59E0B' },
      { name: 'Internacional', description: 'Noticias internacionales', color: '#8B5CF6' },
      { name: 'Tecnología / Ciencia', description: 'Noticias de tecnología y ciencia', color: '#06B6D4' },
      { name: 'Cultura / Arte', description: 'Noticias culturales y artísticas', color: '#EC4899' },
      { name: 'Opinión', description: 'Artículos de opinión y editoriales', color: '#6B7280' },
      { name: 'Estilo de vida / Tendencias', description: 'Noticias de estilo de vida', color: '#84CC16' },
      { name: 'Clima y Medio ambiente', description: 'Noticias ambientales y climáticas', color: '#22C55E' }
    ]
    
    const createdCategories = []
    for (const category of defaultCategories) {
      try {
        const created = await prisma.category.upsert({
          where: { name: category.name },
          update: category,
          create: category
        })
        createdCategories.push(created)
        console.log(`✅ Categoría "${category.name}" inicializada`)
      } catch (error) {
        console.log(`⚠️ Categoría "${category.name}" ya existe`)
      }
    }
    
    // 3. Crear fuentes de scraping por defecto
    console.log('📰 Creando fuentes de scraping por defecto...')
    const defaultSources = [
      {
        name: 'RPP Noticias',
        url: 'https://rpp.pe/noticias',
        category: 'Sociedad / Actualidad',
        description: 'Noticias generales de RPP',
        maxItems: 20
      },
      {
        name: 'El Comercio',
        url: 'https://elcomercio.pe/noticias',
        category: 'Política',
        description: 'Noticias políticas de El Comercio',
        maxItems: 20
      },
      {
        name: 'Perú 21',
        url: 'https://peru21.pe/noticias',
        category: 'Internacional',
        description: 'Noticias internacionales de Perú 21',
        maxItems: 20
      },
      {
        name: 'La República',
        url: 'https://larepublica.pe/noticias',
        category: 'Sociedad / Actualidad',
        description: 'Noticias de sociedad de La República',
        maxItems: 20
      }
    ]
    
    const createdSources = []
    for (const source of defaultSources) {
      try {
        const created = await prisma.scrapingSource.upsert({
          where: { url: source.url },
          update: source,
          create: source
        })
        createdSources.push(created)
        console.log(`✅ Fuente "${source.name}" inicializada`)
      } catch (error) {
        console.log(`⚠️ Fuente "${source.name}" ya existe`)
      }
    }
    
    // 4. Verificar estadísticas
    console.log('📊 Verificando estadísticas...')
    const categoryCount = await prisma.category.count()
    const sourceCount = await prisma.scrapingSource.count()
    const articleCount = await prisma.article.count()
    
    console.log(`📈 Estadísticas del proyecto:`)
    console.log(`   - Categorías: ${categoryCount}`)
    console.log(`   - Fuentes: ${sourceCount}`)
    console.log(`   - Artículos: ${articleCount}`)
    
    res.status(200).json({
      success: true,
      message: 'Proyecto inicializado exitosamente',
      data: {
        categories: createdCategories.length,
        sources: createdSources.length,
        stats: {
          totalCategories: categoryCount,
          totalSources: sourceCount,
          totalArticles: articleCount
        }
      }
    })
    
  } catch (error: any) {
    console.error('❌ Error inicializando proyecto:', error)
    res.status(500).json({
      success: false,
      error: error.message,
      message: 'Error inicializando proyecto'
    })
  }
}
