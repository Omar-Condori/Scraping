import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

const defaultCategories = [
  {
    name: 'Política',
    description: 'Nacional, Gobierno, Congreso, Elecciones',
    color: '#dc2626'
  },
  {
    name: 'Economía / Negocios',
    description: 'Finanzas, Mercados, Empleo, Emprendimiento',
    color: '#059669'
  },
  {
    name: 'Sociedad / Actualidad',
    description: 'Ciudad, Comunidad, Sucesos, Policiales',
    color: '#7c3aed'
  },
  {
    name: 'Internacional',
    description: 'Mundo, Latinoamérica',
    color: '#2563eb'
  },
  {
    name: 'Tecnología / Ciencia',
    description: 'Innovación, Medio ambiente, Salud',
    color: '#0891b2'
  },
  {
    name: 'Cultura / Arte',
    description: 'Música, Cine, Literatura',
    color: '#c2410c'
  },
  {
    name: 'Opinión',
    description: 'Editoriales, Columnistas, Cartas al director',
    color: '#be185d'
  },
  {
    name: 'Estilo de vida / Tendencias',
    description: 'Gastronomía, Viajes, Moda, Belleza',
    color: '#9333ea'
  },
  {
    name: 'Clima y Medio ambiente',
    description: 'Clima, Medio ambiente, Sostenibilidad',
    color: '#16a34a'
  }
]

async function initCategories() {
  try {
    console.log('🚀 Inicializando categorías por defecto...')
    
    for (const category of defaultCategories) {
      await prisma.category.upsert({
        where: { name: category.name },
        update: category,
        create: category
      })
      console.log(`✅ Categoría "${category.name}" inicializada`)
    }
    
    console.log('🎉 Todas las categorías han sido inicializadas!')
  } catch (error) {
    console.error('❌ Error inicializando categorías:', error)
  } finally {
    await prisma.$disconnect()
  }
}

initCategories()
