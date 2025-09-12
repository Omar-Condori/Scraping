# 🚀 Next.js Scraping Dashboard

Un dashboard completo de scraping construido con Next.js, que recolecta datos de múltiples fuentes y los presenta en una interfaz moderna y responsiva.

## ✨ Características

- **Scraping Automático**: Recolección de datos de Hacker News, Quotes y Books
- **Base de Datos**: PostgreSQL con Prisma ORM
- **Cron Jobs**: Actualización automática cada hora
- **Frontend Moderno**: Interfaz con TailwindCSS y componentes reutilizables
- **API RESTful**: Endpoints para consultar datos
- **Dashboard**: Estadísticas y análisis de datos
- **Deployment**: Preparado para Vercel y Docker

## 🛠️ Tecnologías

- **Frontend**: Next.js 14, React, TypeScript, TailwindCSS
- **Backend**: Next.js API Routes, Prisma ORM
- **Base de Datos**: PostgreSQL
- **Scraping**: Cheerio, Axios
- **Cron Jobs**: node-cron
- **Deployment**: Vercel, Docker

## 📁 Estructura del Proyecto

```
├── components/          # Componentes React reutilizables
├── lib/                # Lógica de negocio
│   ├── db.ts          # Configuración de Prisma
│   ├── cron.ts        # Configuración de cron jobs
│   └── scraping/      # Servicios de scraping
├── pages/             # Páginas y API routes
│   ├── api/          # Endpoints de la API
│   ├── articles/     # Página de artículos
│   ├── quotes/       # Página de citas
│   ├── books/        # Página de libros
│   └── dashboard/    # Dashboard de estadísticas
├── prisma/           # Esquema de base de datos
├── styles/           # Estilos globales
└── public/           # Archivos estáticos
```

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd scraping-dashboard
```

### 2. Instalar dependencias

```bash
npm install
```

### 3. Configurar la base de datos

#### Opción A: PostgreSQL Local

1. Instalar PostgreSQL
2. Crear una base de datos:
```sql
CREATE DATABASE scraping_db;
```

3. Configurar variables de entorno:
```bash
cp env.example .env.local
```

Editar `.env.local`:
```env
DATABASE_URL="postgresql://usuario:password@localhost:5432/scraping_db"
```

#### Opción B: Docker (Recomendado)

```bash
docker-compose up -d postgres
```

### 4. Configurar Prisma

```bash
# Generar el cliente de Prisma
npm run db:generate

# Ejecutar migraciones
npm run db:push

# (Opcional) Abrir Prisma Studio
npm run db:studio
```

### 5. Ejecutar la aplicación

```bash
# Desarrollo
npm run dev

# Producción
npm run build
npm start
```

## 🐳 Docker

### Desarrollo con Docker Compose

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f app

# Detener servicios
docker-compose down
```

### Solo la aplicación

```bash
# Construir imagen
docker build -t scraping-app .

# Ejecutar contenedor
docker run -p 3000:3000 \
  -e DATABASE_URL="postgresql://usuario:password@host:5432/db" \
  scraping-app
```

## ☁️ Deployment en Vercel

### 1. Preparar el proyecto

```bash
# Instalar Vercel CLI
npm i -g vercel

# Login en Vercel
vercel login
```

### 2. Configurar variables de entorno

En el dashboard de Vercel, agregar:
- `DATABASE_URL`: URL de tu base de datos PostgreSQL

### 3. Deploy

```bash
vercel --prod
```

### 4. Configurar base de datos en producción

Recomendamos usar:
- **Neon** (PostgreSQL serverless)
- **Supabase** (PostgreSQL con extras)
- **PlanetScale** (MySQL compatible)

## 📊 Fuentes de Datos

### Hacker News
- **URL**: https://news.ycombinator.com/
- **Datos**: Títulos, URLs, puntos, comentarios
- **Frecuencia**: Cada hora

### Quotes
- **URL**: https://quotes.toscrape.com/
- **Datos**: Citas, autores, tags
- **Frecuencia**: Cada hora

### Books
- **URL**: https://books.toscrape.com/
- **Datos**: Títulos, precios, ratings, disponibilidad
- **Frecuencia**: Cada hora

## 🔧 API Endpoints

### Artículos
- `GET /api/articles` - Listar artículos con filtros
- `GET /api/articles?search=query` - Buscar artículos
- `GET /api/articles?category=Technology` - Filtrar por categoría

### Citas
- `GET /api/quotes` - Listar citas
- `GET /api/quotes?author=Einstein` - Filtrar por autor
- `GET /api/quotes?tag=inspirational` - Filtrar por tag

### Libros
- `GET /api/books` - Listar libros
- `GET /api/books?rating=Five` - Filtrar por rating
- `GET /api/books?availability=In stock` - Filtrar por disponibilidad

### Utilidades
- `POST /api/scrape` - Ejecutar scraping manual
- `GET /api/stats` - Obtener estadísticas

## 🎨 Personalización

### Agregar nuevas fuentes

1. Crear servicio en `lib/scraping/`:
```typescript
// lib/scraping/nuevaFuente.ts
export async function scrapeNuevaFuente() {
  // Lógica de scraping
}
```

2. Agregar al index de scraping:
```typescript
// lib/scraping/index.ts
import { scrapeNuevaFuente } from './nuevaFuente'
```

3. Crear modelo en Prisma:
```prisma
// prisma/schema.prisma
model NuevaFuente {
  id   String @id @default(cuid())
  // campos específicos
}
```

4. Crear API route:
```typescript
// pages/api/nueva-fuente/index.ts
export default async function handler(req, res) {
  // Lógica de API
}
```

### Modificar cron jobs

Editar `lib/cron.ts`:
```typescript
// Ejecutar cada 30 minutos
cron.schedule('*/30 * * * *', async () => {
  // Tu lógica aquí
})
```

## 🐛 Solución de Problemas

### Error de conexión a la base de datos

1. Verificar que PostgreSQL esté ejecutándose
2. Comprobar la URL de conexión en `.env.local`
3. Ejecutar `npm run db:push` para crear las tablas

### Error de scraping

1. Verificar conectividad a internet
2. Comprobar que las URLs de las fuentes sean accesibles
3. Revisar los logs en la consola

### Error de build en Vercel

1. Verificar que todas las variables de entorno estén configuradas
2. Comprobar que la base de datos sea accesible desde Vercel
3. Revisar los logs de build en el dashboard de Vercel

## 📝 Scripts Disponibles

```bash
npm run dev          # Desarrollo
npm run build        # Construir para producción
npm run start        # Iniciar en producción
npm run lint         # Linter
npm run db:generate  # Generar cliente Prisma
npm run db:push      # Sincronizar esquema
npm run db:migrate   # Ejecutar migraciones
npm run db:studio    # Abrir Prisma Studio
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [Next.js](https://nextjs.org/) - Framework de React
- [Prisma](https://prisma.io/) - ORM para TypeScript
- [TailwindCSS](https://tailwindcss.com/) - Framework de CSS
- [Cheerio](https://cheerio.js.org/) - Parser de HTML
- [Lucide React](https://lucide.dev/) - Iconos




