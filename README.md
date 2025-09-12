# 📰 Sistema de Scraping de Noticias - Next.js

Un sistema completo de scraping de noticias peruanas desarrollado con Next.js, que permite extraer, almacenar y visualizar noticias de múltiples fuentes con imágenes y descripciones.

## 🚀 Características Principales

- **Scraping Dinámico**: Agrega y gestiona fuentes de noticias desde la interfaz web
- **Categorización Inteligente**: Sistema de categorías personalizable
- **Imágenes y Descripciones**: Extracción automática de contenido multimedia
- **Interfaz Moderna**: Dashboard responsive con TailwindCSS
- **Base de Datos**: PostgreSQL con Prisma ORM
- **Automatización**: Cron jobs para actualizaciones automáticas
- **API REST**: Endpoints para gestión de datos

## 🛠️ Tecnologías Utilizadas

### **Frontend**
- **Next.js 14.0.4**: Framework React full-stack
- **React 18**: Biblioteca de interfaz de usuario
- **TypeScript**: Lenguaje de programación con tipado estático
- **TailwindCSS 3.3.0**: Framework CSS utility-first
- **Lucide React**: Librería de iconos

### **Backend**
- **Node.js**: Runtime de JavaScript
- **Next.js API Routes**: Endpoints del servidor
- **Prisma 5.7.1**: ORM para base de datos
- **PostgreSQL**: Base de datos relacional

### **Scraping y Automatización**
- **Cheerio 1.0.0-rc.12**: Parser HTML del lado del servidor
- **Axios 1.6.2**: Cliente HTTP para requests
- **node-cron 3.0.3**: Programador de tareas (cron jobs)

### **Utilidades**
- **date-fns 2.30.0**: Manipulación de fechas
- **tsx 4.20.5**: Ejecutor de TypeScript
- **PostCSS**: Procesador CSS
- **ESLint**: Linter de código

### **Despliegue y Desarrollo**
- **Docker**: Containerización
- **Vercel**: Plataforma de despliegue
- **Git/GitHub**: Control de versiones

## 📁 Estructura del Proyecto

```
Scraping/
├── lib/
│   ├── db.ts                    # Configuración de Prisma
│   ├── cron.ts                  # Configuración de cron jobs
│   └── scraping/
│       ├── index.ts             # Scraping estático
│       ├── hackerNews.ts        # Scraper de Hacker News
│       └── dynamicScraper.ts    # Scraper dinámico
├── pages/
│   ├── api/                     # API Routes
│   │   ├── articles/            # Endpoints de artículos
│   │   ├── sources/             # Gestión de fuentes
│   │   ├── categories/          # Gestión de categorías
│   │   ├── scrape/              # Scraping manual
│   │   └── stats/               # Estadísticas
│   ├── index.tsx                # Dashboard principal
│   ├── articles/                # Página de artículos
│   ├── dashboard/               # Estadísticas detalladas
│   └── admin/                   # Panel de administración
├── prisma/
│   └── schema.prisma            # Esquema de base de datos
├── scripts/
│   └── initCategories.ts        # Inicialización de categorías
├── styles/
│   └── globals.css              # Estilos globales
└── package.json                 # Dependencias y scripts
```

## 🗄️ Base de Datos

### **Modelos Principales**

#### **Article**
```prisma
model Article {
  id          String   @id @default(cuid())
  title       String
  url         String   @unique
  content     String?
  description String?  // Descripción corta
  imageUrl    String?  // URL de imagen
  author      String?
  source      String
  category    String?
  publishedAt DateTime?
  scrapedAt   DateTime @default(now())
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
}
```

#### **ScrapingSource**
```prisma
model ScrapingSource {
  id          String   @id @default(cuid())
  name        String
  url         String
  category    String
  selector    String?  // Selectores CSS personalizados
  maxItems    Int      @default(10)
  isActive    Boolean  @default(true)
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
}
```

#### **Category**
```prisma
model Category {
  id          String   @id @default(cuid())
  name        String   @unique
  description String?
  color       String?
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
}
```

## 🚀 Instalación y Configuración

### **1. Prerrequisitos**
- Node.js 18+ 
- PostgreSQL 12+
- Git

### **2. Clonar el Repositorio**
```bash
git clone https://github.com/Omar-Condori/Scraping.git
cd Scraping
```

### **3. Instalar Dependencias**
```bash
npm install
```

### **4. Configurar Base de Datos**
```bash
# Crear base de datos PostgreSQL
createdb scraping_db

# Configurar variables de entorno
cp env.example .env.local
# Editar .env.local con tu configuración de PostgreSQL
```

### **5. Configurar Prisma**
```bash
# Generar cliente de Prisma
npm run db:generate

# Aplicar migraciones
npm run db:push

# Inicializar categorías por defecto
npm run init:categories
```

### **6. Ejecutar la Aplicación**
```bash
# Modo desarrollo
npm run dev

# La aplicación estará disponible en http://localhost:3000
```

## 📖 Manual de Usuario

### **🏠 Dashboard Principal**

**URL**: `http://localhost:3000`

**Funcionalidades**:
- **Resumen de estadísticas**: Total de artículos, fuentes activas, categorías
- **Actividad reciente**: Últimos scrapings realizados
- **Navegación rápida**: Enlaces a artículos, dashboard y administración

### **📰 Página de Artículos**

**URL**: `http://localhost:3000/articles`

**Funcionalidades**:
- **Lista de artículos**: Muestra 5 artículos por página
- **Imágenes**: Cada artículo muestra su imagen principal
- **Descripciones**: Resumen de cada noticia
- **Filtros**:
  - **Categoría**: Política, Sociedad, Internacional, etc.
  - **Fuente**: RPP, Perú 21, El Comercio, La República
- **Búsqueda**: Buscar por título o contenido
- **Paginación**: Navegar entre páginas

**Cómo usar**:
1. Selecciona una categoría del filtro
2. Elige una fuente específica
3. Usa la barra de búsqueda para encontrar artículos
4. Haz clic en "Ver artículo" para leer la noticia completa

### **📊 Dashboard de Estadísticas**

**URL**: `http://localhost:3000/dashboard`

**Funcionalidades**:
- **Estadísticas generales**: Total de artículos, fuentes, categorías
- **Distribución por categoría**: Gráfico de barras
- **Distribución por fuente**: Gráfico circular
- **Actividad reciente**: Log de scrapings
- **Artículos por día**: Gráfico de líneas

### **⚙️ Panel de Administración**

**URL**: `http://localhost:3000/admin`

**Funcionalidades**:

#### **Gestión de Fuentes**
- **Agregar fuente**: 
  - Nombre de la fuente
  - URL del sitio web
  - Categoría asignada
  - Número máximo de artículos
- **Editar fuente**: Modificar configuración existente
- **Activar/Desactivar**: Toggle para controlar el scraping

#### **Gestión de Categorías**
- **Agregar categoría**:
  - Nombre de la categoría
  - Descripción
  - Color personalizado
- **Editar categoría**: Modificar información existente

**Cómo usar**:
1. Ve a la sección "Gestión de Fuentes"
2. Completa el formulario con la URL del sitio
3. Selecciona la categoría apropiada
4. Guarda la fuente
5. La fuente aparecerá en los filtros de artículos

## 🔧 Comandos Útiles

### **Desarrollo**
```bash
# Ejecutar en modo desarrollo
npm run dev

# Construir para producción
npm run build

# Ejecutar en producción
npm start
```

### **Base de Datos**
```bash
# Generar cliente de Prisma
npm run db:generate

# Aplicar cambios al esquema
npm run db:push

# Crear migración
npm run db:migrate

# Abrir Prisma Studio
npm run db:studio
```

### **Scraping**
```bash
# Scraping manual (estático)
curl -X POST http://localhost:3000/api/scrape

# Scraping dinámico
curl -X POST http://localhost:3000/api/scrape-dynamic

# Inicializar categorías
npm run init:categories
```

### **Base de Datos PostgreSQL**
```bash
# Conectar a la base de datos
psql -d scraping_db

# Ver todas las tablas
\dt

# Contar artículos por categoría
SELECT category, COUNT(*) FROM articles GROUP BY category;

# Ver artículos con imágenes
SELECT title, "imageUrl" FROM articles WHERE "imageUrl" IS NOT NULL LIMIT 5;

# Ver artículos con descripciones
SELECT title, description FROM articles WHERE description IS NOT NULL LIMIT 5;
```

## 🔄 Automatización

### **Cron Jobs Configurados**

El sistema incluye tareas automáticas:

- **Cada hora**: Scraping dinámico de todas las fuentes activas
- **Cada 6 horas**: Scraping completo del sistema
- **Logs automáticos**: Registro de todas las operaciones

### **Configuración de Cron**
```typescript
// lib/cron.ts
cron.schedule('0 * * * *', () => {
  console.log('Ejecutando scraping cada hora...')
  scrapeAllDynamicSources()
})

cron.schedule('0 */6 * * *', () => {
  console.log('Ejecutando scraping cada 6 horas...')
  scrapeAll()
})
```

## 🌐 API Endpoints

### **Artículos**
- `GET /api/articles` - Listar artículos con filtros
- `GET /api/articles/[id]` - Obtener artículo específico

### **Fuentes**
- `GET /api/sources` - Listar fuentes de scraping
- `POST /api/sources` - Crear nueva fuente
- `PUT /api/sources/[id]` - Actualizar fuente
- `DELETE /api/sources/[id]` - Eliminar fuente

### **Categorías**
- `GET /api/categories` - Listar categorías
- `POST /api/categories` - Crear nueva categoría
- `PUT /api/categories/[id]` - Actualizar categoría

### **Scraping**
- `POST /api/scrape` - Ejecutar scraping estático
- `POST /api/scrape-dynamic` - Ejecutar scraping dinámico

### **Estadísticas**
- `GET /api/stats` - Obtener estadísticas generales

## 🐳 Docker

### **Ejecutar con Docker Compose**
```bash
# Construir y ejecutar
docker-compose up --build

# Ejecutar en segundo plano
docker-compose up -d

# Ver logs
docker-compose logs -f
```

### **Archivos Docker**
- `Dockerfile`: Configuración del contenedor de la aplicación
- `docker-compose.yml`: Orquestación de servicios (app + PostgreSQL)

## 🚀 Despliegue en Vercel

### **Configuración**
1. Conecta tu repositorio GitHub a Vercel
2. Configura las variables de entorno:
   - `DATABASE_URL`: URL de tu base de datos PostgreSQL
3. Despliega automáticamente

### **Variables de Entorno Requeridas**
```env
DATABASE_URL="postgresql://usuario:password@host:puerto/database"
```

## 📊 Estadísticas del Proyecto

### **Fuentes de Noticias Activas**
- **RPP**: Noticias generales y políticas
- **Perú 21**: Noticias internacionales
- **El Comercio**: Noticias políticas y económicas
- **La República**: Noticias de sociedad y actualidad

### **Categorías Disponibles**
1. **Política**: Nacional, Gobierno, Congreso, Elecciones
2. **Economía / Negocios**: Finanzas, Mercados, Empleo
3. **Sociedad / Actualidad**: Ciudad, Comunidad, Sucesos
4. **Internacional**: Mundo, Latinoamérica
5. **Tecnología / Ciencia**: Innovación, Medio ambiente, Salud
6. **Cultura / Arte**: Música, Cine, Literatura
7. **Opinión**: Editoriales, Columnistas
8. **Estilo de vida / Tendencias**: Gastronomía, Viajes, Moda
9. **Clima y Medio ambiente**: Clima, Sostenibilidad

### **Rendimiento Actual**
- **Total de artículos**: ~115
- **Artículos con imágenes**: ~52 (45%)
- **Artículos con descripciones**: ~94 (82%)
- **Fuentes activas**: 4
- **Categorías**: 9

## 🔧 Solución de Problemas

### **Error de Conexión a Base de Datos**
```bash
# Verificar que PostgreSQL esté ejecutándose
brew services start postgresql

# Verificar conexión
psql -d scraping_db -c "SELECT 1;"
```

### **Error de Prisma**
```bash
# Regenerar cliente
npm run db:generate

# Aplicar esquema
npm run db:push
```

### **Error de Scraping**
```bash
# Verificar logs
curl -X POST http://localhost:3000/api/scrape-dynamic

# Verificar fuentes activas
psql -d scraping_db -c "SELECT name, url, \"isActive\" FROM scraping_sources;"
```

## 📝 Notas de Desarrollo

### **Selectores CSS Personalizados**
El sistema permite configurar selectores CSS específicos para cada fuente:

```json
{
  "title": ".story-item__title, h2, h3",
  "link": "a[href*='/noticias/']",
  "content": ".story-item__summary, p",
  "image": ".story-item__image img, img[src*='rpp']"
}
```

### **Límites de Scraping**
- **Máximo por fuente**: 10 artículos (configurable)
- **Timeout de requests**: 30 segundos
- **Intervalo entre scrapings**: 1 hora

### **Optimizaciones Implementadas**
- **Upsert en base de datos**: Evita duplicados
- **Lazy loading de imágenes**: Mejora rendimiento
- **Paginación**: Limita carga de datos
- **Caché de respuestas**: Reduce requests innecesarios

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**Omar Condori**
- GitHub: [@Omar-Condori](https://github.com/Omar-Condori)
- Proyecto: [Scraping](https://github.com/Omar-Condori/Scraping)

---

## 🎯 Próximas Mejoras

- [ ] **Sistema de notificaciones** para nuevos artículos
- [ ] **Exportación de datos** en CSV/JSON
- [ ] **Análisis de sentimientos** de las noticias
- [ ] **API de búsqueda avanzada** con filtros complejos
- [ ] **Sistema de usuarios** y autenticación
- [ ] **Dashboard de métricas** en tiempo real
- [ ] **Integración con redes sociales** para compartir
- [ ] **Sistema de alertas** por palabras clave

---

**¡Disfruta explorando las noticias peruanas con tu sistema de scraping personalizado!** 🎉📰