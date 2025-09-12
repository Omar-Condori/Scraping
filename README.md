# 🚀 Next.js Scraping Dashboard

Un dashboard completo de scraping construido con Next.js, que recolecta datos de múltiples fuentes y los presenta en una interfaz moderna y responsiva.

## ✨ Características

- **Scraping Automático**: Recolección de datos de fuentes de noticias peruanas
- **Base de Datos**: PostgreSQL con Prisma ORM
- **Cron Jobs**: Actualización automática cada hora
- **Frontend Moderno**: Interfaz con TailwindCSS y componentes reutilizables
- **API RESTful**: Endpoints para consultar datos
- **Dashboard**: Estadísticas y análisis de datos
- **Administración**: Panel para gestionar fuentes y categorías
- **Deployment**: Preparado para Vercel y Docker

## 🛠️ Tecnologías

- **Frontend**: Next.js 14, React, TypeScript, TailwindCSS
- **Backend**: Next.js API Routes, Prisma ORM
- **Base de Datos**: PostgreSQL
- **Scraping**: Cheerio, Axios
- **Scheduling**: node-cron
- **Deployment**: Vercel, Docker

## 📊 Fuentes Actuales

- **RPP**: Noticias de política
- **Perú 21**: Noticias internacionales
- **El Comercio**: Noticias de política
- **La República**: Noticias de sociedad/actualidad

## 🎯 Categorías

- Política
- Internacional
- Sociedad / Actualidad
- Tecnología / Ciencia
- Cultura / Arte
- Economía / Negocios
- Opinión
- Estilo de vida / Tendencias
- Clima y Medio ambiente

## 🚀 Instalación

### Prerrequisitos

- Node.js 18+
- PostgreSQL
- npm o yarn

### Pasos

1. **Clonar el repositorio**
```bash
git clone https://github.com/Omar-Condori/Scraping.git
cd Scraping
```

2. **Instalar dependencias**
```bash
npm install
```

3. **Configurar variables de entorno**
```bash
cp env.example .env.local
```

Editar `.env.local`:
```env
DATABASE_URL="postgresql://usuario:password@localhost:5432/scraping_db"
```

4. **Configurar la base de datos**
```bash
# Generar cliente Prisma
npm run db:generate

# Aplicar migraciones
npm run db:push

# Inicializar categorías
npm run init:categories
```

5. **Ejecutar la aplicación**
```bash
npm run dev
```

La aplicación estará disponible en `http://localhost:3000`

## 📱 Uso

### Páginas Principales

- **Dashboard**: `http://localhost:3000` - Vista general y estadísticas
- **Artículos**: `http://localhost:3000/articles` - Lista de noticias con filtros
- **Administración**: `http://localhost:3000/admin` - Gestionar fuentes y categorías

### API Endpoints

- `GET /api/articles` - Obtener artículos
- `GET /api/categories` - Obtener categorías
- `GET /api/sources` - Obtener fuentes
- `GET /api/stats` - Obtener estadísticas
- `POST /api/scrape` - Ejecutar scraping manual
- `POST /api/scrape-dynamic` - Ejecutar scraping dinámico

## 🔧 Comandos Útiles

```bash
# Desarrollo
npm run dev

# Construcción
npm run build
npm run start

# Base de datos
npm run db:generate
npm run db:push
npm run db:migrate
npm run db:studio

# Inicializar categorías
npm run init:categories
```

## 🐳 Docker

```bash
# Construir imagen
docker build -t scraping-app .

# Ejecutar con Docker Compose
docker-compose up -d
```

## 📈 Características Avanzadas

- **Scraping Dinámico**: Agregar nuevas fuentes desde la interfaz
- **Filtros Inteligentes**: Buscar y filtrar por categoría, fuente, fecha
- **Imágenes y Descripciones**: Extracción automática de metadatos
- **Paginación**: Navegación eficiente de grandes volúmenes de datos
- **Responsive Design**: Optimizado para móviles y desktop

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autor

**Omar Condori**
- GitHub: [@Omar-Condori](https://github.com/Omar-Condori)

---

## 📚 Proyectos Anteriores

Este repositorio también contiene scripts de Python para scraping de El Comercio:

- `scraping_elcomercio.py` - Scraping básico
- `scraping_elcomercio_contenido.py` - Con contenido completo
- `scraping_elcomercio_excel.py` - Exportación a Excel
- Y otros scripts especializados

Los archivos de Python están incluidos para referencia histórica del desarrollo del proyecto.