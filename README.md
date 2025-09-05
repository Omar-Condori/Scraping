# 📰 Proyecto de Web Scraping - El Comercio

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de **web scraping** para extraer noticias del periódico **El Comercio** (elcomercio.pe), incluyendo la creación de una base de datos SQLite y generación de reportes en Excel.

## 🎯 Objetivos

- Extraer noticias de la página web de El Comercio
- Clasificar automáticamente las noticias por categorías
- Almacenar los datos en una base de datos SQLite
- Generar reportes en formato Excel
- Crear consultas SQL para análisis de datos

## 🛠️ Tecnologías Utilizadas

- **Python 3.9**
- **Selenium WebDriver** - Para scraping de contenido dinámico
- **BeautifulSoup4** - Para análisis de HTML
- **SQLite3** - Base de datos relacional
- **Pandas** - Manipulación de datos
- **OpenPyXL** - Generación de archivos Excel
- **Git** - Control de versiones

## 📁 Estructura del Proyecto

```
Scraping/
├── README.md                                    # Documentación del proyecto
├── scraping_elcomercio_simple.py               # Script básico de scraping
├── scraping_elcomercio.py                      # Script con filtros de fecha
├── scraping_elcomercio_mejorado.py             # Script mejorado con múltiples selectores
├── scraping_elcomercio_con_fechas.py           # Script con extracción de fechas
├── scraping_elcomercio_titulo_categoria.py     # Script con clasificación por categorías
├── scraping_elcomercio_excel.py                # Script que genera reportes en Excel
├── scraping_elcomercio_semana.py               # Script para noticias de la semana
├── scraping_elcomercio_contenido.py            # Script con extracción de contenido completo
├── scraping_elcomercio_contenido_mejorado.py   # Script mejorado para contenido
├── crear_base_datos_noticias.py                # Script para crear la base de datos
├── consultas_sql_noticias.sql                  # Archivo con consultas SQL
├── ejecutar_consultas_sql.py                   # Script para ejecutar consultas
├── noticias_elcomercio.db                      # Base de datos SQLite
├── reporte_profesor.txt                        # Reporte completo para el profesor
└── base_datos_noticias_profesor.xlsx           # Datos exportados a Excel
```

## 🚀 Instalación y Configuración

### Prerrequisitos

```bash
# Instalar dependencias de Python
pip3 install selenium beautifulsoup4 pandas openpyxl

# Instalar ChromeDriver (macOS con Homebrew)
brew install chromedriver
```

### Configuración

1. **Clonar el repositorio:**
```bash
git clone git@github.com:Omar-Condori/Scraping.git
cd Scraping
```

2. **Verificar que ChromeDriver esté instalado:**
```bash
which chromedriver
# Debe mostrar: /opt/homebrew/bin/chromedriver
```

## 📊 Uso del Proyecto

### 1. Scraping Básico
```bash
python3 scraping_elcomercio_simple.py
```

### 2. Scraping con Categorías
```bash
python3 scraping_elcomercio_titulo_categoria.py
```

### 3. Scraping con Contenido Completo
```bash
python3 scraping_elcomercio_contenido_mejorado.py
```

### 4. Crear Base de Datos
```bash
python3 crear_base_datos_noticias.py
```

### 5. Ejecutar Consultas SQL
```bash
python3 ejecutar_consultas_sql.py
```

## 🗄️ Base de Datos

### Estructura de Tablas

#### Tabla: `noticias`
- `id` - Identificador único
- `titulo` - Título de la noticia
- `categoria` - Categoría clasificada
- `resumen` - Resumen de la noticia
- `contenido_completo` - Contenido completo del artículo
- `fecha_publicacion` - Fecha de publicación
- `fecha_extraccion` - Fecha de extracción
- `enlace` - URL de la noticia
- `longitud_contenido` - Número de caracteres
- `selector_utilizado` - Selector CSS utilizado
- `fuente` - Fuente de la noticia

#### Tabla: `categorias`
- `id` - Identificador único
- `nombre` - Nombre de la categoría
- `descripcion` - Descripción de la categoría
- `total_noticias` - Contador de noticias

#### Tabla: `estadisticas`
- `id` - Identificador único
- `fecha_extraccion` - Fecha de extracción
- `total_noticias` - Total de noticias
- `categorias_unicas` - Número de categorías únicas
- `promedio_longitud` - Promedio de longitud de contenido
- `noticia_mas_larga` - Título de la noticia más larga
- `noticia_mas_corta` - Título de la noticia más corta

### Consultas SQL Principales

```sql
-- Ver todas las noticias
SELECT * FROM noticias ORDER BY fecha_publicacion DESC;

-- Contar noticias por categoría
SELECT categoria, COUNT(*) FROM noticias GROUP BY categoria;

-- Noticias más largas
SELECT titulo, longitud_contenido FROM noticias 
ORDER BY longitud_contenido DESC LIMIT 5;

-- Estadísticas generales
SELECT COUNT(*) as total, AVG(longitud_contenido) as promedio 
FROM noticias;
```

## 📈 Resultados del Proyecto

### Estadísticas Generales
- **Total de noticias extraídas:** 8
- **Categorías identificadas:** 5
- **Promedio de longitud:** 3,584 caracteres
- **Noticia más larga:** 8,003 caracteres
- **Noticia más corta:** 1,145 caracteres

### Distribución por Categorías
- **Política:** 2 noticias (promedio: 4,574 caracteres)
- **Mundo:** 2 noticias (promedio: 4,834 caracteres)
- **Entretenimiento:** 2 noticias (promedio: 2,045 caracteres)
- **Economía:** 1 noticia (2,564 caracteres)
- **Deportes:** 1 noticia (3,200 caracteres)

## 🔍 Metodología

### 1. Web Scraping
- **Selenium WebDriver** para contenido dinámico
- **BeautifulSoup** para análisis de HTML
- **Múltiples selectores CSS** para robustez
- **Filtrado de contenido** no relevante

### 2. Clasificación Automática
- **Clasificación por URL** (rutas de la página)
- **Clasificación por contenido** (palabras clave en títulos)
- **Priorización** de categorías más específicas

### 3. Almacenamiento de Datos
- **Base de datos SQLite** para consultas eficientes
- **Normalización** de datos
- **Relaciones** entre tablas
- **Índices** para optimización

### 4. Generación de Reportes
- **Archivos Excel** con formato profesional
- **Múltiples hojas** de trabajo
- **Estadísticas** y análisis
- **Documentación** completa

## 📊 Archivos de Salida

### Reportes Generados
- `reporte_elcomercio_YYYYMMDD_HHMMSS.xlsx` - Reporte básico
- `reporte_semanal_elcomercio_YYYYMMDD_HHMMSS.xlsx` - Reporte semanal
- `reporte_contenido_completo_elcomercio_YYYYMMDD_HHMMSS.xlsx` - Reporte con contenido
- `base_datos_noticias_profesor.xlsx` - Datos de la base de datos
- `reporte_profesor.txt` - Reporte completo en texto

### Base de Datos
- `noticias_elcomercio.db` - Base de datos SQLite principal

## 🎓 Propósito Académico

Este proyecto fue desarrollado como **tarea académica** para demostrar:
- Conocimientos en **web scraping**
- Manejo de **bases de datos relacionales**
- **Análisis de datos** con SQL
- **Generación de reportes** automatizados
- **Documentación** de proyectos técnicos

## 📝 Notas Técnicas

### Limitaciones
- El scraping depende de la estructura HTML del sitio web
- Los selectores CSS pueden cambiar con actualizaciones del sitio
- El contenido dinámico requiere JavaScript habilitado

### Mejoras Futuras
- Implementar **scraping distribuido**
- Agregar **análisis de sentimientos**
- Crear **API REST** para consultas
- Implementar **monitoreo** de cambios en el sitio

## 👨‍💻 Autor

**Omar Condori**
- Estudiante de Ingeniería de Sistemas
- Proyecto académico de Web Scraping

## 📄 Licencia

Este proyecto es de uso académico y educativo.

---

**Fecha de creación:** 2025-09-05  
**Última actualización:** 2025-09-05  
**Versión:** 1.0.0
