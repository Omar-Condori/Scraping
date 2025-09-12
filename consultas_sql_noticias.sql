-- =====================================================
-- BASE DE DATOS: NOTICIAS EL COMERCIO
-- Archivo: consultas_sql_noticias.sql
-- Fecha: 2025-09-05
-- Autor: Omar (Estudiante)
-- =====================================================

-- =====================================================
-- 1. CREACIÓN DE LA BASE DE DATOS Y TABLAS
-- =====================================================

-- Crear base de datos (SQLite se crea automáticamente)
-- Archivo: noticias_elcomercio.db

-- Tabla: noticias
CREATE TABLE IF NOT EXISTS noticias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    titulo TEXT NOT NULL,
    categoria TEXT NOT NULL,
    resumen TEXT,
    contenido_completo TEXT,
    fecha_publicacion TEXT,
    fecha_extraccion TEXT NOT NULL,
    enlace TEXT NOT NULL,
    longitud_contenido INTEGER,
    selector_utilizado TEXT,
    fuente TEXT DEFAULT 'El Comercio'
);

-- Tabla: categorias
CREATE TABLE IF NOT EXISTS categorias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT UNIQUE NOT NULL,
    descripcion TEXT,
    total_noticias INTEGER DEFAULT 0
);

-- Tabla: estadisticas
CREATE TABLE IF NOT EXISTS estadisticas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_extraccion TEXT NOT NULL,
    total_noticias INTEGER,
    categorias_unicas INTEGER,
    promedio_longitud REAL,
    noticia_mas_larga TEXT,
    noticia_mas_corta TEXT
);

-- =====================================================
-- 2. INSERCIÓN DE DATOS DE CATEGORÍAS
-- =====================================================

INSERT OR IGNORE INTO categorias (nombre, descripcion) VALUES 
('Política', 'Noticias relacionadas con política nacional e internacional'),
('Economía', 'Noticias económicas, financieras y de mercado'),
('Mundo', 'Noticias internacionales y globales'),
('Deportes', 'Noticias deportivas y eventos deportivos'),
('Espectáculos', 'Noticias de entretenimiento, cine, música y cultura'),
('Entretenimiento', 'Contenido de entretenimiento, anime, series'),
('Tecnología', 'Noticias de tecnología e innovación'),
('Sociedad', 'Noticias sociales y comunitarias'),
('Salud', 'Noticias de salud y medicina'),
('Educación', 'Noticias educativas y académicas');

-- =====================================================
-- 3. INSERCIÓN DE NOTICIAS DE EJEMPLO
-- =====================================================

INSERT INTO noticias (titulo, categoria, resumen, contenido_completo, fecha_publicacion, fecha_extraccion, enlace, longitud_contenido, selector_utilizado) VALUES 
('"Lord of Mysteries" Capítulo 12: Hora confirmada de estreno', 'Entretenimiento', 'Información sobre el estreno del capítulo 12 de Lord of Mysteries en Crunchyroll', 'El popular anime "Lord of Mysteries" continúa su segunda temporada con el estreno del capítulo 12. Los fanáticos pueden disfrutar de este episodio en la plataforma Crunchyroll. La serie ha mantenido una gran audiencia desde su debut y este nuevo capítulo promete continuar con la emocionante trama que ha cautivado a los espectadores.', '2025-09-05', '2025-09-05 10:26:00', 'https://elcomercio.pe/saltar-intro/noticias/lord-of-mysteries-capitulo-12-hora-confirmada-de-estreno-crunchyroll-noticia/', 2605, 'a[href*="/noticias/"]'),

('"Kaiju No.8" Temporada 2, Capítulo 8: Hora confirmada de estreno', 'Entretenimiento', 'Nuevo episodio de Kaiju No.8 disponible en plataformas de streaming', 'La segunda temporada de "Kaiju No.8" presenta su octavo capítulo con horarios confirmados para su estreno. Esta serie de anime ha ganado popularidad por su animación de alta calidad y su historia envolvente sobre la lucha contra criaturas gigantes.', '2025-09-05', '2025-09-05 10:26:00', 'https://elcomercio.pe/saltar-intro/noticias/kaiju-no8-temporada-2-capitulo-8-hora-confirmada-de-estreno-crunchyroll-noticia/', 1484, 'a[href*="/noticias/"]'),

('Betssy Chávez sale en libertad tras sentencia del Tribunal Constitucional', 'Política', 'La exministra de Pedro Castillo recupera su libertad tras decisión judicial', 'Betssy Chávez, exministra del gobierno de Pedro Castillo, ha sido liberada tras una sentencia del Tribunal Constitucional que anuló su prisión preventiva. La decisión judicial marca un hito importante en el caso y ha generado diversas reacciones en el ámbito político nacional.', '2025-09-05', '2025-09-05 10:26:00', 'https://elcomercio.pe/politica/actualidad/betssy-chavez-sale-en-libertad-tras-sentencia-del-tribunal-constitucional-que-anulo-su-prision-preventiva-ultimas-noticia/', 1145, 'a[href*="/politica/"]'),

('Alejandro Toledo lavó millones de dólares producto de coimas: Caso Ecoteva', 'Política', 'Nuevas revelaciones sobre el lavado de dinero del expresidente Toledo', 'El Poder Judicial ha revelado detalles sobre cómo Alejandro Toledo lavó millones de dólares producto de coimas en el caso Ecoteva. Los argumentos presentados por la fiscalía muestran una red compleja de transferencias y operaciones financieras que involucran a múltiples personas y empresas. Esta información refuerza la condena contra el expresidente y proporciona evidencia adicional sobre la magnitud de los actos de corrupción.', '2025-09-04', '2025-09-05 10:26:00', 'https://elcomercio.pe/politica/alejandro-toledo-lavo-dinero-ilicito-estos-fueron-los-argumentos-para-condenarlo-por-el-caso-ecoteva-tlcnota-noticia/', 8003, 'a[href*="/politica/"]'),

('Gobierno aprueba reglamento de modernización del sistema de pensiones', 'Economía', 'Nuevas regulaciones para el sistema de pensiones en Perú', 'El gobierno peruano ha aprobado un nuevo reglamento que moderniza el sistema de pensiones, afectando tanto a la ONP como a las AFP. Esta medida busca mejorar la eficiencia del sistema y garantizar mejores beneficios para los pensionistas. El reglamento incluye cambios en los requisitos de afiliación y en los cálculos de pensiones.', '2025-09-04', '2025-09-05 10:26:00', 'https://elcomercio.pe/economia/peru/gobierno-aprueba-reglamento-de-modernizacion-del-sistema-de-pensiones-l-onp-l-afp-l-ultimas-noticia/', 2564, 'a[href*="/economia/"]'),

('Argentina: cinco claves para entender la elección legislativa de Buenos Aires', 'Mundo', 'Análisis de las elecciones legislativas en la provincia de Buenos Aires', 'Las elecciones legislativas en la provincia de Buenos Aires representan un momento crucial para la política argentina. Los bonaerenses elegirán a la mitad de los integrantes del Legislativo provincial: 46 diputados y 23 senadores. Esta elección es fundamental para entender el panorama político nacional y las tendencias electorales en la región más poblada del país.', '2025-09-03', '2025-09-05 10:26:00', 'https://elcomercio.pe/mundo/latinoamerica/argentina-cinco-claves-para-entender-la-importancia-de-la-eleccion-legislativa-de-la-provincia-de-buenos-aires-ultimas-noticia/', 5468, 'a[href*="/mundo/"]'),

('Zelensky espera que países europeos desplieguen miles de soldados en Ucrania', 'Mundo', 'El presidente ucraniano solicita mayor apoyo militar europeo', 'El presidente de Ucrania, Volodymyr Zelensky, ha expresado su esperanza de que los países europeos desplieguen miles de soldados en Ucrania tras el alto el fuego. Esta solicitud forma parte de la estrategia ucraniana para fortalecer su defensa ante la agresión rusa y garantizar la estabilidad en la región.', '2025-09-02', '2025-09-05 10:26:00', 'https://elcomercio.pe/mundo/europa/guerra-rusia-ucrania-volodymyr-zelensky-espera-que-paises-europeos-desplieguen-miles-soldados-en-ucrania-tras-alto-el-fuego-vladimir-putin-coalicion-de-voluntarios-ultimas-noticia/', 4200, 'a[href*="/mundo/"]'),

('Diego Rebagliati: "Si Perú quiere volver al Mundial, el próximo técnico debe tener estas cualidades"', 'Deportes', 'Análisis del periodista deportivo sobre el futuro técnico de la selección peruana', 'El reconocido periodista deportivo Diego Rebagliati ha compartido su análisis sobre las cualidades que debe tener el próximo técnico de la selección peruana de fútbol. Según Rebagliati, para que Perú pueda volver al Mundial, es fundamental que el nuevo entrenador tenga experiencia internacional, conocimiento del fútbol sudamericano y capacidad para trabajar con jugadores jóvenes.', '2025-09-01', '2025-09-05 10:26:00', 'https://elcomercio.pe/respuestas/deportes/lo-tiene-claro-diego-rebagliati-aseguro-que-si-peru-quiere-volver-al-mundial-el-proximo-tecnico-debe-ser-asi-el-perfil-tdpe-noticia/', 3200, 'a[href*="/deportes/"]');

-- =====================================================
-- 4. CONSULTAS DE ANÁLISIS Y ESTADÍSTICAS
-- =====================================================

-- Consulta 1: Ver todas las noticias
SELECT 
    id,
    titulo,
    categoria,
    fecha_publicacion,
    longitud_contenido
FROM noticias
ORDER BY fecha_publicacion DESC;

-- Consulta 2: Contar noticias por categoría
SELECT 
    categoria,
    COUNT(*) as total_noticias,
    AVG(longitud_contenido) as promedio_longitud
FROM noticias
GROUP BY categoria
ORDER BY total_noticias DESC;

-- Consulta 3: Noticias más largas
SELECT 
    titulo,
    categoria,
    longitud_contenido,
    fecha_publicacion
FROM noticias
ORDER BY longitud_contenido DESC
LIMIT 5;

-- Consulta 4: Noticias más cortas
SELECT 
    titulo,
    categoria,
    longitud_contenido,
    fecha_publicacion
FROM noticias
ORDER BY longitud_contenido ASC
LIMIT 5;

-- Consulta 5: Estadísticas generales
SELECT 
    COUNT(*) as total_noticias,
    COUNT(DISTINCT categoria) as categorias_unicas,
    AVG(longitud_contenido) as promedio_longitud,
    MIN(longitud_contenido) as longitud_minima,
    MAX(longitud_contenido) as longitud_maxima
FROM noticias;

-- Consulta 6: Noticias por fecha
SELECT 
    fecha_publicacion,
    COUNT(*) as noticias_del_dia
FROM noticias
GROUP BY fecha_publicacion
ORDER BY fecha_publicacion DESC;

-- Consulta 7: Buscar noticias por palabra clave
SELECT 
    titulo,
    categoria,
    resumen
FROM noticias
WHERE titulo LIKE '%Toledo%' 
   OR contenido_completo LIKE '%Toledo%';

-- Consulta 8: Noticias de entretenimiento
SELECT 
    titulo,
    resumen,
    longitud_contenido
FROM noticias
WHERE categoria = 'Entretenimiento';

-- Consulta 9: Noticias políticas
SELECT 
    titulo,
    fecha_publicacion,
    longitud_contenido
FROM noticias
WHERE categoria = 'Política'
ORDER BY fecha_publicacion DESC;

-- Consulta 10: Análisis de selectores utilizados
SELECT 
    selector_utilizado,
    COUNT(*) as veces_utilizado
FROM noticias
GROUP BY selector_utilizado
ORDER BY veces_utilizado DESC;

-- =====================================================
-- 5. CONSULTAS AVANZADAS
-- =====================================================

-- Consulta 11: Noticias con contenido más rico (más de 5000 caracteres)
SELECT 
    titulo,
    categoria,
    longitud_contenido,
    SUBSTR(contenido_completo, 1, 100) || '...' as preview_contenido
FROM noticias
WHERE longitud_contenido > 5000
ORDER BY longitud_contenido DESC;

-- Consulta 12: Distribución temporal de noticias
SELECT 
    CASE 
        WHEN fecha_publicacion = '2025-09-05' THEN 'Hoy'
        WHEN fecha_publicacion = '2025-09-04' THEN 'Ayer'
        WHEN fecha_publicacion = '2025-09-03' THEN 'Hace 2 días'
        ELSE 'Más de 2 días'
    END as periodo,
    COUNT(*) as cantidad_noticias
FROM noticias
GROUP BY 
    CASE 
        WHEN fecha_publicacion = '2025-09-05' THEN 'Hoy'
        WHEN fecha_publicacion = '2025-09-04' THEN 'Ayer'
        WHEN fecha_publicacion = '2025-09-03' THEN 'Hace 2 días'
        ELSE 'Más de 2 días'
    END;

-- Consulta 13: Top 3 categorías por longitud promedio
SELECT 
    categoria,
    COUNT(*) as total_noticias,
    AVG(longitud_contenido) as promedio_longitud,
    SUM(longitud_contenido) as total_caracteres
FROM noticias
GROUP BY categoria
ORDER BY promedio_longitud DESC
LIMIT 3;

-- Consulta 14: Noticias con enlaces específicos
SELECT 
    titulo,
    categoria,
    CASE 
        WHEN enlace LIKE '%politica%' THEN 'Sección Política'
        WHEN enlace LIKE '%economia%' THEN 'Sección Economía'
        WHEN enlace LIKE '%mundo%' THEN 'Sección Mundo'
        WHEN enlace LIKE '%deportes%' THEN 'Sección Deportes'
        WHEN enlace LIKE '%saltar-intro%' THEN 'Sección Entretenimiento'
        ELSE 'Otra sección'
    END as seccion_web
FROM noticias
ORDER BY seccion_web, categoria;

-- Consulta 15: Resumen ejecutivo
SELECT 
    'RESUMEN EJECUTIVO' as tipo,
    COUNT(*) as valor
FROM noticias
UNION ALL
SELECT 
    'Categorías únicas',
    COUNT(DISTINCT categoria)
FROM noticias
UNION ALL
SELECT 
    'Promedio de caracteres',
    ROUND(AVG(longitud_contenido))
FROM noticias
UNION ALL
SELECT 
    'Noticia más larga (caracteres)',
    MAX(longitud_contenido)
FROM noticias
UNION ALL
SELECT 
    'Noticia más corta (caracteres)',
    MIN(longitud_contenido)
FROM noticias;

-- =====================================================
-- 6. CONSULTAS DE MANTENIMIENTO
-- =====================================================

-- Actualizar contadores de categorías
UPDATE categorias 
SET total_noticias = (
    SELECT COUNT(*) 
    FROM noticias 
    WHERE noticias.categoria = categorias.nombre
);

-- Insertar estadísticas
INSERT INTO estadisticas (
    fecha_extraccion, 
    total_noticias, 
    categorias_unicas, 
    promedio_longitud, 
    noticia_mas_larga, 
    noticia_mas_corta
)
SELECT 
    datetime('now') as fecha_extraccion,
    COUNT(*) as total_noticias,
    COUNT(DISTINCT categoria) as categorias_unicas,
    ROUND(AVG(longitud_contenido), 2) as promedio_longitud,
    (SELECT titulo FROM noticias ORDER BY longitud_contenido DESC LIMIT 1) as noticia_mas_larga,
    (SELECT titulo FROM noticias ORDER BY longitud_contenido ASC LIMIT 1) as noticia_mas_corta
FROM noticias;

-- =====================================================
-- 7. CONSULTAS DE VERIFICACIÓN
-- =====================================================

-- Verificar integridad de datos
SELECT 'Verificación de datos' as tipo, COUNT(*) as cantidad FROM noticias;
SELECT 'Categorías definidas' as tipo, COUNT(*) as cantidad FROM categorias;
SELECT 'Estadísticas generadas' as tipo, COUNT(*) as cantidad FROM estadisticas;

-- Verificar que todas las noticias tienen categoría válida
SELECT 
    n.titulo,
    n.categoria,
    CASE 
        WHEN c.nombre IS NULL THEN 'Categoría no encontrada'
        ELSE 'Categoría válida'
    END as estado_categoria
FROM noticias n
LEFT JOIN categorias c ON n.categoria = c.nombre;

-- =====================================================
-- FIN DEL ARCHIVO SQL
-- =====================================================
-- 
-- INSTRUCCIONES DE USO:
-- 1. Abrir SQLite: sqlite3 noticias_elcomercio.db
-- 2. Ejecutar: .read consultas_sql_noticias.sql
-- 3. O ejecutar consultas individuales copiando y pegando
-- 
-- ARCHIVOS RELACIONADOS:
-- - noticias_elcomercio.db (base de datos)
-- - reporte_profesor.txt (documentación)
-- - base_datos_noticias_profesor.xlsx (datos en Excel)
-- 
-- =====================================================
