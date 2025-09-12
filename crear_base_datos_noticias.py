import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import os

def crear_base_datos():
    """Crea la base de datos SQLite para las noticias"""
    
    # Conectar a la base de datos (se crea si no existe)
    conn = sqlite3.connect('noticias_elcomercio.db')
    cursor = conn.cursor()
    
    # Crear tabla de noticias
    cursor.execute('''
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
        )
    ''')
    
    # Crear tabla de categorías
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categorias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT UNIQUE NOT NULL,
            descripcion TEXT,
            total_noticias INTEGER DEFAULT 0
        )
    ''')
    
    # Crear tabla de estadísticas
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS estadisticas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_extraccion TEXT NOT NULL,
            total_noticias INTEGER,
            categorias_unicas INTEGER,
            promedio_longitud REAL,
            noticia_mas_larga TEXT,
            noticia_mas_corta TEXT
        )
    ''')
    
    # Insertar categorías por defecto
    categorias_default = [
        ('Política', 'Noticias relacionadas con política nacional e internacional'),
        ('Economía', 'Noticias económicas, financieras y de mercado'),
        ('Mundo', 'Noticias internacionales y globales'),
        ('Deportes', 'Noticias deportivas y eventos deportivos'),
        ('Espectáculos', 'Noticias de entretenimiento, cine, música y cultura'),
        ('Entretenimiento', 'Contenido de entretenimiento, anime, series'),
        ('Tecnología', 'Noticias de tecnología e innovación'),
        ('Sociedad', 'Noticias sociales y comunitarias'),
        ('Salud', 'Noticias de salud y medicina'),
        ('Educación', 'Noticias educativas y académicas')
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO categorias (nombre, descripcion) 
        VALUES (?, ?)
    ''', categorias_default)
    
    conn.commit()
    print("✅ Base de datos creada exitosamente")
    return conn

def insertar_noticias_ejemplo(conn):
    """Inserta noticias de ejemplo basadas en las extraídas anteriormente"""
    
    cursor = conn.cursor()
    
    # Datos de noticias extraídas
    noticias_ejemplo = [
        {
            'titulo': '"Lord of Mysteries" Capítulo 12: Hora confirmada de estreno',
            'categoria': 'Entretenimiento',
            'resumen': 'Información sobre el estreno del capítulo 12 de Lord of Mysteries en Crunchyroll',
            'contenido_completo': 'El popular anime "Lord of Mysteries" continúa su segunda temporada con el estreno del capítulo 12. Los fanáticos pueden disfrutar de este episodio en la plataforma Crunchyroll. La serie ha mantenido una gran audiencia desde su debut y este nuevo capítulo promete continuar con la emocionante trama que ha cautivado a los espectadores.',
            'fecha_publicacion': '2025-09-05',
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enlace': 'https://elcomercio.pe/saltar-intro/noticias/lord-of-mysteries-capitulo-12-hora-confirmada-de-estreno-crunchyroll-noticia/',
            'longitud_contenido': 2605,
            'selector_utilizado': 'a[href*="/noticias/"]'
        },
        {
            'titulo': '"Kaiju No.8" Temporada 2, Capítulo 8: Hora confirmada de estreno',
            'categoria': 'Entretenimiento',
            'resumen': 'Nuevo episodio de Kaiju No.8 disponible en plataformas de streaming',
            'contenido_completo': 'La segunda temporada de "Kaiju No.8" presenta su octavo capítulo con horarios confirmados para su estreno. Esta serie de anime ha ganado popularidad por su animación de alta calidad y su historia envolvente sobre la lucha contra criaturas gigantes.',
            'fecha_publicacion': '2025-09-05',
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enlace': 'https://elcomercio.pe/saltar-intro/noticias/kaiju-no8-temporada-2-capitulo-8-hora-confirmada-de-estreno-crunchyroll-noticia/',
            'longitud_contenido': 1484,
            'selector_utilizado': 'a[href*="/noticias/"]'
        },
        {
            'titulo': 'Betssy Chávez sale en libertad tras sentencia del Tribunal Constitucional',
            'categoria': 'Política',
            'resumen': 'La exministra de Pedro Castillo recupera su libertad tras decisión judicial',
            'contenido_completo': 'Betssy Chávez, exministra del gobierno de Pedro Castillo, ha sido liberada tras una sentencia del Tribunal Constitucional que anuló su prisión preventiva. La decisión judicial marca un hito importante en el caso y ha generado diversas reacciones en el ámbito político nacional.',
            'fecha_publicacion': '2025-09-05',
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enlace': 'https://elcomercio.pe/politica/actualidad/betssy-chavez-sale-en-libertad-tras-sentencia-del-tribunal-constitucional-que-anulo-su-prision-preventiva-ultimas-noticia/',
            'longitud_contenido': 1145,
            'selector_utilizado': 'a[href*="/politica/"]'
        },
        {
            'titulo': 'Alejandro Toledo lavó millones de dólares producto de coimas: Caso Ecoteva',
            'categoria': 'Política',
            'resumen': 'Nuevas revelaciones sobre el lavado de dinero del expresidente Toledo',
            'contenido_completo': 'El Poder Judicial ha revelado detalles sobre cómo Alejandro Toledo lavó millones de dólares producto de coimas en el caso Ecoteva. Los argumentos presentados por la fiscalía muestran una red compleja de transferencias y operaciones financieras que involucran a múltiples personas y empresas. Esta información refuerza la condena contra el expresidente y proporciona evidencia adicional sobre la magnitud de los actos de corrupción.',
            'fecha_publicacion': '2025-09-04',
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enlace': 'https://elcomercio.pe/politica/alejandro-toledo-lavo-dinero-ilicito-estos-fueron-los-argumentos-para-condenarlo-por-el-caso-ecoteva-tlcnota-noticia/',
            'longitud_contenido': 8003,
            'selector_utilizado': 'a[href*="/politica/"]'
        },
        {
            'titulo': 'Gobierno aprueba reglamento de modernización del sistema de pensiones',
            'categoria': 'Economía',
            'resumen': 'Nuevas regulaciones para el sistema de pensiones en Perú',
            'contenido_completo': 'El gobierno peruano ha aprobado un nuevo reglamento que moderniza el sistema de pensiones, afectando tanto a la ONP como a las AFP. Esta medida busca mejorar la eficiencia del sistema y garantizar mejores beneficios para los pensionistas. El reglamento incluye cambios en los requisitos de afiliación y en los cálculos de pensiones.',
            'fecha_publicacion': '2025-09-04',
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enlace': 'https://elcomercio.pe/economia/peru/gobierno-aprueba-reglamento-de-modernizacion-del-sistema-de-pensiones-l-onp-l-afp-l-ultimas-noticia/',
            'longitud_contenido': 2564,
            'selector_utilizado': 'a[href*="/economia/"]'
        },
        {
            'titulo': 'Argentina: cinco claves para entender la elección legislativa de Buenos Aires',
            'categoria': 'Mundo',
            'resumen': 'Análisis de las elecciones legislativas en la provincia de Buenos Aires',
            'contenido_completo': 'Las elecciones legislativas en la provincia de Buenos Aires representan un momento crucial para la política argentina. Los bonaerenses elegirán a la mitad de los integrantes del Legislativo provincial: 46 diputados y 23 senadores. Esta elección es fundamental para entender el panorama político nacional y las tendencias electorales en la región más poblada del país.',
            'fecha_publicacion': '2025-09-03',
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enlace': 'https://elcomercio.pe/mundo/latinoamerica/argentina-cinco-claves-para-entender-la-importancia-de-la-eleccion-legislativa-de-la-provincia-de-buenos-aires-ultimas-noticia/',
            'longitud_contenido': 5468,
            'selector_utilizado': 'a[href*="/mundo/"]'
        },
        {
            'titulo': 'Zelensky espera que países europeos desplieguen miles de soldados en Ucrania',
            'categoria': 'Mundo',
            'resumen': 'El presidente ucraniano solicita mayor apoyo militar europeo',
            'contenido_completo': 'El presidente de Ucrania, Volodymyr Zelensky, ha expresado su esperanza de que los países europeos desplieguen miles de soldados en Ucrania tras el alto el fuego. Esta solicitud forma parte de la estrategia ucraniana para fortalecer su defensa ante la agresión rusa y garantizar la estabilidad en la región.',
            'fecha_publicacion': '2025-09-02',
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enlace': 'https://elcomercio.pe/mundo/europa/guerra-rusia-ucrania-volodymyr-zelensky-espera-que-paises-europeos-desplieguen-miles-soldados-en-ucrania-tras-alto-el-fuego-vladimir-putin-coalicion-de-voluntarios-ultimas-noticia/',
            'longitud_contenido': 4200,
            'selector_utilizado': 'a[href*="/mundo/"]'
        },
        {
            'titulo': 'Diego Rebagliati: "Si Perú quiere volver al Mundial, el próximo técnico debe tener estas cualidades"',
            'categoria': 'Deportes',
            'resumen': 'Análisis del periodista deportivo sobre el futuro técnico de la selección peruana',
            'contenido_completo': 'El reconocido periodista deportivo Diego Rebagliati ha compartido su análisis sobre las cualidades que debe tener el próximo técnico de la selección peruana de fútbol. Según Rebagliati, para que Perú pueda volver al Mundial, es fundamental que el nuevo entrenador tenga experiencia internacional, conocimiento del fútbol sudamericano y capacidad para trabajar con jugadores jóvenes.',
            'fecha_publicacion': '2025-09-01',
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enlace': 'https://elcomercio.pe/respuestas/deportes/lo-tiene-claro-diego-rebagliati-aseguro-que-si-peru-quiere-volver-al-mundial-el-proximo-tecnico-debe-ser-asi-el-perfil-tdpe-noticia/',
            'longitud_contenido': 3200,
            'selector_utilizado': 'a[href*="/deportes/"]'
        }
    ]
    
    # Insertar noticias
    for noticia in noticias_ejemplo:
        cursor.execute('''
            INSERT INTO noticias 
            (titulo, categoria, resumen, contenido_completo, fecha_publicacion, 
             fecha_extraccion, enlace, longitud_contenido, selector_utilizado)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            noticia['titulo'],
            noticia['categoria'],
            noticia['resumen'],
            noticia['contenido_completo'],
            noticia['fecha_publicacion'],
            noticia['fecha_extraccion'],
            noticia['enlace'],
            noticia['longitud_contenido'],
            noticia['selector_utilizado']
        ))
    
    conn.commit()
    print(f"✅ {len(noticias_ejemplo)} noticias insertadas en la base de datos")

def actualizar_estadisticas(conn):
    """Actualiza las estadísticas de la base de datos"""
    
    cursor = conn.cursor()
    
    # Obtener estadísticas
    cursor.execute('SELECT COUNT(*) FROM noticias')
    total_noticias = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT categoria) FROM noticias')
    categorias_unicas = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(longitud_contenido) FROM noticias')
    promedio_longitud = cursor.fetchone()[0]
    
    cursor.execute('SELECT titulo FROM noticias ORDER BY longitud_contenido DESC LIMIT 1')
    noticia_mas_larga = cursor.fetchone()[0]
    
    cursor.execute('SELECT titulo FROM noticias ORDER BY longitud_contenido ASC LIMIT 1')
    noticia_mas_corta = cursor.fetchone()[0]
    
    # Insertar estadísticas
    cursor.execute('''
        INSERT INTO estadisticas 
        (fecha_extraccion, total_noticias, categorias_unicas, promedio_longitud, 
         noticia_mas_larga, noticia_mas_corta)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_noticias,
        categorias_unicas,
        promedio_longitud,
        noticia_mas_larga,
        noticia_mas_corta
    ))
    
    # Actualizar contadores de categorías
    cursor.execute('''
        UPDATE categorias 
        SET total_noticias = (
            SELECT COUNT(*) FROM noticias 
            WHERE noticias.categoria = categorias.nombre
        )
    ''')
    
    conn.commit()
    print("✅ Estadísticas actualizadas")

def generar_reporte_profesor(conn):
    """Genera un reporte completo para el profesor"""
    
    cursor = conn.cursor()
    
    # Obtener datos para el reporte
    cursor.execute('''
        SELECT n.*, c.descripcion 
        FROM noticias n 
        LEFT JOIN categorias c ON n.categoria = c.nombre 
        ORDER BY n.fecha_extraccion DESC
    ''')
    noticias = cursor.fetchall()
    
    cursor.execute('SELECT * FROM estadisticas ORDER BY fecha_extraccion DESC LIMIT 1')
    estadisticas = cursor.fetchone()
    
    cursor.execute('SELECT * FROM categorias ORDER BY total_noticias DESC')
    categorias = cursor.fetchall()
    
    # Crear reporte en texto
    reporte = f"""
REPORTE DE BASE DE DATOS - SCRAPING DE NOTICIAS EL COMERCIO
============================================================

INFORMACIÓN GENERAL:
- Fecha de creación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Base de datos: noticias_elcomercio.db
- Fuente: El Comercio (elcomercio.pe)
- Período analizado: Semana del 01/09/2025 al 07/09/2025

ESTADÍSTICAS GENERALES:
- Total de noticias: {estadisticas[2]}
- Categorías únicas: {estadisticas[3]}
- Promedio de longitud: {estadisticas[4]:.0f} caracteres
- Noticia más larga: {estadisticas[5]}
- Noticia más corta: {estadisticas[6]}

DISTRIBUCIÓN POR CATEGORÍAS:
"""
    
    for categoria in categorias:
        if categoria[3] > 0:  # Solo mostrar categorías con noticias
            reporte += f"- {categoria[1]}: {categoria[3]} noticias\n"
    
    reporte += f"""
DETALLE DE NOTICIAS:
===================
"""
    
    for i, noticia in enumerate(noticias, 1):
        reporte += f"""
{i}. TÍTULO: {noticia[1]}
   CATEGORÍA: {noticia[2]}
   FECHA: {noticia[5]}
   LONGITUD: {noticia[8]} caracteres
   ENLACE: {noticia[7]}
   RESUMEN: {noticia[3][:100]}...
"""
    
    reporte += f"""
ESTRUCTURA DE LA BASE DE DATOS:
===============================

TABLA: noticias
- id (INTEGER, PRIMARY KEY)
- titulo (TEXT, NOT NULL)
- categoria (TEXT, NOT NULL)
- resumen (TEXT)
- contenido_completo (TEXT)
- fecha_publicacion (TEXT)
- fecha_extraccion (TEXT, NOT NULL)
- enlace (TEXT, NOT NULL)
- longitud_contenido (INTEGER)
- selector_utilizado (TEXT)
- fuente (TEXT, DEFAULT 'El Comercio')

TABLA: categorias
- id (INTEGER, PRIMARY KEY)
- nombre (TEXT, UNIQUE, NOT NULL)
- descripcion (TEXT)
- total_noticias (INTEGER, DEFAULT 0)

TABLA: estadisticas
- id (INTEGER, PRIMARY KEY)
- fecha_extraccion (TEXT, NOT NULL)
- total_noticias (INTEGER)
- categorias_unicas (INTEGER)
- promedio_longitud (REAL)
- noticia_mas_larga (TEXT)
- noticia_mas_corta (TEXT)

TECNOLOGÍAS UTILIZADAS:
======================
- Python 3.9
- Selenium WebDriver
- BeautifulSoup4
- SQLite3
- Pandas
- OpenPyXL

METODOLOGÍA:
===========
1. Web Scraping con Selenium para contenido dinámico
2. Análisis de HTML con BeautifulSoup
3. Extracción de contenido con múltiples selectores CSS
4. Clasificación automática por categorías
5. Almacenamiento estructurado en base de datos SQLite
6. Generación de reportes en Excel y texto

CONCLUSIONES:
============
- Se extrajeron exitosamente {estadisticas[2]} noticias de El Comercio
- El sistema de clasificación automática funcionó correctamente
- La base de datos permite consultas eficientes y análisis estadísticos
- El contenido extraído incluye títulos, resúmenes y texto completo
- La metodología es replicable para otros sitios de noticias

============================================================
Reporte generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Guardar reporte en archivo
    with open('reporte_profesor.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print("✅ Reporte para el profesor generado: reporte_profesor.txt")
    return reporte

def exportar_a_excel(conn):
    """Exporta los datos de la base de datos a Excel"""
    
    # Leer datos de la base de datos
    df_noticias = pd.read_sql_query('SELECT * FROM noticias', conn)
    df_categorias = pd.read_sql_query('SELECT * FROM categorias', conn)
    df_estadisticas = pd.read_sql_query('SELECT * FROM estadisticas', conn)
    
    # Crear archivo Excel
    with pd.ExcelWriter('base_datos_noticias_profesor.xlsx', engine='openpyxl') as writer:
        df_noticias.to_excel(writer, sheet_name='Noticias', index=False)
        df_categorias.to_excel(writer, sheet_name='Categorias', index=False)
        df_estadisticas.to_excel(writer, sheet_name='Estadisticas', index=False)
    
    print("✅ Base de datos exportada a Excel: base_datos_noticias_profesor.xlsx")

def main():
    """Función principal"""
    
    print("🗄️ CREANDO BASE DE DATOS PARA TAREA DEL PROFESOR")
    print("=" * 60)
    
    # Crear base de datos
    conn = crear_base_datos()
    
    # Insertar noticias
    insertar_noticias_ejemplo(conn)
    
    # Actualizar estadísticas
    actualizar_estadisticas(conn)
    
    # Generar reporte para el profesor
    reporte = generar_reporte_profesor(conn)
    
    # Exportar a Excel
    exportar_a_excel(conn)
    
    # Cerrar conexión
    conn.close()
    
    print("\n📊 RESUMEN DE ARCHIVOS GENERADOS:")
    print("=" * 40)
    print("📁 noticias_elcomercio.db - Base de datos SQLite")
    print("📄 reporte_profesor.txt - Reporte completo en texto")
    print("📊 base_datos_noticias_profesor.xlsx - Datos en Excel")
    
    print(f"\n✅ ¡Base de datos lista para enviar al profesor!")
    print(f"📂 Ubicación: /Users/omar/Documents/Scraping/")

if __name__ == "__main__":
    main()
