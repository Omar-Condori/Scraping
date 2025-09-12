import sqlite3
import pandas as pd
from datetime import datetime

def ejecutar_consultas_sql():
    """Ejecuta las consultas SQL principales y muestra los resultados"""
    
    # Conectar a la base de datos
    conn = sqlite3.connect('noticias_elcomercio.db')
    
    print("🗄️ EJECUTANDO CONSULTAS SQL - BASE DE DATOS NOTICIAS")
    print("=" * 60)
    
    # Consulta 1: Estadísticas generales
    print("\n📊 1. ESTADÍSTICAS GENERALES:")
    print("-" * 40)
    query1 = """
    SELECT 
        COUNT(*) as total_noticias,
        COUNT(DISTINCT categoria) as categorias_unicas,
        ROUND(AVG(longitud_contenido), 2) as promedio_longitud,
        MIN(longitud_contenido) as longitud_minima,
        MAX(longitud_contenido) as longitud_maxima
    FROM noticias
    """
    df1 = pd.read_sql_query(query1, conn)
    print(df1.to_string(index=False))
    
    # Consulta 2: Noticias por categoría
    print("\n📈 2. NOTICIAS POR CATEGORÍA:")
    print("-" * 40)
    query2 = """
    SELECT 
        categoria,
        COUNT(*) as total_noticias,
        ROUND(AVG(longitud_contenido), 2) as promedio_longitud
    FROM noticias
    GROUP BY categoria
    ORDER BY total_noticias DESC
    """
    df2 = pd.read_sql_query(query2, conn)
    print(df2.to_string(index=False))
    
    # Consulta 3: Noticias más largas
    print("\n📰 3. NOTICIAS MÁS LARGAS:")
    print("-" * 40)
    query3 = """
    SELECT 
        titulo,
        categoria,
        longitud_contenido,
        fecha_publicacion
    FROM noticias
    ORDER BY longitud_contenido DESC
    LIMIT 3
    """
    df3 = pd.read_sql_query(query3, conn)
    for _, row in df3.iterrows():
        print(f"• {row['titulo'][:60]}...")
        print(f"  Categoría: {row['categoria']} | Longitud: {row['longitud_contenido']} caracteres")
        print()
    
    # Consulta 4: Noticias por fecha
    print("\n📅 4. NOTICIAS POR FECHA:")
    print("-" * 40)
    query4 = """
    SELECT 
        fecha_publicacion,
        COUNT(*) as noticias_del_dia
    FROM noticias
    GROUP BY fecha_publicacion
    ORDER BY fecha_publicacion DESC
    """
    df4 = pd.read_sql_query(query4, conn)
    print(df4.to_string(index=False))
    
    # Consulta 5: Análisis de selectores
    print("\n🔍 5. ANÁLISIS DE SELECTORES UTILIZADOS:")
    print("-" * 40)
    query5 = """
    SELECT 
        selector_utilizado,
        COUNT(*) as veces_utilizado
    FROM noticias
    GROUP BY selector_utilizado
    ORDER BY veces_utilizado DESC
    """
    df5 = pd.read_sql_query(query5, conn)
    print(df5.to_string(index=False))
    
    # Consulta 6: Noticias con contenido rico
    print("\n💎 6. NOTICIAS CON CONTENIDO RICO (>5000 caracteres):")
    print("-" * 40)
    query6 = """
    SELECT 
        titulo,
        categoria,
        longitud_contenido
    FROM noticias
    WHERE longitud_contenido > 5000
    ORDER BY longitud_contenido DESC
    """
    df6 = pd.read_sql_query(query6, conn)
    if len(df6) > 0:
        for _, row in df6.iterrows():
            print(f"• {row['titulo'][:50]}...")
            print(f"  Categoría: {row['categoria']} | Longitud: {row['longitud_contenido']} caracteres")
            print()
    else:
        print("No hay noticias con más de 5000 caracteres")
    
    # Consulta 7: Resumen ejecutivo
    print("\n📋 7. RESUMEN EJECUTIVO:")
    print("-" * 40)
    query7 = """
    SELECT 
        'Total de noticias' as metrica,
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
    FROM noticias
    """
    df7 = pd.read_sql_query(query7, conn)
    for _, row in df7.iterrows():
        print(f"• {row['metrica']}: {row['valor']}")
    
    # Consulta 8: Verificar integridad
    print("\n✅ 8. VERIFICACIÓN DE INTEGRIDAD:")
    print("-" * 40)
    query8 = """
    SELECT 
        'Noticias en BD' as tabla,
        COUNT(*) as registros
    FROM noticias
    UNION ALL
    SELECT 
        'Categorías definidas',
        COUNT(*)
    FROM categorias
    UNION ALL
    SELECT 
        'Estadísticas generadas',
        COUNT(*)
    FROM estadisticas
    """
    df8 = pd.read_sql_query(query8, conn)
    print(df8.to_string(index=False))
    
    # Consulta 9: Top categorías por longitud
    print("\n🏆 9. TOP 3 CATEGORÍAS POR LONGITUD PROMEDIO:")
    print("-" * 40)
    query9 = """
    SELECT 
        categoria,
        COUNT(*) as total_noticias,
        ROUND(AVG(longitud_contenido), 2) as promedio_longitud,
        SUM(longitud_contenido) as total_caracteres
    FROM noticias
    GROUP BY categoria
    ORDER BY promedio_longitud DESC
    LIMIT 3
    """
    df9 = pd.read_sql_query(query9, conn)
    print(df9.to_string(index=False))
    
    # Consulta 10: Distribución temporal
    print("\n⏰ 10. DISTRIBUCIÓN TEMPORAL:")
    print("-" * 40)
    query10 = """
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
        END
    """
    df10 = pd.read_sql_query(query10, conn)
    print(df10.to_string(index=False))
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ CONSULTAS SQL EJECUTADAS EXITOSAMENTE")
    print(f"📅 Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def mostrar_estructura_bd():
    """Muestra la estructura de la base de datos"""
    
    conn = sqlite3.connect('noticias_elcomercio.db')
    cursor = conn.cursor()
    
    print("\n🏗️ ESTRUCTURA DE LA BASE DE DATOS:")
    print("=" * 50)
    
    # Mostrar tablas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tablas = cursor.fetchall()
    
    for tabla in tablas:
        print(f"\n📋 TABLA: {tabla[0]}")
        print("-" * 30)
        
        # Mostrar estructura de cada tabla
        cursor.execute(f"PRAGMA table_info({tabla[0]});")
        columnas = cursor.fetchall()
        
        for columna in columnas:
            print(f"  • {columna[1]} ({columna[2]}) - {'NOT NULL' if columna[3] else 'NULL'}")
    
    conn.close()

if __name__ == "__main__":
    mostrar_estructura_bd()
    ejecutar_consultas_sql()
