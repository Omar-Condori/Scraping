from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime

def extraer_categoria_desde_url(url):
    """Extrae la categoría basándose en la URL de la noticia"""
    
    # Mapeo de URLs a categorías
    categorias_url = {
        '/politica/': 'Política',
        '/economia/': 'Economía', 
        '/mundo/': 'Mundo',
        '/deportes/': 'Deportes',
        '/espectaculos/': 'Espectáculos',
        '/tecnologia/': 'Tecnología',
        '/sociedad/': 'Sociedad',
        '/cultura/': 'Cultura',
        '/salud/': 'Salud',
        '/educacion/': 'Educación',
        '/turismo/': 'Turismo',
        '/gastronomia/': 'Gastronomía',
        '/saltar-intro/': 'Entretenimiento',
        '/noticias/': 'Noticias Generales'
    }
    
    for path, categoria in categorias_url.items():
        if path in url:
            return categoria
    
    return 'Sin categoría'

def extraer_categoria_desde_titulo(titulo):
    """Extrae categoría basándose en palabras clave del título"""
    
    palabras_clave = {
        'congreso': 'Política',
        'presidente': 'Política', 
        'gobierno': 'Política',
        'ministro': 'Política',
        'elecciones': 'Política',
        'partido': 'Política',
        'dólar': 'Economía',
        'inflación': 'Economía',
        'empresa': 'Economía',
        'mercado': 'Economía',
        'banco': 'Economía',
        'futbol': 'Deportes',
        'fútbol': 'Deportes',
        'deporte': 'Deportes',
        'mundial': 'Deportes',
        'liga': 'Deportes',
        'película': 'Espectáculos',
        'serie': 'Espectáculos',
        'actor': 'Espectáculos',
        'música': 'Espectáculos',
        'anime': 'Entretenimiento',
        'manga': 'Entretenimiento',
        'temporada': 'Entretenimiento',
        'capítulo': 'Entretenimiento',
        'covid': 'Salud',
        'salud': 'Salud',
        'hospital': 'Salud',
        'vacuna': 'Salud',
        'terremoto': 'Sociedad',
        'accidente': 'Sociedad',
        'crimen': 'Sociedad',
        'robo': 'Sociedad'
    }
    
    titulo_lower = titulo.lower()
    for palabra, categoria in palabras_clave.items():
        if palabra in titulo_lower:
            return categoria
    
    return None

# Configuración de Chrome
chrome_options = Options()
chrome_options.add_argument("--headless=new")  
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

service = Service("/opt/homebrew/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Abrir página de últimas noticias
    print("🔍 Accediendo a El Comercio...")
    driver.get("https://elcomercio.pe/ultimas-noticias/")
    time.sleep(8)  # Esperar más tiempo para que cargue todo el contenido
    
    # Obtener el HTML de la página
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    print("📰 Analizando estructura de la página...")
    
    # Buscar diferentes tipos de selectores que podrían contener noticias
    selectores_posibles = [
        "article",
        ".story-item",
        ".story-card", 
        ".Card-link",
        ".StoryLink",
        ".stories__item-link",
        "a[href*='/noticias/']",
        "a[href*='/politica/']",
        "a[href*='/economia/']",
        "a[href*='/mundo/']",
        "a[href*='/deportes/']",
        "a[href*='/espectaculos/']",
        ".headline",
        ".title",
        "h1 a",
        "h2 a", 
        "h3 a",
        ".news-item",
        ".noticia"
    ]
    
    noticias_encontradas = []
    
    for selector in selectores_posibles:
        elementos = soup.select(selector)
        if elementos:
            print(f"✅ Selector '{selector}' encontró {len(elementos)} elementos")
            
            for elem in elementos[:10]:  # Limitar a 10 por selector
                # Buscar enlaces y títulos
                if elem.name == 'a':
                    enlace = elem.get('href', '')
                    titulo = elem.get_text(strip=True)
                else:
                    enlace_elem = elem.find('a')
                    if enlace_elem:
                        enlace = enlace_elem.get('href', '')
                        titulo = enlace_elem.get_text(strip=True)
                    else:
                        continue
                
                # Filtrar enlaces válidos de noticias
                if (enlace and 
                    ('/noticias/' in enlace or '/politica/' in enlace or 
                     '/economia/' in enlace or '/mundo/' in enlace or
                     '/deportes/' in enlace or '/espectaculos/' in enlace or
                     '/saltar-intro/' in enlace) and
                    titulo and len(titulo) > 10 and
                    not any(palabra in titulo.lower() for palabra in ['publicidad', 'anuncio', 'sponsor', 'cookies'])):
                    
                    # Completar URL si es relativa
                    if enlace.startswith("/"):
                        enlace = "https://elcomercio.pe" + enlace
                    
                    # Determinar categoría
                    categoria_url = extraer_categoria_desde_url(enlace)
                    categoria_titulo = extraer_categoria_desde_titulo(titulo)
                    
                    # Priorizar categoría del título si es más específica
                    categoria_final = categoria_titulo if categoria_titulo else categoria_url
                    
                    noticias_encontradas.append({
                        'titulo': titulo,
                        'enlace': enlace,
                        'categoria': categoria_final,
                        'selector': selector
                    })
    
    # Eliminar duplicados basándose en el título
    noticias_unicas = []
    titulos_vistos = set()
    
    for noticia in noticias_encontradas:
        titulo_normalizado = re.sub(r'[^\w\s]', '', noticia['titulo'].lower())
        if titulo_normalizado not in titulos_vistos:
            titulos_vistos.add(titulo_normalizado)
            noticias_unicas.append(noticia)
    
    # Mostrar las 5 noticias más importantes con título y categoría
    print(f"\n📊 REPORTE DE NOTICIAS CON TÍTULO Y CATEGORÍA - EL COMERCIO")
    print("=" * 90)
    print(f"Total de noticias encontradas: {len(noticias_unicas)}")
    print("=" * 90)
    
    if noticias_unicas:
        print("\n🔥 LAS 5 NOTICIAS MÁS IMPORTANTES CON TÍTULO Y CATEGORÍA:")
        print("-" * 90)
        
        for i, noticia in enumerate(noticias_unicas[:5], 1):
            print(f"\n{i}. 📰 TÍTULO: {noticia['titulo']}")
            print(f"   🏷️  CATEGORÍA: {noticia['categoria']}")
            print(f"   🔗 ENLACE: {noticia['enlace']}")
            print(f"   📍 SELECTOR: {noticia['selector']}")
            print("-" * 60)
        
        # Mostrar resumen por categorías
        print(f"\n📈 RESUMEN POR CATEGORÍAS:")
        print("-" * 40)
        categorias_count = {}
        for noticia in noticias_unicas:
            cat = noticia['categoria']
            categorias_count[cat] = categorias_count.get(cat, 0) + 1
        
        for categoria, count in sorted(categorias_count.items(), key=lambda x: x[1], reverse=True):
            print(f"   {categoria}: {count} noticias")
            
    else:
        print("\n❌ No se encontraron noticias con los selectores actuales.")
        print("💡 La página puede haber cambiado su estructura.")

except Exception as e:
    print(f"❌ Error: {e}")
    
finally:
    driver.quit()
    print("\n✅ Proceso completado.")
