from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime

def buscar_fecha_en_elemento(elemento):
    """Busca fechas en diferentes formatos dentro de un elemento HTML"""
    
    # Selectores comunes para fechas
    selectores_fecha = [
        "time",
        ".date",
        ".fecha", 
        ".timestamp",
        ".publish-date",
        ".publication-date",
        "[datetime]",
        ".story-date",
        ".news-date"
    ]
    
    # Buscar en el elemento actual
    for selector in selectores_fecha:
        fecha_elem = elemento.select_one(selector)
        if fecha_elem:
            # Intentar obtener fecha del atributo datetime
            if fecha_elem.has_attr('datetime'):
                return fecha_elem['datetime']
            # O del texto del elemento
            fecha_texto = fecha_elem.get_text(strip=True)
            if fecha_texto:
                return fecha_texto
    
    # Buscar en elementos padre
    padre = elemento.parent
    while padre and padre.name != 'body':
        for selector in selectores_fecha:
            fecha_elem = padre.select_one(selector)
            if fecha_elem:
                if fecha_elem.has_attr('datetime'):
                    return fecha_elem['datetime']
                fecha_texto = fecha_elem.get_text(strip=True)
                if fecha_texto:
                    return fecha_texto
        padre = padre.parent
    
    # Buscar en elementos hermanos
    if elemento.parent:
        for hermano in elemento.parent.find_all():
            if hermano != elemento:
                for selector in selectores_fecha:
                    fecha_elem = hermano.select_one(selector)
                    if fecha_elem:
                        if fecha_elem.has_attr('datetime'):
                            return fecha_elem['datetime']
                        fecha_texto = fecha_elem.get_text(strip=True)
                        if fecha_texto:
                            return fecha_texto
    
    # Buscar patrones de fecha en el texto
    texto_completo = elemento.get_text()
    patrones_fecha = [
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2} de \w+ de \d{4}',
        r'hace \d+ \w+',
        r'\d{1,2}:\d{2}',
        r'hoy',
        r'ayer'
    ]
    
    for patron in patrones_fecha:
        match = re.search(patron, texto_completo, re.IGNORECASE)
        if match:
            return match.group()
    
    return "Fecha no encontrada"

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
                    # Buscar fecha en el elemento padre o hermanos
                    fecha = buscar_fecha_en_elemento(elem)
                else:
                    enlace_elem = elem.find('a')
                    if enlace_elem:
                        enlace = enlace_elem.get('href', '')
                        titulo = enlace_elem.get_text(strip=True)
                        # Buscar fecha en el elemento actual
                        fecha = buscar_fecha_en_elemento(elem)
                    else:
                        continue
                
                # Filtrar enlaces válidos de noticias
                if (enlace and 
                    ('/noticias/' in enlace or '/politica/' in enlace or 
                     '/economia/' in enlace or '/mundo/' in enlace or
                     '/deportes/' in enlace or '/espectaculos/' in enlace) and
                    titulo and len(titulo) > 10 and
                    not any(palabra in titulo.lower() for palabra in ['publicidad', 'anuncio', 'sponsor', 'cookies'])):
                    
                    # Completar URL si es relativa
                    if enlace.startswith("/"):
                        enlace = "https://elcomercio.pe" + enlace
                    
                    noticias_encontradas.append({
                        'titulo': titulo,
                        'enlace': enlace,
                        'fecha': fecha,
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
    
    # Mostrar las 5 noticias más importantes con fechas
    print(f"\n📊 REPORTE DE NOTICIAS CON FECHAS - EL COMERCIO")
    print("=" * 80)
    print(f"Total de noticias encontradas: {len(noticias_unicas)}")
    print("=" * 80)
    
    if noticias_unicas:
        print("\n🔥 LAS 5 NOTICIAS MÁS IMPORTANTES CON FECHAS:")
        print("-" * 80)
        
        for i, noticia in enumerate(noticias_unicas[:5], 1):
            print(f"\n{i}. {noticia['titulo']}")
            print(f"   📅 Fecha: {noticia['fecha']}")
            print(f"   🔗 {noticia['enlace']}")
            print(f"   📍 Selector: {noticia['selector']}")
            print("-" * 50)
    else:
        print("\n❌ No se encontraron noticias con los selectores actuales.")
        print("💡 La página puede haber cambiado su estructura.")

except Exception as e:
    print(f"❌ Error: {e}")
    
finally:
    driver.quit()
    print("\n✅ Proceso completado.")
