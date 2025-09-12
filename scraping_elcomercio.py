from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

# Configuración de Chrome
chrome_options = Options()
chrome_options.add_argument("--headless=new")  
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

service = Service("/opt/homebrew/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)

# Abrir página de últimas noticias
driver.get("https://elcomercio.pe/ultimas-noticias/")
time.sleep(5)  # espera que cargue todo el contenido

soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# Fecha de hoy para filtrar noticias
hoy = datetime.today().strftime("%Y-%m-%d")

# Extraer noticias
noticias = []
articulos = soup.select("article")
for art in articulos:
    titulo_tag = art.select_one("h2 a, h3 a")
    if not titulo_tag:
        continue
    titulo = titulo_tag.get_text(strip=True)
    enlace = titulo_tag["href"]
    if enlace.startswith("/"):
        enlace = "https://elcomercio.pe" + enlace

    # Extraer fecha si existe
    fecha_tag = art.select_one("time")
    fecha = fecha_tag["datetime"][:10] if fecha_tag and fecha_tag.has_attr("datetime") else "N/A"

    # Filtrar solo noticias de hoy
    if fecha == hoy:
        noticias.append({
            "fecha": fecha,
            "titulo": titulo,
            "url": enlace
        })

# Mostrar resultados en tabla
df = pd.DataFrame(noticias)
pd.set_option("display.max_colwidth", None)
print(df)

