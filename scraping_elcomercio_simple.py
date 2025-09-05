from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
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

# Nuevo selector: busca los titulares dentro de los divs de clase 'story-item' o 'story-card'
articulos = soup.select("a.Card-link, a.StoryLink, a.stories__item-link")  # prueba varios selectores
print("Noticias encontradas:", len(articulos))

for art in articulos[:10]:
    titulo = art.get_text(strip=True)
    enlace = art["href"]
    if enlace.startswith("/"):
        enlace = "https://elcomercio.pe" + enlace
    print(f"{titulo} -> {enlace}")

