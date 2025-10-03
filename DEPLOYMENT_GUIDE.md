# 🚀 Guía de Despliegue y Scraping Automático

## ✅ Cambios Subidos a GitHub

Los siguientes archivos han sido agregados para el despliegue automático:

- `.github/workflows/scraping.yml` - GitHub Actions para scraping automático
- `vercel.json` - Configuración de Vercel
- `scripts/scrape-cron.js` - Script de scraping para la nube
- `.gitignore` actualizado - Excluye archivos grandes de ML

## 🌐 Pasos para Desplegar en Vercel

### 1. Conectar con Vercel
1. Ve a [vercel.com](https://vercel.com)
2. Conecta tu cuenta de GitHub
3. Importa el repositorio `Omar-Condori/Scraping`
4. Vercel detectará automáticamente que es un proyecto Next.js

### 2. Configurar Variables de Entorno
En el dashboard de Vercel, agrega estas variables:

```
DATABASE_URL=postgresql://usuario:password@host:puerto/database
NODE_ENV=production
APP_URL=https://tu-app.vercel.app
```

### 3. Desplegar
- Vercel desplegará automáticamente
- Tu app estará disponible en: `https://scraping-news-omar.vercel.app`

## 🤖 Scraping Automático con GitHub Actions

### Configuración Automática
El scraping automático ya está configurado y se ejecutará:
- **Cada 6 horas** automáticamente
- **Manual** desde GitHub Actions

### Configurar Secretos en GitHub
1. Ve a tu repositorio en GitHub
2. Settings → Secrets and variables → Actions
3. Agrega el secreto:
   - `DATABASE_URL`: Tu URL de base de datos PostgreSQL

### Verificar Funcionamiento
1. Ve a la pestaña "Actions" en tu repositorio
2. Verás el workflow "Auto Scraping"
3. Se ejecutará automáticamente cada 6 horas

## 🗄️ Base de Datos en la Nube

### Opción 1: Railway (Recomendado)
1. Ve a [railway.app](https://railway.app)
2. Conecta GitHub
3. Crea un nuevo proyecto PostgreSQL
4. Copia la URL de conexión para Vercel

### Opción 2: Supabase
1. Ve a [supabase.com](https://supabase.com)
2. Crea un nuevo proyecto
3. Copia la URL de conexión

### Opción 3: Neon
1. Ve a [neon.tech](https://neon.tech)
2. Crea una base de datos PostgreSQL
3. Copia la URL de conexión

## 📊 Monitoreo

### Verificar Scraping
- **GitHub Actions**: Ve a Actions → Auto Scraping
- **Vercel**: Ve a Functions → Logs
- **Base de datos**: Verifica que se estén agregando artículos

### URLs Importantes
- **App**: `https://scraping-news-omar.vercel.app`
- **Dashboard**: `https://scraping-news-omar.vercel.app/dashboard`
- **Predicción ML**: `https://scraping-news-omar.vercel.app/prediction-dashboard`
- **Admin**: `https://scraping-news-omar.vercel.app/admin`

## 🔧 Solución de Problemas

### Error de Base de Datos
```bash
# Verificar conexión
psql "tu-database-url"
```

### Error de GitHub Actions
1. Verifica que el secreto `DATABASE_URL` esté configurado
2. Revisa los logs en la pestaña Actions

### Error de Vercel
1. Verifica las variables de entorno
2. Revisa los logs en Vercel Dashboard

## 🎯 Próximos Pasos

1. **Desplegar en Vercel** (5 minutos)
2. **Configurar base de datos en la nube** (10 minutos)
3. **Configurar secretos en GitHub** (2 minutos)
4. **¡Listo!** El scraping funcionará automáticamente

---

**¡Tu sistema de scraping automático está listo para funcionar 24/7 en la nube!** 🎉
