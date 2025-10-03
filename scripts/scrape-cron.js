#!/usr/bin/env node

const axios = require('axios');

async function runScraping() {
  try {
    console.log('🚀 Iniciando scraping automático...');
    
    // URL de tu aplicación desplegada
    const baseUrl = process.env.APP_URL || 'https://scraping-news-omar.vercel.app';
    
    // Ejecutar scraping dinámico
    console.log('📰 Ejecutando scraping dinámico...');
    const dynamicResponse = await axios.post(`${baseUrl}/api/scrape-dynamic`);
    console.log('✅ Scraping dinámico completado:', dynamicResponse.data);
    
    // Ejecutar scraping estático
    console.log('📰 Ejecutando scraping estático...');
    const staticResponse = await axios.post(`${baseUrl}/api/scrape`);
    console.log('✅ Scraping estático completado:', staticResponse.data);
    
    console.log('🎉 Scraping automático completado exitosamente');
    console.log('Fecha:', new Date().toISOString());
    
  } catch (error) {
    console.error('❌ Error en scraping automático:', error.message);
    process.exit(1);
  }
}

// Ejecutar si es llamado directamente
if (require.main === module) {
  runScraping();
}

module.exports = runScraping;
