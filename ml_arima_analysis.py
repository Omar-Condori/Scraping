#!/usr/bin/env python3
"""
Script específico para análisis con ARIMA (versión simplificada)
Implementa ARIMA básico para análisis temporal de noticias
"""

import sys
import argparse
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Para ARIMA básico
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️ ARIMA no disponible. Instala: pip install statsmodels")

class ARIMAAnalyzer:
    def __init__(self, db_url="postgresql://omar@localhost:5432/scraping_db"):
        self.db_url = db_url
        self.conn = None
        self.df = None
        
    def connect_db(self):
        try:
            self.conn = psycopg2.connect(self.db_url)
            return True
        except Exception as e:
            print(f"❌ Error conectando a la base de datos: {e}")
            return False
    
    def load_data(self):
        if not self.conn:
            if not self.connect_db():
                return False
        
        query = """
        SELECT 
            id, title, content, description, category, source, 
            "imageUrl", author, "publishedAt", "scrapedAt"
        FROM articles 
        WHERE content IS NOT NULL AND LENGTH(content) > 50
        ORDER BY "scrapedAt" DESC
        """
        
        try:
            self.df = pd.read_sql(query, self.conn)
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def analyze_temporal_trends_arima(self):
        """Análisis de tendencias temporales con ARIMA"""
        print("📈 ANÁLISIS DE TENDENCIAS TEMPORALES CON ARIMA")
        print("=" * 70)
        
        if not ARIMA_AVAILABLE:
            print("❌ ARIMA no está disponible. Instala: pip install statsmodels")
            return
        
        # Preparar datos temporales
        self.df['scrapedAt'] = pd.to_datetime(self.df['scrapedAt'])
        self.df['date'] = self.df['scrapedAt'].dt.date
        
        # Crear series temporales diarias
        daily_counts = self.df.groupby('date').size()
        daily_counts.index = pd.to_datetime(daily_counts.index)
        
        print(f"📊 Datos temporales:")
        print(f"   Período: {daily_counts.index.min()} a {daily_counts.index.max()}")
        print(f"   Días con datos: {len(daily_counts)}")
        print(f"   Promedio diario: {daily_counts.mean():.2f} artículos")
        
        if len(daily_counts) < 10:
            print("❌ Insuficientes datos temporales para ARIMA")
            return
        
        try:
            # Dividir en train/test
            split_point = int(len(daily_counts) * 0.8)
            train_data = daily_counts[:split_point]
            test_data = daily_counts[split_point:]
            
            print(f"   Datos de entrenamiento: {len(train_data)} días")
            print(f"   Datos de prueba: {len(test_data)} días")
            
            # Ajustar modelo ARIMA básico (1,1,1)
            print("\n🔄 Ajustando modelo ARIMA(1,1,1)...")
            model = ARIMA(train_data, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Hacer predicciones
            predictions = model_fit.forecast(steps=len(test_data))
            
            # Calcular métricas
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mse)
            
            print(f"\n📊 Métricas de ARIMA:")
            print(f"   MSE: {mse:.2f}")
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            
            # Mostrar predicciones vs realidad
            print(f"\n🔮 Predicciones vs Realidad:")
            for i, (date, actual, predicted) in enumerate(zip(test_data.index, test_data.values, predictions)):
                print(f"   {date.strftime('%Y-%m-%d')}: Real={actual}, Predicho={predicted:.1f}")
                if i >= 4:  # Mostrar solo los primeros 5
                    break
            
            # Análisis de estacionariedad
            print(f"\n📊 Análisis de Estacionariedad:")
            adf_result = adfuller(train_data)
            print(f"   Estadística ADF: {adf_result[0]:.4f}")
            print(f"   p-valor: {adf_result[1]:.4f}")
            if adf_result[1] < 0.05:
                print("   ✅ Serie es estacionaria")
            else:
                print("   ⚠️ Serie no es estacionaria")
            
            # Predicción futura
            print(f"\n🔮 Predicción para los próximos 7 días:")
            future_predictions = model_fit.forecast(steps=7)
            last_date = daily_counts.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
            
            for date, pred in zip(future_dates, future_predictions):
                print(f"   {date.strftime('%Y-%m-%d')}: {pred:.1f} artículos")
                
        except Exception as e:
            print(f"❌ Error en análisis ARIMA: {e}")

    def analyze_category_trends_arima(self):
        """Análisis de tendencias por categoría con ARIMA"""
        print("📊 ANÁLISIS DE TENDENCIAS POR CATEGORÍA CON ARIMA")
        print("=" * 70)
        
        if not ARIMA_AVAILABLE:
            print("❌ ARIMA no está disponible. Instala: pip install statsmodels")
            return
        
        # Preparar datos temporales por categoría
        self.df['scrapedAt'] = pd.to_datetime(self.df['scrapedAt'])
        self.df['date'] = self.df['scrapedAt'].dt.date
        
        # Filtrar categorías válidas
        df_categorized = self.df[self.df['category'].notna() & (self.df['category'] != 'Noticias')].copy()
        
        if len(df_categorized) < 50:
            print("❌ Insuficientes datos categorizados")
            return
        
        # Crear series temporales por categoría
        category_trends = df_categorized.groupby(['date', 'category']).size().unstack(fill_value=0)
        category_trends.index = pd.to_datetime(category_trends.index)
        
        print(f"📊 Categorías analizadas: {len(category_trends.columns)}")
        print(f"   Período: {category_trends.index.min()} a {category_trends.index.max()}")
        
        # Analizar cada categoría
        for category in category_trends.columns:
            series = category_trends[category]
            if len(series) >= 10 and series.sum() > 5:  # Mínimo de datos
                print(f"\n🔍 Analizando categoría: {category}")
                print(f"   Total artículos: {series.sum()}")
                print(f"   Promedio diario: {series.mean():.2f}")
                
                try:
                    # Dividir en train/test
                    split_point = max(5, int(len(series) * 0.8))
                    train_data = series[:split_point]
                    test_data = series[split_point:]
                    
                    if len(train_data) >= 5 and len(test_data) > 0:
                        # Ajustar modelo ARIMA básico
                        model = ARIMA(train_data, order=(1, 1, 1))
                        model_fit = model.fit()
                        predictions = model_fit.forecast(steps=len(test_data))
                        
                        # Calcular métricas
                        mse = mean_squared_error(test_data, predictions)
                        print(f"   Modelo: ARIMA(1,1,1)")
                        print(f"   RMSE: {np.sqrt(mse):.2f}")
                        
                        # Mostrar tendencia
                        if len(test_data) > 0:
                            actual_trend = test_data.mean()
                            predicted_trend = predictions.mean()
                            print(f"   Tendencia real: {actual_trend:.2f}")
                            print(f"   Tendencia predicha: {predicted_trend:.2f}")
                    else:
                        print("   ⚠️ Datos insuficientes para análisis")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")

    def analyze_sentiment_trends_arima(self):
        """Análisis de tendencias de sentimientos con ARIMA"""
        print("😊 ANÁLISIS DE TENDENCIAS DE SENTIMIENTOS CON ARIMA")
        print("=" * 70)
        
        if not ARIMA_AVAILABLE:
            print("❌ ARIMA no está disponible. Instala: pip install statsmodels")
            return
        
        # Preparar datos temporales de sentimientos
        self.df['scrapedAt'] = pd.to_datetime(self.df['scrapedAt'])
        self.df['date'] = self.df['scrapedAt'].dt.date
        
        # Calcular sentimiento promedio por día
        daily_sentiment = self.df.groupby('date')['sentiment_score'].mean()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        print(f"📊 Datos de sentimientos:")
        print(f"   Período: {daily_sentiment.index.min()} a {daily_sentiment.index.max()}")
        print(f"   Días con datos: {len(daily_sentiment)}")
        print(f"   Sentimiento promedio: {daily_sentiment.mean():.3f}")
        
        if len(daily_sentiment) < 10:
            print("❌ Insuficientes datos temporales para ARIMA")
            return
        
        try:
            # Dividir en train/test
            split_point = int(len(daily_sentiment) * 0.8)
            train_data = daily_sentiment[:split_point]
            test_data = daily_sentiment[split_point:]
            
            # Ajustar modelo ARIMA básico
            print("\n🔄 Ajustando modelo ARIMA para sentimientos...")
            model = ARIMA(train_data, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Hacer predicciones
            predictions = model_fit.forecast(steps=len(test_data))
            
            # Calcular métricas
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mse)
            
            print(f"\n📊 Métricas de ARIMA para sentimientos:")
            print(f"   MSE: {mse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            
            # Mostrar predicciones vs realidad
            print(f"\n🔮 Predicciones de sentimientos vs Realidad:")
            for i, (date, actual, predicted) in enumerate(zip(test_data.index, test_data.values, predictions)):
                sentiment_label = "positivo" if actual > 0.1 else "negativo" if actual < -0.1 else "neutral"
                pred_label = "positivo" if predicted > 0.1 else "negativo" if predicted < -0.1 else "neutral"
                print(f"   {date.strftime('%Y-%m-%d')}: Real={actual:.3f} ({sentiment_label}), Predicho={predicted:.3f} ({pred_label})")
                if i >= 4:  # Mostrar solo los primeros 5
                    break
            
            # Predicción futura
            print(f"\n🔮 Predicción de sentimientos para los próximos 7 días:")
            future_predictions = model_fit.forecast(steps=7)
            last_date = daily_sentiment.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
            
            for date, pred in zip(future_dates, future_predictions):
                sentiment_label = "positivo" if pred > 0.1 else "negativo" if pred < -0.1 else "neutral"
                print(f"   {date.strftime('%Y-%m-%d')}: {pred:.3f} ({sentiment_label})")
                
        except Exception as e:
            print(f"❌ Error en análisis ARIMA de sentimientos: {e}")

def main():
    parser = argparse.ArgumentParser(description='Análisis específico con ARIMA')
    parser.add_argument('--type', required=True, 
                       choices=['temporal', 'category', 'sentiment'],
                       help='Tipo de análisis a ejecutar')
    
    args = parser.parse_args()
    
    analyzer = ARIMAAnalyzer()
    
    if not analyzer.load_data():
        return
    
    analysis_type = args.type
    
    if analysis_type == 'temporal':
        analyzer.analyze_temporal_trends_arima()
    elif analysis_type == 'category':
        analyzer.analyze_category_trends_arima()
    elif analysis_type == 'sentiment':
        analyzer.analyze_sentiment_trends_arima()
    
    print(f"\n✅ Análisis ARIMA {analysis_type} completado exitosamente!")

if __name__ == "__main__":
    main()