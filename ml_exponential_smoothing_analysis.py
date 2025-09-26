#!/usr/bin/env python3
"""
Script específico para análisis con Suavizado Exponencial
Implementa Suavizado Exponencial para análisis temporal de noticias
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

# Para Suavizado Exponencial
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    EXPONENTIAL_SMOOTHING_AVAILABLE = True
except ImportError:
    EXPONENTIAL_SMOOTHING_AVAILABLE = False
    print("⚠️ Suavizado Exponencial no disponible. Instala: pip install statsmodels")

class ExponentialSmoothingAnalyzer:
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
    
    def analyze_temporal_trends_exponential_smoothing(self):
        """Análisis de tendencias temporales con Suavizado Exponencial"""
        print("📈 ANÁLISIS DE TENDENCIAS TEMPORALES CON SUAVIZADO EXPONENCIAL")
        print("=" * 80)
        
        if not EXPONENTIAL_SMOOTHING_AVAILABLE:
            print("❌ Suavizado Exponencial no está disponible. Instala: pip install statsmodels")
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
            print("❌ Insuficientes datos temporales para Suavizado Exponencial")
            return
        
        try:
            # Dividir en train/test
            split_point = int(len(daily_counts) * 0.8)
            train_data = daily_counts[:split_point]
            test_data = daily_counts[split_point:]
            
            print(f"   Datos de entrenamiento: {len(train_data)} días")
            print(f"   Datos de prueba: {len(test_data)} días")
            
            # Ajustar modelo de Suavizado Exponencial
            print("\n🔄 Ajustando modelo de Suavizado Exponencial...")
            model = ExponentialSmoothing(train_data, seasonal=None, trend='add')
            model_fit = model.fit()
            
            # Hacer predicciones
            predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
            
            # Calcular métricas
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mse)
            
            print(f"\n📊 Métricas de Suavizado Exponencial:")
            print(f"   MSE: {mse:.2f}")
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            
            # Mostrar predicciones vs realidad
            print(f"\n🔮 Predicciones vs Realidad:")
            for i, (date, actual, predicted) in enumerate(zip(test_data.index, test_data.values, predictions)):
                print(f"   {date.strftime('%Y-%m-%d')}: Real={actual}, Predicho={predicted:.1f}")
                if i >= 4:  # Mostrar solo los primeros 5
                    break
            
            # Análisis de tendencia
            print(f"\n📊 Análisis de Tendencia:")
            trend_slope = model_fit.params.get('trend', 0)
            print(f"   Pendiente de tendencia: {trend_slope:.4f}")
            if trend_slope > 0:
                print("   📈 Tendencia creciente")
            elif trend_slope < 0:
                print("   📉 Tendencia decreciente")
            else:
                print("   ➡️ Tendencia estable")
            
            # Predicción futura
            print(f"\n🔮 Predicción para los próximos 7 días:")
            future_predictions = model_fit.predict(start=len(daily_counts), end=len(daily_counts) + 6)
            last_date = daily_counts.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
            
            for date, pred in zip(future_dates, future_predictions):
                print(f"   {date.strftime('%Y-%m-%d')}: {pred:.1f} artículos")
                
        except Exception as e:
            print(f"❌ Error en análisis de Suavizado Exponencial: {e}")

    def analyze_category_trends_exponential_smoothing(self):
        """Análisis de tendencias por categoría con Suavizado Exponencial"""
        print("📊 ANÁLISIS DE TENDENCIAS POR CATEGORÍA CON SUAVIZADO EXPONENCIAL")
        print("=" * 80)
        
        if not EXPONENTIAL_SMOOTHING_AVAILABLE:
            print("❌ Suavizado Exponencial no está disponible. Instala: pip install statsmodels")
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
                        # Ajustar modelo de Suavizado Exponencial
                        model = ExponentialSmoothing(train_data, seasonal=None, trend='add')
                        model_fit = model.fit()
                        predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
                        
                        # Calcular métricas
                        mse = mean_squared_error(test_data, predictions)
                        print(f"   Modelo: Suavizado Exponencial (tendencia aditiva)")
                        print(f"   RMSE: {np.sqrt(mse):.2f}")
                        
                        # Mostrar tendencia
                        if len(test_data) > 0:
                            actual_trend = test_data.mean()
                            predicted_trend = predictions.mean()
                            print(f"   Tendencia real: {actual_trend:.2f}")
                            print(f"   Tendencia predicha: {predicted_trend:.2f}")
                            
                            # Análisis de tendencia
                            trend_slope = model_fit.params.get('trend', 0)
                            if trend_slope > 0:
                                print(f"   📈 Tendencia creciente ({trend_slope:.4f})")
                            elif trend_slope < 0:
                                print(f"   📉 Tendencia decreciente ({trend_slope:.4f})")
                            else:
                                print(f"   ➡️ Tendencia estable")
                    else:
                        print("   ⚠️ Datos insuficientes para análisis")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")

    def analyze_sentiment_trends_exponential_smoothing(self):
        """Análisis de tendencias de sentimientos con Suavizado Exponencial"""
        print("😊 ANÁLISIS DE TENDENCIAS DE SENTIMIENTOS CON SUAVIZADO EXPONENCIAL")
        print("=" * 80)
        
        if not EXPONENTIAL_SMOOTHING_AVAILABLE:
            print("❌ Suavizado Exponencial no está disponible. Instala: pip install statsmodels")
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
            print("❌ Insuficientes datos temporales para Suavizado Exponencial")
            return
        
        try:
            # Dividir en train/test
            split_point = int(len(daily_sentiment) * 0.8)
            train_data = daily_sentiment[:split_point]
            test_data = daily_sentiment[split_point:]
            
            # Ajustar modelo de Suavizado Exponencial
            print("\n🔄 Ajustando modelo de Suavizado Exponencial para sentimientos...")
            model = ExponentialSmoothing(train_data, seasonal=None, trend='add')
            model_fit = model.fit()
            
            # Hacer predicciones
            predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
            
            # Calcular métricas
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mse)
            
            print(f"\n📊 Métricas de Suavizado Exponencial para sentimientos:")
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
            
            # Análisis de tendencia de sentimientos
            trend_slope = model_fit.params.get('trend', 0)
            print(f"\n📊 Análisis de Tendencia de Sentimientos:")
            print(f"   Pendiente de tendencia: {trend_slope:.4f}")
            if trend_slope > 0:
                print("   😊 Tendencia hacia sentimientos más positivos")
            elif trend_slope < 0:
                print("   😔 Tendencia hacia sentimientos más negativos")
            else:
                print("   😐 Tendencia estable en sentimientos")
            
            # Predicción futura
            print(f"\n🔮 Predicción de sentimientos para los próximos 7 días:")
            future_predictions = model_fit.predict(start=len(daily_sentiment), end=len(daily_sentiment) + 6)
            last_date = daily_sentiment.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
            
            for date, pred in zip(future_dates, future_predictions):
                sentiment_label = "positivo" if pred > 0.1 else "negativo" if pred < -0.1 else "neutral"
                print(f"   {date.strftime('%Y-%m-%d')}: {pred:.3f} ({sentiment_label})")
                
        except Exception as e:
            print(f"❌ Error en análisis de Suavizado Exponencial de sentimientos: {e}")

    def analyze_engagement_trends_exponential_smoothing(self):
        """Análisis de tendencias de engagement con Suavizado Exponencial"""
        print("📈 ANÁLISIS DE TENDENCIAS DE ENGAGEMENT CON SUAVIZADO EXPONENCIAL")
        print("=" * 80)
        
        if not EXPONENTIAL_SMOOTHING_AVAILABLE:
            print("❌ Suavizado Exponencial no está disponible. Instala: pip install statsmodels")
            return
        
        # Preparar datos temporales de engagement
        self.df['scrapedAt'] = pd.to_datetime(self.df['scrapedAt'])
        self.df['date'] = self.df['scrapedAt'].dt.date
        
        # Calcular engagement promedio por día
        daily_engagement = self.df.groupby('date')['engagement_score'].mean()
        daily_engagement.index = pd.to_datetime(daily_engagement.index)
        
        print(f"📊 Datos de engagement:")
        print(f"   Período: {daily_engagement.index.min()} a {daily_engagement.index.max()}")
        print(f"   Días con datos: {len(daily_engagement)}")
        print(f"   Engagement promedio: {daily_engagement.mean():.2f}")
        
        if len(daily_engagement) < 10:
            print("❌ Insuficientes datos temporales para Suavizado Exponencial")
            return
        
        try:
            # Dividir en train/test
            split_point = int(len(daily_engagement) * 0.8)
            train_data = daily_engagement[:split_point]
            test_data = daily_engagement[split_point:]
            
            # Ajustar modelo de Suavizado Exponencial
            print("\n🔄 Ajustando modelo de Suavizado Exponencial para engagement...")
            model = ExponentialSmoothing(train_data, seasonal=None, trend='add')
            model_fit = model.fit()
            
            # Hacer predicciones
            predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
            
            # Calcular métricas
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mse)
            
            print(f"\n📊 Métricas de Suavizado Exponencial para engagement:")
            print(f"   MSE: {mse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            
            # Mostrar predicciones vs realidad
            print(f"\n🔮 Predicciones de engagement vs Realidad:")
            for i, (date, actual, predicted) in enumerate(zip(test_data.index, test_data.values, predictions)):
                engagement_label = "alto" if actual >= 3 else "medio" if actual >= 2 else "bajo"
                pred_label = "alto" if predicted >= 3 else "medio" if predicted >= 2 else "bajo"
                print(f"   {date.strftime('%Y-%m-%d')}: Real={actual:.2f} ({engagement_label}), Predicho={predicted:.2f} ({pred_label})")
                if i >= 4:  # Mostrar solo los primeros 5
                    break
            
            # Análisis de tendencia de engagement
            trend_slope = model_fit.params.get('trend', 0)
            print(f"\n📊 Análisis de Tendencia de Engagement:")
            print(f"   Pendiente de tendencia: {trend_slope:.4f}")
            if trend_slope > 0:
                print("   📈 Tendencia hacia mayor engagement")
            elif trend_slope < 0:
                print("   📉 Tendencia hacia menor engagement")
            else:
                print("   ➡️ Tendencia estable en engagement")
            
            # Predicción futura
            print(f"\n🔮 Predicción de engagement para los próximos 7 días:")
            future_predictions = model_fit.predict(start=len(daily_engagement), end=len(daily_engagement) + 6)
            last_date = daily_engagement.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
            
            for date, pred in zip(future_dates, future_predictions):
                engagement_label = "alto" if pred >= 3 else "medio" if pred >= 2 else "bajo"
                print(f"   {date.strftime('%Y-%m-%d')}: {pred:.2f} ({engagement_label})")
                
        except Exception as e:
            print(f"❌ Error en análisis de Suavizado Exponencial de engagement: {e}")

def main():
    parser = argparse.ArgumentParser(description='Análisis específico con Suavizado Exponencial')
    parser.add_argument('--type', required=True, 
                       choices=['temporal', 'category', 'sentiment', 'engagement'],
                       help='Tipo de análisis a ejecutar')
    
    args = parser.parse_args()
    
    analyzer = ExponentialSmoothingAnalyzer()
    
    if not analyzer.load_data():
        return
    
    analysis_type = args.type
    
    if analysis_type == 'temporal':
        analyzer.analyze_temporal_trends_exponential_smoothing()
    elif analysis_type == 'category':
        analyzer.analyze_category_trends_exponential_smoothing()
    elif analysis_type == 'sentiment':
        analyzer.analyze_sentiment_trends_exponential_smoothing()
    elif analysis_type == 'engagement':
        analyzer.analyze_engagement_trends_exponential_smoothing()
    
    print(f"\n✅ Análisis Suavizado Exponencial {analysis_type} completado exitosamente!")

if __name__ == "__main__":
    main()
