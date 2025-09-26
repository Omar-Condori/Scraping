#!/usr/bin/env python3
"""
Script específico para análisis con XGBoost
Implementa XGBoost para análisis de noticias
"""

import sys
import argparse
import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, Binarizer, StandardScaler
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Para XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible. Instala: pip install xgboost")

# Descargar recursos de NLTK
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class XGBoostAnalyzer:
    def __init__(self, db_url="postgresql://omar@localhost:5432/scraping_db"):
        self.db_url = db_url
        self.conn = None
        self.df = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
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
    
    def preprocess_text(self, text):
        """Preprocesar texto para análisis"""
        if pd.isna(text):
            return ""
        
        # Limpiar texto
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_features(self):
        """Extraer características de los artículos"""
        print("🔍 Extrayendo características de los artículos...")
        
        # Características básicas
        self.df['title_length'] = self.df['title'].str.len()
        self.df['content_length'] = self.df['content'].str.len()
        self.df['has_image'] = ~self.df['imageUrl'].isna()
        self.df['has_description'] = ~self.df['description'].isna()
        self.df['has_author'] = ~self.df['author'].isna()
        
        # Preprocesar texto
        self.df['title_clean'] = self.df['title'].apply(self.preprocess_text)
        self.df['content_clean'] = self.df['content'].apply(self.preprocess_text)
        
        # Análisis de sentimientos
        self.df['sentiment_score'] = self.df['content_clean'].apply(
            lambda x: self.sentiment_analyzer.polarity_scores(x)['compound']
        )
        
        # Clasificar sentimientos
        self.df['sentiment'] = self.df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
        )
        
        # Calcular engagement score
        self.df['engagement_score'] = (
            self.df['has_image'].astype(int) * 2 +
            self.df['has_description'].astype(int) * 1 +
            self.df['has_author'].astype(int) * 1 +
            (self.df['content_length'] > 500).astype(int) * 1
        )
        
        print("✅ Características extraídas exitosamente")
    
    def mide_error(self, model_name, y_pred, y_test):
        """Medir error del modelo"""
        if len(np.unique(y_test)) == 2:  # Clasificación binaria
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_pred)
                print(f"📊 {model_name} - Precisión: {accuracy:.4f}, AUC: {auc:.4f}")
            except:
                print(f"📊 {model_name} - Precisión: {accuracy:.4f}")
        else:  # Clasificación multiclase
            accuracy = accuracy_score(y_test, y_pred)
            print(f"📊 {model_name} - Precisión: {accuracy:.4f}")
        return accuracy

    def analyze_category_classification_xgboost(self):
        """Análisis de clasificación de categorías con XGBoost"""
        print("🎯 CLASIFICACIÓN AUTOMÁTICA DE CATEGORÍAS CON XGBOOST")
        print("=" * 80)
        
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost no está disponible. Instala: pip install xgboost")
            return
        
        # Filtrar artículos con categorías válidas
        df_categorized = self.df[self.df['category'].notna() & (self.df['category'] != 'Noticias')].copy()
        
        if len(df_categorized) < 50:
            print("❌ Insuficientes datos categorizados para entrenar")
            return
        
        # Preparar datos
        X_text = df_categorized['title_clean'] + ' ' + df_categorized['content_clean']
        y = df_categorized['category']
        
        # Codificar etiquetas categóricas para XGBoost
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Vectorizar texto
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Entrenar XGBoost
        print("🔄 Entrenando XGBoost...")
        xgb_classifier = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_classifier.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred_encoded = xgb_classifier.predict(X_test)
        y_pred_proba = xgb_classifier.predict_proba(X_test)
        
        # Decodificar predicciones
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        
        # Calcular métricas
        accuracy = self.mide_error('XGBoost', y_pred, y_test_decoded)
        
        print(f"\n📊 Datos del entrenamiento:")
        print(f"   Categorías disponibles: {df_categorized['category'].nunique()}")
        print(f"   Artículos categorizados: {len(df_categorized)}")
        print(f"   Artículos de entrenamiento: {X_train.shape[0]}")
        print(f"   Artículos de prueba: {X_test.shape[0]}")
        print(f"   Estimadores XGBoost: {xgb_classifier.n_estimators}")
        
        # Mostrar importancia de características
        print(f"\n📊 Importancia de características (top 10):")
        feature_names = self.vectorizer.get_feature_names_out()
        importances = xgb_classifier.feature_importances_
        top_features = np.argsort(importances)[-10:][::-1]
        
        for i, idx in enumerate(top_features):
            print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Mostrar reporte de clasificación
        print(f"\n📋 Reporte de Clasificación:")
        print(classification_report(y_test_decoded, y_pred))
        
        return xgb_classifier, accuracy

    def analyze_quality_detection_xgboost(self):
        """Análisis de detección de calidad con XGBoost"""
        print("🔍 DETECCIÓN DE CALIDAD DE CONTENIDO CON XGBOOST")
        print("=" * 80)
        
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost no está disponible. Instala: pip install xgboost")
            return
        
        # Definir calidad basada en características
        self.df['quality_score'] = (
            self.df['has_image'].astype(int) * 2 +
            self.df['has_description'].astype(int) * 2 +
            self.df['has_author'].astype(int) * 1 +
            (self.df['content_length'] > 200).astype(int) * 2 +
            (self.df['title_length'] > 20).astype(int) * 1
        )
        
        # Clasificar como alta/baja calidad
        self.df['quality'] = (self.df['quality_score'] >= 5).astype(int)
        
        # Preparar datos
        X_text = self.df['title_clean'] + ' ' + self.df['content_clean']
        y = self.df['quality']
        
        # Vectorizar texto
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar XGBoost
        print("🔄 Entrenando XGBoost para detección de calidad...")
        xgb_classifier = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_classifier.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = xgb_classifier.predict(X_test)
        y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        accuracy = self.mide_error('XGBoost', y_pred, y_test)
        
        print(f"\n📊 Datos del entrenamiento:")
        print(f"   Artículos de alta calidad: {(y == 1).sum()}")
        print(f"   Artículos de baja calidad: {(y == 0).sum()}")
        print(f"   Artículos de entrenamiento: {X_train.shape[0]}")
        print(f"   Artículos de prueba: {X_test.shape[0]}")
        
        return xgb_classifier, accuracy

    def analyze_sentiment_classification_xgboost(self):
        """Análisis de clasificación de sentimientos con XGBoost"""
        print("😊 CLASIFICACIÓN DE SENTIMIENTOS CON XGBOOST")
        print("=" * 80)
        
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost no está disponible. Instala: pip install xgboost")
            return
        
        # Preparar datos
        X_text = self.df['title_clean'] + ' ' + self.df['content_clean']
        y = self.df['sentiment']
        
        # Vectorizar texto
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar XGBoost
        print("🔄 Entrenando XGBoost para análisis de sentimientos...")
        xgb_classifier = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_classifier.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = xgb_classifier.predict(X_test)
        
        # Calcular métricas
        accuracy = self.mide_error('XGBoost', y_pred, y_test)
        
        print(f"\n📊 Datos del entrenamiento:")
        print(f"   Sentimientos positivos: {(y == 'positive').sum()}")
        print(f"   Sentimientos negativos: {(y == 'negative').sum()}")
        print(f"   Sentimientos neutrales: {(y == 'neutral').sum()}")
        print(f"   Artículos de entrenamiento: {X_train.shape[0]}")
        print(f"   Artículos de prueba: {X_test.shape[0]}")
        
        return xgb_classifier, accuracy

    def analyze_engagement_prediction_xgboost(self):
        """Análisis de predicción de engagement con XGBoost"""
        print("📈 PREDICCIÓN DE ENGAGEMENT CON XGBOOST")
        print("=" * 80)
        
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost no está disponible. Instala: pip install xgboost")
            return
        
        # Clasificar engagement como alto/medio/bajo
        self.df['engagement_level'] = pd.cut(
            self.df['engagement_score'], 
            bins=[0, 2, 4, 6], 
            labels=['bajo', 'medio', 'alto']
        )
        
        # Preparar datos
        X_text = self.df['title_clean'] + ' ' + self.df['content_clean']
        y = self.df['engagement_level']
        
        # Vectorizar texto
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar XGBoost
        print("🔄 Entrenando XGBoost para predicción de engagement...")
        xgb_classifier = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_classifier.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = xgb_classifier.predict(X_test)
        
        # Calcular métricas
        accuracy = self.mide_error('XGBoost', y_pred, y_test)
        
        print(f"\n📊 Datos del entrenamiento:")
        print(f"   Engagement alto: {(y == 'alto').sum()}")
        print(f"   Engagement medio: {(y == 'medio').sum()}")
        print(f"   Engagement bajo: {(y == 'bajo').sum()}")
        print(f"   Artículos de entrenamiento: {X_train.shape[0]}")
        print(f"   Artículos de prueba: {X_test.shape[0]}")
        
        return xgb_classifier, accuracy

    def analyze_source_classification_xgboost(self):
        """Análisis de clasificación de fuentes con XGBoost"""
        print("📰 CLASIFICACIÓN DE FUENTES CON XGBOOST")
        print("=" * 80)
        
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost no está disponible. Instala: pip install xgboost")
            return
        
        # Filtrar fuentes con suficientes artículos
        source_counts = self.df['source'].value_counts()
        valid_sources = source_counts[source_counts >= 5].index
        df_filtered = self.df[self.df['source'].isin(valid_sources)].copy()
        
        if len(df_filtered) < 50:
            print("❌ Insuficientes datos de fuentes para entrenar")
            return
        
        # Preparar datos
        X_text = df_filtered['title_clean'] + ' ' + df_filtered['content_clean']
        y = df_filtered['source']
        
        # Vectorizar texto
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar XGBoost
        print("🔄 Entrenando XGBoost para clasificación de fuentes...")
        xgb_classifier = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_classifier.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = xgb_classifier.predict(X_test)
        
        # Calcular métricas
        accuracy = self.mide_error('XGBoost', y_pred, y_test)
        
        print(f"\n📊 Datos del entrenamiento:")
        print(f"   Fuentes disponibles: {df_filtered['source'].nunique()}")
        print(f"   Artículos filtrados: {len(df_filtered)}")
        print(f"   Artículos de entrenamiento: {X_train.shape[0]}")
        print(f"   Artículos de prueba: {X_test.shape[0]}")
        
        return xgb_classifier, accuracy

def main():
    parser = argparse.ArgumentParser(description='Análisis específico con XGBoost')
    parser.add_argument('--type', required=True, 
                       choices=['category', 'quality', 'sentiment', 'engagement', 'source'],
                       help='Tipo de análisis a ejecutar')
    
    args = parser.parse_args()
    
    analyzer = XGBoostAnalyzer()
    
    if not analyzer.load_data():
        return
    
    analyzer.extract_features()
    
    analysis_type = args.type
    
    if analysis_type == 'category':
        analyzer.analyze_category_classification_xgboost()
    elif analysis_type == 'quality':
        analyzer.analyze_quality_detection_xgboost()
    elif analysis_type == 'sentiment':
        analyzer.analyze_sentiment_classification_xgboost()
    elif analysis_type == 'engagement':
        analyzer.analyze_engagement_prediction_xgboost()
    elif analysis_type == 'source':
        analyzer.analyze_source_classification_xgboost()
    
    print(f"\n✅ Análisis XGBoost {analysis_type} completado exitosamente!")

if __name__ == "__main__":
    main()
