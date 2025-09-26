#!/usr/bin/env python3
"""
Sistema de Análisis de Noticias con Regresión Logística
Implementa 6 funcionalidades de ML para el sistema de scraping
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Descargar recursos de NLTK
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class NewsAnalyzer:
    def __init__(self, db_url="postgresql://omar@localhost:5432/scraping_db"):
        """Inicializar el analizador de noticias"""
        self.db_url = db_url
        self.conn = None
        self.df = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def connect_db(self):
        """Conectar a la base de datos PostgreSQL"""
        try:
            self.conn = psycopg2.connect(self.db_url)
            print("✅ Conectado a la base de datos PostgreSQL")
            return True
        except Exception as e:
            print(f"❌ Error conectando a la base de datos: {e}")
            return False
    
    def load_data(self):
        """Cargar datos de noticias desde la base de datos"""
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
            print(f"✅ Cargados {len(self.df)} artículos de la base de datos")
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
        
        print("✅ Características extraídas exitosamente")
    
    def mide_error(self, model_name, y_pred, y_test):
        """Medir error del modelo"""
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📊 {model_name} - Precisión: {accuracy:.4f}")
        return accuracy

class CategoryClassifier(NewsAnalyzer):
    """Clasificador automático de categorías"""
    
    def train_category_classifier(self):
        """Entrenar modelo para clasificar categorías"""
        print("\n🎯 1. CLASIFICACIÓN AUTOMÁTICA DE CATEGORÍAS")
        print("=" * 50)
        
        # Filtrar artículos con categorías válidas
        df_categorized = self.df[self.df['category'].notna() & (self.df['category'] != 'Noticias')].copy()
        
        if len(df_categorized) < 50:
            print("❌ Insuficientes datos categorizados para entrenar")
            return None
        
        # Preparar datos
        X_text = df_categorized['title_clean'] + ' ' + df_categorized['content_clean']
        y = df_categorized['category']
        
        # Vectorizar texto
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar modelo
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        
        # Predecir
        y_pred = logreg.predict(X_test)
        
        # Medir precisión
        accuracy = self.mide_error('Clasificador de Categorías', y_pred, y_test)
        
        print(f"📈 Categorías disponibles: {df_categorized['category'].nunique()}")
        print(f"📊 Artículos categorizados: {len(df_categorized)}")
        
        # Mostrar reporte de clasificación
        print("\n📋 Reporte de Clasificación:")
        print(classification_report(y_test, y_pred))
        
        return logreg, self.vectorizer

class QualityDetector(NewsAnalyzer):
    """Detector de calidad de contenido"""
    
    def train_quality_detector(self):
        """Entrenar modelo para detectar calidad de contenido"""
        print("\n🔍 2. DETECCIÓN DE CALIDAD DE CONTENIDO")
        print("=" * 50)
        
        # Definir criterios de calidad
        self.df['quality_score'] = 0
        
        # Criterios de calidad
        self.df.loc[self.df['content_length'] > 500, 'quality_score'] += 1
        self.df.loc[self.df['has_image'], 'quality_score'] += 1
        self.df.loc[self.df['has_description'], 'quality_score'] += 1
        self.df.loc[self.df['has_author'], 'quality_score'] += 1
        self.df.loc[self.df['title_length'] > 20, 'quality_score'] += 1
        
        # Clasificar calidad (0-2: baja, 3-4: media, 5: alta)
        self.df['quality'] = self.df['quality_score'].apply(
            lambda x: 'alta' if x >= 4 else 'media' if x >= 2 else 'baja'
        )
        
        # Preparar características
        features = ['title_length', 'content_length', 'has_image', 'has_description', 'has_author']
        X = self.df[features].fillna(0)
        y = self.df['quality']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar modelo
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        
        # Predecir
        y_pred = logreg.predict(X_test)
        
        # Medir precisión
        accuracy = self.mide_error('Detector de Calidad', y_pred, y_test)
        
        print(f"📊 Distribución de calidad:")
        print(self.df['quality'].value_counts())
        
        return logreg

class SentimentAnalyzer(NewsAnalyzer):
    """Analizador de sentimientos"""
    
    def analyze_sentiments(self):
        """Analizar sentimientos de las noticias"""
        print("\n😊 3. ANÁLISIS DE SENTIMIENTOS")
        print("=" * 50)
        
        # Usar análisis de sentimientos ya calculado
        sentiment_counts = self.df['sentiment'].value_counts()
        print(f"📊 Distribución de sentimientos:")
        print(sentiment_counts)
        
        # Análisis por categoría
        print("\n📈 Sentimientos por categoría:")
        sentiment_by_category = pd.crosstab(self.df['category'], self.df['sentiment'])
        print(sentiment_by_category)
        
        # Análisis por fuente
        print("\n📰 Sentimientos por fuente:")
        sentiment_by_source = pd.crosstab(self.df['source'], self.df['sentiment'])
        print(sentiment_by_source.head(10))
        
        return self.df['sentiment']

class EngagementPredictor(NewsAnalyzer):
    """Predictor de engagement"""
    
    def train_engagement_predictor(self):
        """Entrenar modelo para predecir engagement"""
        print("\n📈 4. PREDICCIÓN DE ENGAGEMENT")
        print("=" * 50)
        
        # Simular engagement basado en características del artículo
        self.df['engagement_score'] = 0
        
        # Factores que influyen en engagement
        self.df.loc[self.df['content_length'] > 300, 'engagement_score'] += 1
        self.df.loc[self.df['has_image'], 'engagement_score'] += 1
        self.df.loc[self.df['sentiment_score'] > 0.1, 'engagement_score'] += 1
        self.df.loc[self.df['title_length'] > 30, 'engagement_score'] += 1
        
        # Clasificar engagement
        self.df['engagement'] = self.df['engagement_score'].apply(
            lambda x: 'alto' if x >= 3 else 'medio' if x >= 2 else 'bajo'
        )
        
        # Preparar características
        features = ['title_length', 'content_length', 'has_image', 'sentiment_score', 'has_description']
        X = self.df[features].fillna(0)
        y = self.df['engagement']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar modelo
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        
        # Predecir
        y_pred = logreg.predict(X_test)
        
        # Medir precisión
        accuracy = self.mide_error('Predictor de Engagement', y_pred, y_test)
        
        print(f"📊 Distribución de engagement:")
        print(self.df['engagement'].value_counts())
        
        return logreg

class DuplicateDetector(NewsAnalyzer):
    """Detector de duplicados"""
    
    def detect_duplicates(self):
        """Detectar artículos duplicados o similares"""
        print("\n🔍 5. DETECCIÓN DE DUPLICADOS")
        print("=" * 50)
        
        # Vectorizar títulos para detectar similitudes
        title_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        title_vectors = title_vectorizer.fit_transform(self.df['title_clean'])
        
        # Calcular similitudes usando producto punto
        similarities = np.dot(title_vectors, title_vectors.T).toarray()
        
        # Encontrar pares similares (similitud > 0.8)
        similar_pairs = []
        for i in range(len(similarities)):
            for j in range(i+1, len(similarities)):
                if similarities[i][j] > 0.8:
                    similar_pairs.append((i, j, similarities[i][j]))
        
        print(f"🔍 Encontrados {len(similar_pairs)} pares de artículos similares")
        
        # Mostrar algunos ejemplos
        if similar_pairs:
            print("\n📋 Ejemplos de artículos similares:")
            for i, (idx1, idx2, sim) in enumerate(similar_pairs[:5]):
                print(f"\n{i+1}. Similitud: {sim:.3f}")
                print(f"   Título 1: {self.df.iloc[idx1]['title'][:80]}...")
                print(f"   Título 2: {self.df.iloc[idx2]['title'][:80]}...")
        
        return similar_pairs

class SourceClassifier(NewsAnalyzer):
    """Clasificador de fuentes"""
    
    def train_source_classifier(self):
        """Entrenar modelo para clasificar fuentes"""
        print("\n📰 6. CLASIFICACIÓN DE FUENTES")
        print("=" * 50)
        
        # Filtrar fuentes con suficientes artículos
        source_counts = self.df['source'].value_counts()
        reliable_sources = source_counts[source_counts >= 10].index
        
        df_filtered = self.df[self.df['source'].isin(reliable_sources)].copy()
        
        if len(df_filtered) < 50:
            print("❌ Insuficientes datos para clasificar fuentes")
            return None
        
        # Preparar datos
        X_text = df_filtered['title_clean'] + ' ' + df_filtered['content_clean']
        y = df_filtered['source']
        
        # Vectorizar texto
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar modelo
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        
        # Predecir
        y_pred = logreg.predict(X_test)
        
        # Medir precisión
        accuracy = self.mide_error('Clasificador de Fuentes', y_pred, y_test)
        
        print(f"📈 Fuentes analizadas: {len(reliable_sources)}")
        print(f"📊 Artículos por fuente:")
        print(source_counts.head(10))
        
        return logreg, self.vectorizer

def main():
    """Función principal para ejecutar todos los análisis"""
    print("🚀 SISTEMA DE ANÁLISIS DE NOTICIAS CON REGRESIÓN LOGÍSTICA")
    print("=" * 70)
    
    # Inicializar analizador
    analyzer = NewsAnalyzer()
    
    # Cargar datos
    if not analyzer.load_data():
        return
    
    # Extraer características
    analyzer.extract_features()
    
    # Ejecutar todos los análisis
    print(f"\n📊 RESUMEN DE DATOS:")
    print(f"   Total de artículos: {len(analyzer.df)}")
    print(f"   Categorías únicas: {analyzer.df['category'].nunique()}")
    print(f"   Fuentes únicas: {analyzer.df['source'].nunique()}")
    print(f"   Artículos con imágenes: {analyzer.df['has_image'].sum()}")
    print(f"   Artículos con descripción: {analyzer.df['has_description'].sum()}")
    
    # 1. Clasificación de categorías
    category_classifier = CategoryClassifier()
    category_classifier.df = analyzer.df.copy()
    category_classifier.train_category_classifier()
    
    # 2. Detección de calidad
    quality_detector = QualityDetector()
    quality_detector.df = analyzer.df.copy()
    quality_detector.train_quality_detector()
    
    # 3. Análisis de sentimientos
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.df = analyzer.df.copy()
    sentiment_analyzer.analyze_sentiments()
    
    # 4. Predicción de engagement
    engagement_predictor = EngagementPredictor()
    engagement_predictor.df = analyzer.df.copy()
    engagement_predictor.train_engagement_predictor()
    
    # 5. Detección de duplicados
    duplicate_detector = DuplicateDetector()
    duplicate_detector.df = analyzer.df.copy()
    duplicate_detector.detect_duplicates()
    
    # 6. Clasificación de fuentes
    source_classifier = SourceClassifier()
    source_classifier.df = analyzer.df.copy()
    source_classifier.train_source_classifier()
    
    print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
    print("=" * 70)
    print("📋 Funcionalidades implementadas:")
    print("   1. ✅ Clasificación automática de categorías")
    print("   2. ✅ Detección de calidad de contenido")
    print("   3. ✅ Análisis de sentimientos")
    print("   4. ✅ Predicción de engagement")
    print("   5. ✅ Detección de duplicados")
    print("   6. ✅ Clasificación de fuentes")

if __name__ == "__main__":
    main()
