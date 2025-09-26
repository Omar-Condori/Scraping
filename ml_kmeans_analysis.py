#!/usr/bin/env python3
"""
Script específico para análisis con K-Means
Implementa K-Means para las 6 funcionalidades del sistema
"""

import sys
import argparse
import psycopg2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Descargar recursos de NLTK
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class KMeansAnalyzer:
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
        if pd.isna(text):
            return ""
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_features(self):
        self.df['title_length'] = self.df['title'].str.len()
        self.df['content_length'] = self.df['content'].str.len()
        self.df['has_image'] = ~self.df['imageUrl'].isna()
        self.df['has_description'] = ~self.df['description'].isna()
        self.df['has_author'] = ~self.df['author'].isna()
        
        self.df['title_clean'] = self.df['title'].apply(self.preprocess_text)
        self.df['content_clean'] = self.df['content'].apply(self.preprocess_text)
        
        self.df['sentiment_score'] = self.df['content_clean'].apply(
            lambda x: self.sentiment_analyzer.polarity_scores(x)['compound']
        )
        
        self.df['sentiment'] = self.df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
        )

    def analyze_category_classification_kmeans(self):
        """Análisis específico de clasificación de categorías con K-Means"""
        print("🎯 ANÁLISIS DE CLASIFICACIÓN DE CATEGORÍAS CON K-MEANS")
        print("=" * 70)
        
        df_categorized = self.df[self.df['category'].notna() & (self.df['category'] != 'Noticias')].copy()
        
        if len(df_categorized) < 50:
            print("❌ Insuficientes datos categorizados")
            return
        
        X_text = df_categorized['title_clean'] + ' ' + df_categorized['content_clean']
        y = df_categorized['category']
        
        X_vectorized = self.vectorizer.fit_transform(X_text)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Probar diferentes números de clusters
        n_categories = df_categorized['category'].nunique()
        cluster_range = range(2, min(n_categories + 3, 10))
        
        best_kmeans = None
        best_accuracy = 0
        best_k = n_categories
        
        print("🔍 Probando diferentes números de clusters:")
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_train.toarray())
            cluster_labels = kmeans.predict(X_test.toarray())
            
            # Mapear clusters a categorías
            cluster_to_category = {}
            for cluster_id in range(k):
                cluster_mask = kmeans.labels_ == cluster_id
                if np.any(cluster_mask):
                    most_common_category = Counter(y_train[cluster_mask]).most_common(1)[0][0]
                    cluster_to_category[cluster_id] = most_common_category
            
            # Convertir clusters a categorías
            y_pred = [cluster_to_category.get(cluster_id, y_test.iloc[0]) for cluster_id in cluster_labels]
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calcular silhouette score
            silhouette_avg = silhouette_score(X_test.toarray(), cluster_labels)
            
            print(f"   k={k}: Precisión = {accuracy:.4f}, Silhouette = {silhouette_avg:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_kmeans = kmeans
                best_k = k
        
        print(f"\n📊 Mejor K-Means (k={best_k}) - Precisión: {best_accuracy:.4f}")
        print(f"📈 Categorías: {df_categorized['category'].nunique()}")
        print(f"📊 Artículos: {len(df_categorized)}")
        
        # Mostrar predicciones para artículos sin categoría
        uncategorized = self.df[self.df['category'].isna() | (self.df['category'] == 'Noticias')].copy()
        if len(uncategorized) > 0:
            print(f"\n🔮 Predicciones K-Means para {len(uncategorized)} artículos sin categoría:")
            X_uncat = self.vectorizer.transform(uncategorized['title_clean'] + ' ' + uncategorized['content_clean'])
            cluster_labels = best_kmeans.predict(X_uncat.toarray())
            
            # Mapear clusters a categorías usando datos de entrenamiento
            cluster_to_category = {}
            for cluster_id in range(best_k):
                cluster_mask = best_kmeans.labels_ == cluster_id
                if np.any(cluster_mask):
                    most_common_category = Counter(y_train[cluster_mask]).most_common(1)[0][0]
                    cluster_to_category[cluster_id] = most_common_category
            
            predictions = [cluster_to_category.get(cluster_id, 'Política') for cluster_id in cluster_labels]
            
            for i, (idx, row) in enumerate(uncategorized.head(5).iterrows()):
                print(f"\n{i+1}. {row['title'][:60]}...")
                print(f"   Predicción K-Means: {predictions[i]}")
                print(f"   Cluster: {cluster_labels[i]}")

    def analyze_quality_detection_kmeans(self):
        """Análisis específico de detección de calidad con K-Means"""
        print("🔍 ANÁLISIS DE DETECCIÓN DE CALIDAD CON K-MEANS")
        print("=" * 70)
        
        self.df['quality_score'] = 0
        self.df.loc[self.df['content_length'] > 500, 'quality_score'] += 1
        self.df.loc[self.df['has_image'], 'quality_score'] += 1
        self.df.loc[self.df['has_description'], 'quality_score'] += 1
        self.df.loc[self.df['has_author'], 'quality_score'] += 1
        self.df.loc[self.df['title_length'] > 20, 'quality_score'] += 1
        
        self.df['quality'] = self.df['quality_score'].apply(
            lambda x: 'alta' if x >= 4 else 'media' if x >= 2 else 'baja'
        )
        
        features = ['title_length', 'content_length', 'has_image', 'has_description', 'has_author']
        X = self.df[features].fillna(0)
        y = self.df['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar datos para K-Means
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Probar diferentes números de clusters
        n_quality_levels = len(self.df['quality'].unique())
        cluster_range = range(2, min(n_quality_levels + 3, 8))
        
        best_kmeans = None
        best_accuracy = 0
        best_k = n_quality_levels
        
        print("🔍 Probando diferentes números de clusters:")
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_train_scaled)
            cluster_labels = kmeans.predict(X_test_scaled)
            
            # Mapear clusters a calidad
            cluster_to_quality = {}
            for cluster_id in range(k):
                cluster_mask = kmeans.labels_ == cluster_id
                if np.any(cluster_mask):
                    most_common_quality = Counter(y_train[cluster_mask]).most_common(1)[0][0]
                    cluster_to_quality[cluster_id] = most_common_quality
            
            # Convertir clusters a calidad
            y_pred = [cluster_to_quality.get(cluster_id, y_test.iloc[0]) for cluster_id in cluster_labels]
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calcular silhouette score
            silhouette_avg = silhouette_score(X_test_scaled, cluster_labels)
            
            print(f"   k={k}: Precisión = {accuracy:.4f}, Silhouette = {silhouette_avg:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_kmeans = kmeans
                best_k = k
        
        print(f"\n📊 Mejor K-Means (k={best_k}) - Precisión: {best_accuracy:.4f}")
        print(f"📊 Distribución de calidad:")
        print(self.df['quality'].value_counts())
        
        # Mostrar artículos de baja calidad
        low_quality = self.df[self.df['quality'] == 'baja'].head(5)
        print(f"\n⚠️ Ejemplos de artículos de baja calidad:")
        for i, (idx, row) in enumerate(low_quality.iterrows()):
            print(f"\n{i+1}. {row['title'][:60]}...")
            print(f"   Puntuación: {row['quality_score']}/5")
            print(f"   Fuente: {row['source']}")

    def analyze_sentiment_kmeans(self):
        """Análisis específico de sentimientos con K-Means"""
        print("😊 ANÁLISIS DE SENTIMIENTOS CON K-MEANS")
        print("=" * 70)
        
        # Preparar datos para clasificación binaria
        self.df['sentiment_binary'] = self.df['sentiment'].apply(
            lambda x: 1 if x == 'positive' else 0
        )
        
        X_text = self.df['title_clean'] + ' ' + self.df['content_clean']
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        features_numeric = self.df[['title_length', 'content_length', 'sentiment_score']].fillna(0)
        X_combined = np.hstack([X_vectorized.toarray(), features_numeric.values])
        
        y = self.df['sentiment_binary']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar datos para K-Means
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Probar diferentes números de clusters
        cluster_range = range(2, 6)
        
        best_kmeans = None
        best_accuracy = 0
        best_k = 2
        
        print("🔍 Probando diferentes números de clusters:")
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_train_scaled)
            cluster_labels = kmeans.predict(X_test_scaled)
            
            # Mapear clusters a sentimientos
            cluster_to_sentiment = {}
            for cluster_id in range(k):
                cluster_mask = kmeans.labels_ == cluster_id
                if np.any(cluster_mask):
                    most_common_sentiment = Counter(y_train[cluster_mask]).most_common(1)[0][0]
                    cluster_to_sentiment[cluster_id] = most_common_sentiment
            
            # Convertir clusters a sentimientos
            y_pred = [cluster_to_sentiment.get(cluster_id, y_test.iloc[0]) for cluster_id in cluster_labels]
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calcular silhouette score
            silhouette_avg = silhouette_score(X_test_scaled, cluster_labels)
            
            print(f"   k={k}: Precisión = {accuracy:.4f}, Silhouette = {silhouette_avg:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_kmeans = kmeans
                best_k = k
        
        print(f"\n📊 Mejor K-Means (k={best_k}) - Precisión: {best_accuracy:.4f}")
        
        sentiment_counts = self.df['sentiment'].value_counts()
        print(f"\n📊 Distribución de sentimientos:")
        print(sentiment_counts)
        
        # Mostrar artículos más positivos y negativos
        most_positive = self.df.nlargest(3, 'sentiment_score')
        most_negative = self.df.nsmallest(3, 'sentiment_score')
        
        print(f"\n😊 Artículos más positivos:")
        for i, (idx, row) in enumerate(most_positive.iterrows()):
            print(f"{i+1}. {row['title'][:60]}... (Score: {row['sentiment_score']:.3f})")
        
        print(f"\n😞 Artículos más negativos:")
        for i, (idx, row) in enumerate(most_negative.iterrows()):
            print(f"{i+1}. {row['title'][:60]}... (Score: {row['sentiment_score']:.3f})")

    def analyze_engagement_kmeans(self):
        """Análisis específico de engagement con K-Means"""
        print("📈 ANÁLISIS DE ENGAGEMENT CON K-MEANS")
        print("=" * 70)
        
        self.df['engagement_score'] = 0
        self.df.loc[self.df['content_length'] > 300, 'engagement_score'] += 1
        self.df.loc[self.df['has_image'], 'engagement_score'] += 1
        self.df.loc[self.df['sentiment_score'] > 0.1, 'engagement_score'] += 1
        self.df.loc[self.df['title_length'] > 30, 'engagement_score'] += 1
        
        self.df['engagement'] = self.df['engagement_score'].apply(
            lambda x: 'alto' if x >= 3 else 'medio' if x >= 2 else 'bajo'
        )
        
        features = ['title_length', 'content_length', 'has_image', 'sentiment_score', 'has_description']
        X = self.df[features].fillna(0)
        y = self.df['engagement']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar datos para K-Means
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Probar diferentes números de clusters
        n_engagement_levels = len(self.df['engagement'].unique())
        cluster_range = range(2, min(n_engagement_levels + 3, 8))
        
        best_kmeans = None
        best_accuracy = 0
        best_k = n_engagement_levels
        
        print("🔍 Probando diferentes números de clusters:")
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_train_scaled)
            cluster_labels = kmeans.predict(X_test_scaled)
            
            # Mapear clusters a engagement
            cluster_to_engagement = {}
            for cluster_id in range(k):
                cluster_mask = kmeans.labels_ == cluster_id
                if np.any(cluster_mask):
                    most_common_engagement = Counter(y_train[cluster_mask]).most_common(1)[0][0]
                    cluster_to_engagement[cluster_id] = most_common_engagement
            
            # Convertir clusters a engagement
            y_pred = [cluster_to_engagement.get(cluster_id, y_test.iloc[0]) for cluster_id in cluster_labels]
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calcular silhouette score
            silhouette_avg = silhouette_score(X_test_scaled, cluster_labels)
            
            print(f"   k={k}: Precisión = {accuracy:.4f}, Silhouette = {silhouette_avg:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_kmeans = kmeans
                best_k = k
        
        print(f"\n📊 Mejor K-Means (k={best_k}) - Precisión: {best_accuracy:.4f}")
        print(f"📊 Distribución de engagement:")
        print(self.df['engagement'].value_counts())
        
        # Mostrar artículos con mayor engagement potencial
        high_engagement = self.df[self.df['engagement'] == 'alto'].head(5)
        print(f"\n🚀 Artículos con alto engagement potencial:")
        for i, (idx, row) in enumerate(high_engagement.iterrows()):
            print(f"{i+1}. {row['title'][:60]}...")
            print(f"   Puntuación: {row['engagement_score']}/4")

    def analyze_sources_kmeans(self):
        """Análisis específico de fuentes con K-Means"""
        print("📰 ANÁLISIS DE FUENTES CON K-MEANS")
        print("=" * 70)
        
        source_counts = self.df['source'].value_counts()
        reliable_sources = source_counts[source_counts >= 10].index
        
        df_filtered = self.df[self.df['source'].isin(reliable_sources)].copy()
        
        if len(df_filtered) < 50:
            print("❌ Insuficientes datos para clasificar fuentes")
            return
        
        X_text = df_filtered['title_clean'] + ' ' + df_filtered['content_clean']
        y = df_filtered['source']
        
        X_vectorized = self.vectorizer.fit_transform(X_text)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Probar diferentes números de clusters
        n_sources = len(reliable_sources)
        cluster_range = range(2, min(n_sources + 3, 15))
        
        best_kmeans = None
        best_accuracy = 0
        best_k = n_sources
        
        print("🔍 Probando diferentes números de clusters:")
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_train.toarray())
            cluster_labels = kmeans.predict(X_test.toarray())
            
            # Mapear clusters a fuentes
            cluster_to_source = {}
            for cluster_id in range(k):
                cluster_mask = kmeans.labels_ == cluster_id
                if np.any(cluster_mask):
                    most_common_source = Counter(y_train[cluster_mask]).most_common(1)[0][0]
                    cluster_to_source[cluster_id] = most_common_source
            
            # Convertir clusters a fuentes
            y_pred = [cluster_to_source.get(cluster_id, y_test.iloc[0]) for cluster_id in cluster_labels]
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calcular silhouette score
            silhouette_avg = silhouette_score(X_test.toarray(), cluster_labels)
            
            print(f"   k={k}: Precisión = {accuracy:.4f}, Silhouette = {silhouette_avg:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_kmeans = kmeans
                best_k = k
        
        print(f"\n📊 Mejor K-Means (k={best_k}) - Precisión: {best_accuracy:.4f}")
        print(f"📈 Fuentes analizadas: {len(reliable_sources)}")
        print(f"📊 Top fuentes por volumen:")
        print(source_counts.head(10))
        
        # Análisis de calidad por fuente
        print(f"\n📊 Calidad promedio por fuente:")
        quality_by_source = df_filtered.groupby('source').agg({
            'content_length': 'mean',
            'has_image': 'mean',
            'has_description': 'mean',
            'sentiment_score': 'mean'
        }).round(2)
        print(quality_by_source.head(10))

def main():
    parser = argparse.ArgumentParser(description='Análisis específico con K-Means')
    parser.add_argument('--type', required=True, 
                       choices=['category', 'quality', 'sentiment', 'engagement', 'sources'],
                       help='Tipo de análisis a ejecutar')
    
    args = parser.parse_args()
    
    analyzer = KMeansAnalyzer()
    
    if not analyzer.load_data():
        return
    
    analyzer.extract_features()
    
    analysis_type = args.type
    
    if analysis_type == 'category':
        analyzer.analyze_category_classification_kmeans()
    elif analysis_type == 'quality':
        analyzer.analyze_quality_detection_kmeans()
    elif analysis_type == 'sentiment':
        analyzer.analyze_sentiment_kmeans()
    elif analysis_type == 'engagement':
        analyzer.analyze_engagement_kmeans()
    elif analysis_type == 'sources':
        analyzer.analyze_sources_kmeans()
    
    print(f"\n✅ Análisis K-Means {analysis_type} completado exitosamente!")

if __name__ == "__main__":
    main()
