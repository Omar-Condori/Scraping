#!/usr/bin/env python3
"""
Script específico para análisis con Árbol de Decisión
Implementa Árbol de Decisión para las 6 funcionalidades del sistema
"""

import sys
import argparse
import psycopg2
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import Binarizer
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

class DecisionTreeAnalyzer:
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

    def analyze_category_classification_tree(self):
        """Análisis específico de clasificación de categorías con Árbol de Decisión"""
        print("🎯 ANÁLISIS DE CLASIFICACIÓN DE CATEGORÍAS CON ÁRBOL DE DECISIÓN")
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
        
        # Probar diferentes configuraciones de Árbol de Decisión
        tree_configs = {
            'Árbol Estándar': DecisionTreeClassifier(random_state=42),
            'Árbol Limitado (max_depth=5)': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Árbol Profundo (max_depth=15)': DecisionTreeClassifier(random_state=42, max_depth=15),
            'Árbol con Mínimo Split': DecisionTreeClassifier(random_state=42, min_samples_split=10),
            'Árbol con Mínimo Leaf': DecisionTreeClassifier(random_state=42, min_samples_leaf=5)
        }
        
        best_tree = None
        best_accuracy = 0
        best_config = None
        
        print("🔍 Probando diferentes configuraciones de Árbol de Decisión:")
        for config_name, tree_model in tree_configs.items():
            try:
                tree_model.fit(X_train, y_train)
                y_pred = tree_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"   {config_name}: Precisión = {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_tree = tree_model
                    best_config = config_name
            except Exception as e:
                print(f"   {config_name}: Error - {e}")
        
        print(f"\n📊 Mejor Árbol de Decisión ({best_config}) - Precisión: {best_accuracy:.4f}")
        print(f"📈 Categorías: {df_categorized['category'].nunique()}")
        print(f"📊 Artículos: {len(df_categorized)}")
        
        # Mostrar importancia de características
        if hasattr(best_tree, 'feature_importances_'):
            print(f"\n🌳 Top 10 características más importantes:")
            feature_names = self.vectorizer.get_feature_names_out()
            importances = best_tree.feature_importances_
            top_features = np.argsort(importances)[-10:][::-1]
            
            for i, idx in enumerate(top_features):
                print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Mostrar predicciones para artículos sin categoría
        uncategorized = self.df[self.df['category'].isna() | (self.df['category'] == 'Noticias')].copy()
        if len(uncategorized) > 0:
            print(f"\n🔮 Predicciones Árbol de Decisión para {len(uncategorized)} artículos sin categoría:")
            X_uncat = self.vectorizer.transform(uncategorized['title_clean'] + ' ' + uncategorized['content_clean'])
            predictions = best_tree.predict(X_uncat)
            probabilities = best_tree.predict_proba(X_uncat)
            
            for i, (idx, row) in enumerate(uncategorized.head(5).iterrows()):
                print(f"\n{i+1}. {row['title'][:60]}...")
                print(f"   Predicción Árbol de Decisión: {predictions[i]}")
                print(f"   Confianza: {max(probabilities[i]):.3f}")

    def analyze_quality_detection_tree(self):
        """Análisis específico de detección de calidad con Árbol de Decisión"""
        print("🔍 ANÁLISIS DE DETECCIÓN DE CALIDAD CON ÁRBOL DE DECISIÓN")
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
        
        # Probar diferentes configuraciones de Árbol de Decisión
        tree_configs = {
            'Árbol Estándar': DecisionTreeClassifier(random_state=42),
            'Árbol Limitado (max_depth=3)': DecisionTreeClassifier(random_state=42, max_depth=3),
            'Árbol Profundo (max_depth=10)': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Árbol con Mínimo Split': DecisionTreeClassifier(random_state=42, min_samples_split=20),
            'Árbol con Mínimo Leaf': DecisionTreeClassifier(random_state=42, min_samples_leaf=10)
        }
        
        best_tree = None
        best_accuracy = 0
        best_config = None
        
        print("🔍 Probando diferentes configuraciones de Árbol de Decisión:")
        for config_name, tree_model in tree_configs.items():
            try:
                tree_model.fit(X_train, y_train)
                y_pred = tree_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"   {config_name}: Precisión = {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_tree = tree_model
                    best_config = config_name
            except Exception as e:
                print(f"   {config_name}: Error - {e}")
        
        print(f"\n📊 Mejor Árbol de Decisión ({best_config}) - Precisión: {best_accuracy:.4f}")
        print(f"📊 Distribución de calidad:")
        print(self.df['quality'].value_counts())
        
        # Mostrar importancia de características
        if hasattr(best_tree, 'feature_importances_'):
            print(f"\n🌳 Importancia de características:")
            feature_names = features
            importances = best_tree.feature_importances_
            
            for i, (feature, importance) in enumerate(zip(feature_names, importances)):
                print(f"   {i+1}. {feature}: {importance:.4f}")
        
        # Mostrar artículos de baja calidad
        low_quality = self.df[self.df['quality'] == 'baja'].head(5)
        print(f"\n⚠️ Ejemplos de artículos de baja calidad:")
        for i, (idx, row) in enumerate(low_quality.iterrows()):
            print(f"\n{i+1}. {row['title'][:60]}...")
            print(f"   Puntuación: {row['quality_score']}/5")
            print(f"   Fuente: {row['source']}")

    def analyze_sentiment_tree(self):
        """Análisis específico de sentimientos con Árbol de Decisión"""
        print("😊 ANÁLISIS DE SENTIMIENTOS CON ÁRBOL DE DECISIÓN")
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
        
        # Probar diferentes configuraciones de Árbol de Decisión
        tree_configs = {
            'Árbol Estándar': DecisionTreeClassifier(random_state=42),
            'Árbol Limitado (max_depth=5)': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Árbol Profundo (max_depth=15)': DecisionTreeClassifier(random_state=42, max_depth=15),
            'Árbol con Mínimo Split': DecisionTreeClassifier(random_state=42, min_samples_split=50),
            'Árbol con Mínimo Leaf': DecisionTreeClassifier(random_state=42, min_samples_leaf=25)
        }
        
        best_tree = None
        best_accuracy = 0
        best_config = None
        
        print("🔍 Probando diferentes configuraciones de Árbol de Decisión:")
        for config_name, tree_model in tree_configs.items():
            try:
                tree_model.fit(X_train, y_train)
                y_pred = tree_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"   {config_name}: Precisión = {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_tree = tree_model
                    best_config = config_name
            except Exception as e:
                print(f"   {config_name}: Error - {e}")
        
        print(f"\n📊 Mejor Árbol de Decisión ({best_config}) - Precisión: {best_accuracy:.4f}")
        
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

    def analyze_engagement_tree(self):
        """Análisis específico de engagement con Árbol de Decisión"""
        print("📈 ANÁLISIS DE ENGAGEMENT CON ÁRBOL DE DECISIÓN")
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
        
        # Probar diferentes configuraciones de Árbol de Decisión
        tree_configs = {
            'Árbol Estándar': DecisionTreeClassifier(random_state=42),
            'Árbol Limitado (max_depth=4)': DecisionTreeClassifier(random_state=42, max_depth=4),
            'Árbol Profundo (max_depth=12)': DecisionTreeClassifier(random_state=42, max_depth=12),
            'Árbol con Mínimo Split': DecisionTreeClassifier(random_state=42, min_samples_split=30),
            'Árbol con Mínimo Leaf': DecisionTreeClassifier(random_state=42, min_samples_leaf=15)
        }
        
        best_tree = None
        best_accuracy = 0
        best_config = None
        
        print("🔍 Probando diferentes configuraciones de Árbol de Decisión:")
        for config_name, tree_model in tree_configs.items():
            try:
                tree_model.fit(X_train, y_train)
                y_pred = tree_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"   {config_name}: Precisión = {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_tree = tree_model
                    best_config = config_name
            except Exception as e:
                print(f"   {config_name}: Error - {e}")
        
        print(f"\n📊 Mejor Árbol de Decisión ({best_config}) - Precisión: {best_accuracy:.4f}")
        print(f"📊 Distribución de engagement:")
        print(self.df['engagement'].value_counts())
        
        # Mostrar importancia de características
        if hasattr(best_tree, 'feature_importances_'):
            print(f"\n🌳 Importancia de características para engagement:")
            feature_names = features
            importances = best_tree.feature_importances_
            
            for i, (feature, importance) in enumerate(zip(feature_names, importances)):
                print(f"   {i+1}. {feature}: {importance:.4f}")
        
        # Mostrar artículos con mayor engagement potencial
        high_engagement = self.df[self.df['engagement'] == 'alto'].head(5)
        print(f"\n🚀 Artículos con alto engagement potencial:")
        for i, (idx, row) in enumerate(high_engagement.iterrows()):
            print(f"{i+1}. {row['title'][:60]}...")
            print(f"   Puntuación: {row['engagement_score']}/4")

    def analyze_sources_tree(self):
        """Análisis específico de fuentes con Árbol de Decisión"""
        print("📰 ANÁLISIS DE FUENTES CON ÁRBOL DE DECISIÓN")
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
        
        # Probar diferentes configuraciones de Árbol de Decisión
        tree_configs = {
            'Árbol Estándar': DecisionTreeClassifier(random_state=42),
            'Árbol Limitado (max_depth=8)': DecisionTreeClassifier(random_state=42, max_depth=8),
            'Árbol Profundo (max_depth=20)': DecisionTreeClassifier(random_state=42, max_depth=20),
            'Árbol con Mínimo Split': DecisionTreeClassifier(random_state=42, min_samples_split=15),
            'Árbol con Mínimo Leaf': DecisionTreeClassifier(random_state=42, min_samples_leaf=8)
        }
        
        best_tree = None
        best_accuracy = 0
        best_config = None
        
        print("🔍 Probando diferentes configuraciones de Árbol de Decisión:")
        for config_name, tree_model in tree_configs.items():
            try:
                tree_model.fit(X_train, y_train)
                y_pred = tree_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"   {config_name}: Precisión = {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_tree = tree_model
                    best_config = config_name
            except Exception as e:
                print(f"   {config_name}: Error - {e}")
        
        print(f"\n📊 Mejor Árbol de Decisión ({best_config}) - Precisión: {best_accuracy:.4f}")
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
    parser = argparse.ArgumentParser(description='Análisis específico con Árbol de Decisión')
    parser.add_argument('--type', required=True, 
                       choices=['category', 'quality', 'sentiment', 'engagement', 'sources'],
                       help='Tipo de análisis a ejecutar')
    
    args = parser.parse_args()
    
    analyzer = DecisionTreeAnalyzer()
    
    if not analyzer.load_data():
        return
    
    analyzer.extract_features()
    
    analysis_type = args.type
    
    if analysis_type == 'category':
        analyzer.analyze_category_classification_tree()
    elif analysis_type == 'quality':
        analyzer.analyze_quality_detection_tree()
    elif analysis_type == 'sentiment':
        analyzer.analyze_sentiment_tree()
    elif analysis_type == 'engagement':
        analyzer.analyze_engagement_tree()
    elif analysis_type == 'sources':
        analyzer.analyze_sources_tree()
    
    print(f"\n✅ Análisis Árbol de Decisión {analysis_type} completado exitosamente!")

if __name__ == "__main__":
    main()
