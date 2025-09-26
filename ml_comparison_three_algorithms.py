#!/usr/bin/env python3
"""
Sistema de Análisis de Noticias con Regresión Logística, KNN y Naive Bayes
Implementa comparación entre los tres algoritmos de ML
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, Binarizer
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

class NewsAnalyzerML:
    def __init__(self, db_url="postgresql://omar@localhost:5432/scraping_db"):
        """Inicializar el analizador de noticias con ML"""
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

class CategoryClassifierML(NewsAnalyzerML):
    """Clasificador automático de categorías con LR, KNN y Naive Bayes"""
    
    def train_category_classifiers(self):
        """Entrenar modelos LR, KNN y Naive Bayes para clasificar categorías"""
        print("\n🎯 CLASIFICACIÓN AUTOMÁTICA DE CATEGORÍAS (LR vs KNN vs Naive Bayes)")
        print("=" * 80)
        
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
        
        # Entrenar Regresión Logística
        print("🔄 Entrenando Regresión Logística...")
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)
        accuracy_lr = self.mide_error('Regresión Logística', y_pred_lr, y_test)
        
        # Entrenar KNN
        print("🔄 Entrenando KNN...")
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = self.mide_error('KNN', y_pred_knn, y_test)
        
        # Entrenar Naive Bayes (MultinomialNB para texto)
        print("🔄 Entrenando Naive Bayes...")
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train, y_train)
        y_pred_nb = naive_bayes.predict(X_test)
        accuracy_nb = self.mide_error('Naive Bayes', y_pred_nb, y_test)
        
        print(f"\n📈 Comparación de Modelos:")
        print(f"   Regresión Logística: {accuracy_lr:.4f}")
        print(f"   KNN: {accuracy_knn:.4f}")
        print(f"   Naive Bayes: {accuracy_nb:.4f}")
        
        # Determinar el mejor modelo
        accuracies = {'lr': accuracy_lr, 'knn': accuracy_knn, 'nb': accuracy_nb}
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = max(accuracies.values())
        
        print(f"\n🏆 Mejor modelo: {best_model.upper()} ({best_accuracy:.4f})")
        
        print(f"\n📊 Datos del entrenamiento:")
        print(f"   Categorías disponibles: {df_categorized['category'].nunique()}")
        print(f"   Artículos categorizados: {len(df_categorized)}")
        print(f"   Artículos de entrenamiento: {X_train.shape[0]}")
        print(f"   Artículos de prueba: {X_test.shape[0]}")
        
        # Mostrar reporte de clasificación para el mejor modelo
        best_predictions = y_pred_lr if best_model == 'lr' else y_pred_knn if best_model == 'knn' else y_pred_nb
        
        print(f"\n📋 Reporte de Clasificación ({best_model.upper()}):")
        print(classification_report(y_test, best_predictions))
        
        return {
            'logreg': logreg,
            'knn': knn,
            'naive_bayes': naive_bayes,
            'vectorizer': self.vectorizer,
            'accuracies': accuracies,
            'best_model': best_model
        }

class QualityDetectorML(NewsAnalyzerML):
    """Detector de calidad con LR, KNN y Naive Bayes"""
    
    def train_quality_detectors(self):
        """Entrenar modelos LR, KNN y Naive Bayes para detectar calidad"""
        print("\n🔍 DETECCIÓN DE CALIDAD DE CONTENIDO (LR vs KNN vs Naive Bayes)")
        print("=" * 80)
        
        # Definir criterios de calidad
        self.df['quality_score'] = 0
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
        
        # Entrenar Regresión Logística
        print("🔄 Entrenando Regresión Logística...")
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)
        accuracy_lr = self.mide_error('Regresión Logística', y_pred_lr, y_test)
        
        # Entrenar KNN
        print("🔄 Entrenando KNN...")
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = self.mide_error('KNN', y_pred_knn, y_test)
        
        # Entrenar Naive Bayes (GaussianNB para características numéricas)
        print("🔄 Entrenando Naive Bayes...")
        naive_bayes = GaussianNB()
        naive_bayes.fit(X_train, y_train)
        y_pred_nb = naive_bayes.predict(X_test)
        accuracy_nb = self.mide_error('Naive Bayes', y_pred_nb, y_test)
        
        print(f"\n📈 Comparación de Modelos:")
        print(f"   Regresión Logística: {accuracy_lr:.4f}")
        print(f"   KNN: {accuracy_knn:.4f}")
        print(f"   Naive Bayes: {accuracy_nb:.4f}")
        
        # Determinar el mejor modelo
        accuracies = {'lr': accuracy_lr, 'knn': accuracy_knn, 'nb': accuracy_nb}
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = max(accuracies.values())
        
        print(f"\n🏆 Mejor modelo: {best_model.upper()} ({best_accuracy:.4f})")
        
        print(f"\n📊 Distribución de calidad:")
        print(self.df['quality'].value_counts())
        
        return {
            'logreg': logreg,
            'knn': knn,
            'naive_bayes': naive_bayes,
            'accuracies': accuracies,
            'best_model': best_model
        }

class SentimentAnalyzerML(NewsAnalyzerML):
    """Analizador de sentimientos con LR, KNN y Naive Bayes"""
    
    def analyze_sentiments_ml(self):
        """Analizar sentimientos con los tres modelos"""
        print("\n😊 ANÁLISIS DE SENTIMIENTOS (LR vs KNN vs Naive Bayes)")
        print("=" * 80)
        
        # Preparar datos para clasificación binaria (positivo vs no positivo)
        self.df['sentiment_binary'] = self.df['sentiment'].apply(
            lambda x: 1 if x == 'positive' else 0
        )
        
        # Usar características de texto y numéricas
        X_text = self.df['title_clean'] + ' ' + self.df['content_clean']
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Agregar características numéricas
        features_numeric = self.df[['title_length', 'content_length', 'sentiment_score']].fillna(0)
        X_combined = np.hstack([X_vectorized.toarray(), features_numeric.values])
        
        y = self.df['sentiment_binary']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar Regresión Logística
        print("🔄 Entrenando Regresión Logística...")
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)
        y_prob_lr = logreg.predict_proba(X_test)[:, 1]
        accuracy_lr = self.mide_error('Regresión Logística', y_pred_lr, y_test)
        
        # Entrenar KNN
        print("🔄 Entrenando KNN...")
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        y_prob_knn = knn.predict_proba(X_test)[:, 1]
        accuracy_knn = self.mide_error('KNN', y_pred_knn, y_test)
        
        # Entrenar Naive Bayes (BernoulliNB para características binarias)
        print("🔄 Entrenando Naive Bayes...")
        # Binarizar características para BernoulliNB
        X_train_binary = Binarizer().fit_transform(X_train)
        X_test_binary = Binarizer().fit_transform(X_test)
        naive_bayes = BernoulliNB()
        naive_bayes.fit(X_train_binary, y_train)
        y_pred_nb = naive_bayes.predict(X_test_binary)
        y_prob_nb = naive_bayes.predict_proba(X_test_binary)[:, 1]
        accuracy_nb = self.mide_error('Naive Bayes', y_pred_nb, y_test)
        
        print(f"\n📈 Comparación de Modelos:")
        print(f"   Regresión Logística: {accuracy_lr:.4f}")
        print(f"   KNN: {accuracy_knn:.4f}")
        print(f"   Naive Bayes: {accuracy_nb:.4f}")
        
        # Determinar el mejor modelo
        accuracies = {'lr': accuracy_lr, 'knn': accuracy_knn, 'nb': accuracy_nb}
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = max(accuracies.values())
        
        print(f"\n🏆 Mejor modelo: {best_model.upper()} ({best_accuracy:.4f})")
        
        print(f"\n📊 Distribución de sentimientos:")
        sentiment_counts = self.df['sentiment'].value_counts()
        print(sentiment_counts)
        
        return {
            'logreg': logreg,
            'knn': knn,
            'naive_bayes': naive_bayes,
            'accuracies': accuracies,
            'best_model': best_model,
            'sentiment_distribution': sentiment_counts
        }

class EngagementPredictorML(NewsAnalyzerML):
    """Predictor de engagement con LR, KNN y Naive Bayes"""
    
    def train_engagement_predictors(self):
        """Entrenar modelos LR, KNN y Naive Bayes para predecir engagement"""
        print("\n📈 PREDICCIÓN DE ENGAGEMENT (LR vs KNN vs Naive Bayes)")
        print("=" * 80)
        
        # Simular engagement basado en características del artículo
        self.df['engagement_score'] = 0
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
        
        # Entrenar Regresión Logística
        print("🔄 Entrenando Regresión Logística...")
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)
        accuracy_lr = self.mide_error('Regresión Logística', y_pred_lr, y_test)
        
        # Entrenar KNN
        print("🔄 Entrenando KNN...")
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = self.mide_error('KNN', y_pred_knn, y_test)
        
        # Entrenar Naive Bayes (GaussianNB para características numéricas)
        print("🔄 Entrenando Naive Bayes...")
        naive_bayes = GaussianNB()
        naive_bayes.fit(X_train, y_train)
        y_pred_nb = naive_bayes.predict(X_test)
        accuracy_nb = self.mide_error('Naive Bayes', y_pred_nb, y_test)
        
        print(f"\n📈 Comparación de Modelos:")
        print(f"   Regresión Logística: {accuracy_lr:.4f}")
        print(f"   KNN: {accuracy_knn:.4f}")
        print(f"   Naive Bayes: {accuracy_nb:.4f}")
        
        # Determinar el mejor modelo
        accuracies = {'lr': accuracy_lr, 'knn': accuracy_knn, 'nb': accuracy_nb}
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = max(accuracies.values())
        
        print(f"\n🏆 Mejor modelo: {best_model.upper()} ({best_accuracy:.4f})")
        
        print(f"\n📊 Distribución de engagement:")
        print(self.df['engagement'].value_counts())
        
        return {
            'logreg': logreg,
            'knn': knn,
            'naive_bayes': naive_bayes,
            'accuracies': accuracies,
            'best_model': best_model
        }

class SourceClassifierML(NewsAnalyzerML):
    """Clasificador de fuentes con LR, KNN y Naive Bayes"""
    
    def train_source_classifiers(self):
        """Entrenar modelos LR, KNN y Naive Bayes para clasificar fuentes"""
        print("\n📰 CLASIFICACIÓN DE FUENTES (LR vs KNN vs Naive Bayes)")
        print("=" * 80)
        
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
        
        # Entrenar Regresión Logística
        print("🔄 Entrenando Regresión Logística...")
        logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)
        accuracy_lr = self.mide_error('Regresión Logística', y_pred_lr, y_test)
        
        # Entrenar KNN
        print("🔄 Entrenando KNN...")
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = self.mide_error('KNN', y_pred_knn, y_test)
        
        # Entrenar Naive Bayes (MultinomialNB para texto)
        print("🔄 Entrenando Naive Bayes...")
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train, y_train)
        y_pred_nb = naive_bayes.predict(X_test)
        accuracy_nb = self.mide_error('Naive Bayes', y_pred_nb, y_test)
        
        print(f"\n📈 Comparación de Modelos:")
        print(f"   Regresión Logística: {accuracy_lr:.4f}")
        print(f"   KNN: {accuracy_knn:.4f}")
        print(f"   Naive Bayes: {accuracy_nb:.4f}")
        
        # Determinar el mejor modelo
        accuracies = {'lr': accuracy_lr, 'knn': accuracy_knn, 'nb': accuracy_nb}
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = max(accuracies.values())
        
        print(f"\n🏆 Mejor modelo: {best_model.upper()} ({best_accuracy:.4f})")
        
        print(f"\n📈 Fuentes analizadas: {len(reliable_sources)}")
        print(f"📊 Top fuentes por volumen:")
        print(source_counts.head(10))
        
        return {
            'logreg': logreg,
            'knn': knn,
            'naive_bayes': naive_bayes,
            'vectorizer': self.vectorizer,
            'accuracies': accuracies,
            'best_model': best_model
        }

def main():
    """Función principal para ejecutar análisis comparativo LR vs KNN vs Naive Bayes"""
    print("🚀 SISTEMA DE ANÁLISIS DE NOTICIAS: REGRESIÓN LOGÍSTICA vs KNN vs NAIVE BAYES")
    print("=" * 90)
    
    # Inicializar analizador
    analyzer = NewsAnalyzerML()
    
    # Cargar datos
    if not analyzer.load_data():
        return
    
    # Extraer características
    analyzer.extract_features()
    
    print(f"\n📊 RESUMEN DE DATOS:")
    print(f"   Total de artículos: {len(analyzer.df)}")
    print(f"   Categorías únicas: {analyzer.df['category'].nunique()}")
    print(f"   Fuentes únicas: {analyzer.df['source'].nunique()}")
    print(f"   Artículos con imágenes: {analyzer.df['has_image'].sum()}")
    print(f"   Artículos con descripción: {analyzer.df['has_description'].sum()}")
    
    # Ejecutar análisis comparativos
    results = {}
    
    # 1. Clasificación de categorías
    category_classifier = CategoryClassifierML()
    category_classifier.df = analyzer.df.copy()
    results['category'] = category_classifier.train_category_classifiers()
    
    # 2. Detección de calidad
    quality_detector = QualityDetectorML()
    quality_detector.df = analyzer.df.copy()
    results['quality'] = quality_detector.train_quality_detectors()
    
    # 3. Análisis de sentimientos
    sentiment_analyzer = SentimentAnalyzerML()
    sentiment_analyzer.df = analyzer.df.copy()
    results['sentiment'] = sentiment_analyzer.analyze_sentiments_ml()
    
    # 4. Predicción de engagement
    engagement_predictor = EngagementPredictorML()
    engagement_predictor.df = analyzer.df.copy()
    results['engagement'] = engagement_predictor.train_engagement_predictors()
    
    # 5. Clasificación de fuentes
    source_classifier = SourceClassifierML()
    source_classifier.df = analyzer.df.copy()
    results['sources'] = source_classifier.train_source_classifiers()
    
    # Resumen final
    print("\n🏆 RESUMEN COMPARATIVO: REGRESIÓN LOGÍSTICA vs KNN vs NAIVE BAYES")
    print("=" * 90)
    
    for task, result in results.items():
        if result:
            print(f"\n📊 {task.upper()}:")
            print(f"   Regresión Logística: {result['accuracies']['lr']:.4f}")
            print(f"   KNN: {result['accuracies']['knn']:.4f}")
            print(f"   Naive Bayes: {result['accuracies']['nb']:.4f}")
            print(f"   Mejor modelo: {result['best_model'].upper()}")
    
    # Estadísticas generales
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    lr_wins = sum(1 for r in results.values() if r and r['best_model'] == 'lr')
    knn_wins = sum(1 for r in results.values() if r and r['best_model'] == 'knn')
    nb_wins = sum(1 for r in results.values() if r and r['best_model'] == 'nb')
    
    print(f"   Regresión Logística gana: {lr_wins} tareas")
    print(f"   KNN gana: {knn_wins} tareas")
    print(f"   Naive Bayes gana: {nb_wins} tareas")
    
    print("\n🎉 ANÁLISIS COMPARATIVO COMPLETADO EXITOSAMENTE!")
    print("=" * 90)
    print("📋 Funcionalidades implementadas:")
    print("   1. ✅ Clasificación automática de categorías (LR vs KNN vs Naive Bayes)")
    print("   2. ✅ Detección de calidad de contenido (LR vs KNN vs Naive Bayes)")
    print("   3. ✅ Análisis de sentimientos (LR vs KNN vs Naive Bayes)")
    print("   4. ✅ Predicción de engagement (LR vs KNN vs Naive Bayes)")
    print("   5. ✅ Clasificación de fuentes (LR vs KNN vs Naive Bayes)")

if __name__ == "__main__":
    main()
