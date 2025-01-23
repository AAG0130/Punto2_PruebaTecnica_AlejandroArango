from src.data_loader import DataLoader
from src.eda import EDA
from src.sentiment_analysis import SentimentAnalysis
from src.best_books import BestBooks


def main():
    """
    Ejecuta el flujo completo del análisis de datos.
    """
    # Inicializar el cargador de datos
    data_loader = DataLoader()

    # Cargar y procesar los datos
    print("Cargando y procesando los datos...")
    raw_data = data_loader.load_data()
    processed_data, unmatched_data = data_loader.process_data(raw_data)

    if processed_data.empty:
        print("No se pudo procesar la información. Verifique los datos de entrada.")
        return

    # Mostrar información sobre los registros no coincidentes
    print(f"Registros no coincidentes:\n{unmatched_data}")

    # Iniciar el análisis exploratorio
    print("\nIniciando análisis exploratorio de datos (EDA)...")
    eda = EDA(processed_data)

    # Visualización: Valoraciones promedio por libro
    print("\nGenerando visualización: Valoraciones promedio por libro...")
    eda.average_rating_per_book()

    # Total de reseñas y valoraciones
    print("\nCalculando total de reseñas y valoraciones...")
    eda.total_reviews_and_ratings()

    # Visualización: Autores más populares
    print("\nGenerando visualización: Autores más populares...")
    eda.most_popular_authors()

    # Visualización: Categorías más populares
    print("\nGenerando visualización: Categorías más populares...")
    eda.most_popular_categories()

    # Visualización: Top 10 libros con más reseñas
    print("\nGenerando visualización: Top 10 libros con más reseñas...")
    eda.visualize_top_books_by_reviews()

    # Visualización: Top 10 libros mejor calificados con más de 1000 reseñas
    print("\nGenerando visualización: Top 10 libros mejor calificados con más de 3000 reseñas...")
    eda.visualize_top_books_by_ratings()

    # Visualización: Top 5 autores con más calificaciones de 5
    print("\nGenerando visualización: Top 5 autores con más calificaciones de 5...")
    eda.visualize_top_authors_by_ratings(rating=5)

    # Visualización: Top 5 autores con más calificaciones de 1
    print("\nGenerando visualización: Top 5 autores con más calificaciones de 1...")
    eda.visualize_top_authors_by_ratings(rating=1)

    # Iniciar el análisis de sentimientos
    print("\nIniciando análisis de sentimientos...")
    sentiment_analyzer = SentimentAnalysis(processed_data)

    # Preprocesar texto de las reseñas
    sentiment_analyzer.preprocess_text()

    # Calcular puntuaciones de sentimiento
    processed_data = sentiment_analyzer.calculate_sentiment_scores()

    # Visualizaciones de distribución de sentimientos
    print("\nGenerando visualizaciones de la distribución de sentimientos...")
    sentiment_analyzer.visualize_sentiment_distribution()

    # Visualización: Top 20 libros con más reseñas positivas, neutras y negativas
    print("\nGenerando visualización: Top 20 libros por tipo de sentimiento...")
    sentiment_analyzer.visualize_top_books_by_sentiment()

    # Visualización: Top 20 autores por puntuación promedio de sentimiento
    print("\nGenerando visualización: Autores con puntuación promedio más alta y más baja...")
    sentiment_analyzer.visualize_top_authors_by_sentiment_score()

    # Visualización: Top 20 autores con más reseñas positivas y negativas
    print("\nGenerando visualización: Autores con más reseñas positivas y negativas...")
    sentiment_analyzer.visualize_top_authors_by_review_sentiment("positivo")
    sentiment_analyzer.visualize_top_authors_by_review_sentiment("negativo")

    # Visualización: Top 20 categorías con más reseñas positivas y negativas
    print("\nGenerando visualización: Categorías con más reseñas positivas y negativas...")
    sentiment_analyzer.visualize_top_categories_by_review_sentiment("positivo")
    sentiment_analyzer.visualize_top_categories_by_review_sentiment("negativo")

    # Análisis de sentimiento promedio por libro
    print("\nCalculando sentimiento promedio por libro...")
    sentiment_analyzer.average_sentiment_by_book()

    # Análisis de sentimiento promedio por categoría
    print("\nCalculando sentimiento promedio por categoría...")
    sentiment_analyzer.average_sentiment_by_category()

    # Identificar y exportar los mejores libros
    print("\nIdentificando y exportando los mejores libros...")
    best_books = BestBooks(processed_data)  # Usar el processed_data actualizado
    best_books.top_books_by_reviews()
    best_books.top_books_by_average_rating()
    best_books.top_books_by_sentiment()

    print("\nAnálisis finalizado.")


if __name__ == "__main__":
    main()
