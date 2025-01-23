import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class EDA:
    """
    Clase para realizar el análisis exploratorio de datos (EDA) en el DataFrame procesado.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa la clase EDA con el DataFrame procesado.

        Args:
            data (pd.DataFrame): DataFrame procesado que contiene los datos combinados.
        """
        self.data = data

    def average_rating_per_book(self) -> pd.DataFrame:
        """
        Calcula las valoraciones promedio por libro y genera una visualización de la distribución de calificaciones promedio por libro único.

        Returns:
            pd.DataFrame: DataFrame con las columnas 'Title' y 'Average Rating'.
        """
        print("Calculando valoraciones promedio por libro...")
        avg_rating = self.data.groupby("Title")["review/score"].mean().reset_index()
        avg_rating.rename(columns={"review/score": "Average Rating"}, inplace=True)

        # Filtrar valores fuera del rango 1-5
        avg_rating = avg_rating[(avg_rating["Average Rating"] >= 1) & (avg_rating["Average Rating"] <= 5)]

        # Visualización: Histograma de distribución de calificaciones promedio
        plt.figure(figsize=(12, 6))
        bins = [1 + i * 0.2 for i in range(21)]  # Generar bins de 1.0 a 5.0 en incrementos de 0.2
        sns.histplot(avg_rating["Average Rating"], bins=bins, kde=False)
        plt.title("Distribución de Calificaciones Promedio por Libro", fontsize=14)
        plt.xlabel("Calificación Promedio (Intervalos de 0.2)")
        plt.ylabel("Cantidad de Libros")
        plt.xticks(bins, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        print("Distribución generada.")

        # Añadir la columna "Average Rating" al DataFrame original
        merged_data = self.data.merge(avg_rating, on="Title", how="left")
        print("Valoraciones promedio calculadas.")
        return merged_data


    def total_reviews_and_ratings(self) -> dict:
        """
        Determina el número total de reseñas y el número total de valoraciones e imprime los resultados.

        Returns:
            dict: Diccionario con las claves 'Total Reviews' y 'Total Ratings'.
        """
        print("Calculando el número total de reseñas y valoraciones...")
        total_reviews = self.data["review/text"].notna().sum()  # Cuenta reseñas no nulas
        total_ratings = self.data["review/score"].notna().sum()  # Cuenta valoraciones no nulas
        results = {"Total Reviews": total_reviews, "Total Ratings": total_ratings}
        print(f"Total de reseñas: {results['Total Reviews']}")
        print(f"Total de calificaciones: {results['Total Ratings']}")
        return results

    def most_popular_authors(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identifica los autores más populares en función de la cantidad de reseñas y genera una visualización.

        Args:
            top_n (int): Número de autores más populares a devolver.

        Returns:
            pd.DataFrame: DataFrame con los autores más populares y el conteo de reseñas.
        """
        print(f"Identificando los {top_n} autores más populares...")
        authors_split = self.data["authors"].str.split(", ")
        authors_exploded = authors_split.explode()  # Separar cada autor en filas
        popular_authors = (
            authors_exploded.value_counts().head(top_n).reset_index()
        )
        popular_authors.columns = ["Author", "Review Count"]

        # Visualización de los autores más populares
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Review Count", y="Author", data=popular_authors)
        plt.title(f"Top {top_n} Autores Más Populares", fontsize=14)
        plt.xlabel("Cantidad de Reseñas")
        plt.ylabel("Autor")
        plt.show()

        print(f"Autores más populares identificados.")
        return popular_authors

    def most_popular_categories(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identifica las categorías más reseñadas y genera una visualización.

        Args:
            top_n (int): Número de categorías más populares a devolver.

        Returns:
            pd.DataFrame: DataFrame con las categorías más reseñadas y el conteo de reseñas.
        """
        print(f"Identificando las {top_n} categorías más populares...")
        categories_split = self.data["categories"].str.split(r",\s+(?![a-z])", regex=True)
        categories_exploded = categories_split.explode()  # Separar cada categoría en filas
        popular_categories = (
            categories_exploded.value_counts().head(top_n).reset_index()
        )
        popular_categories.columns = ["Category", "Review Count"]

        # Visualización de las categorías más populares
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Review Count", y="Category", data=popular_categories)
        plt.title(f"Top {top_n} Categorías Más Populares", fontsize=14)
        plt.xlabel("Cantidad de Reseñas")
        plt.ylabel("Categoría")
        plt.show()

        print(f"Categorías más populares identificadas.")
        return popular_categories


    def visualize_top_books_by_reviews(self):
        """
        Visualiza el top 10 de libros con más reseñas.
        """
        review_counts = self.data.groupby("Title")["review/text"].count().reset_index()
        review_counts.rename(columns={"review/text": "Review Count"}, inplace=True)
        top_books = review_counts.sort_values("Review Count", ascending=False).head(10)

        plt.figure(figsize=(12, 8))
        sns.barplot(x="Review Count", y="Title", data=top_books)
        plt.title("Top 10 Libros con Más Reseñas", fontsize=14)
        plt.xlabel("Cantidad de Reseñas")
        plt.ylabel("Libro")
        plt.show()

    def visualize_top_books_by_ratings(self):
        """
        Visualiza el top 10 de libros mejor calificados con más de 3000 reseñas.
        """
        filtered_data = self.data.groupby("Title").filter(lambda x: len(x) > 3000)
        avg_ratings = filtered_data.groupby("Title")["review/score"].mean().reset_index()
        avg_ratings.rename(columns={"review/score": "Average Rating"}, inplace=True)
        top_books = avg_ratings.sort_values("Average Rating", ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        sns.barplot(x="Average Rating", y="Title", data=top_books)
        plt.title("Top de Libros Mejor Calificados con Más de 3000 Reseñas", fontsize=14)
        plt.xlabel("Calificación Promedio")
        plt.ylabel("Libro")
        plt.show()


    def visualize_top_authors_by_ratings(self, rating: int, top_n: int = 5):
        """
        Visualiza el top de autores con más calificaciones de 5 o 1.

        Args:
            rating (int): Calificación (por ejemplo, 5 o 1).
            top_n (int): Número de autores a mostrar.
        """
        filtered_data = self.data[self.data["review/score"] == rating]
        authors_split = filtered_data["authors"].str.split(", ")
        authors_exploded = authors_split.explode()
        top_authors = authors_exploded.value_counts().head(top_n).reset_index()
        top_authors.columns = ["Author", "Count"]

        plt.figure(figsize=(12, 6))
        sns.barplot(x="Count", y="Author", data=top_authors)
        plt.title(f"Top {top_n} Autores con Calificación {rating}", fontsize=14)
        plt.xlabel("Cantidad de Calificaciones")
        plt.ylabel("Autor")
        plt.show()