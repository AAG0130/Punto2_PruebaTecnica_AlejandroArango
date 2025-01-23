import pandas as pd

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
        Calcula las valoraciones promedio por libro.

        Returns:
            pd.DataFrame: DataFrame con las columnas 'Title' y 'Average Rating'.
        """
        print("Calculando valoraciones promedio por libro...")
        avg_rating = self.data.groupby("Title")["review/score"].mean().reset_index()
        avg_rating.rename(columns={"review/score": "Average Rating"}, inplace=True)
        print("Valoraciones promedio calculadas.")
        return avg_rating

    def total_reviews_and_ratings(self) -> dict:
        """
        Determina el número total de reseñas y el número total de valoraciones.

        Returns:
            dict: Diccionario con las claves 'Total Reviews' y 'Total Ratings'.
        """
        print("Calculando el número total de reseñas y valoraciones...")
        total_reviews = len(self.data)
        total_ratings = self.data["ratingsCount"].sum()
        print("Cálculos completados.")
        return {"Total Reviews": total_reviews, "Total Ratings": total_ratings}

    def most_popular_authors(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identifica los autores más populares en función de la cantidad de reseñas.

        Args:
            top_n (int): Número de autores más populares a devolver.

        Returns:
            pd.DataFrame: DataFrame con los autores más populares y el conteo de reseñas.
        """
        print(f"Identificando los {top_n} autores más populares...")
        popular_authors = (
            self.data["authors"]
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        popular_authors.columns = ["Author", "Review Count"]
        print(f"Autores más populares identificados.")
        return popular_authors

    def most_popular_categories(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identifica las categorías más reseñadas.

        Args:
            top_n (int): Número de categorías más populares a devolver.

        Returns:
            pd.DataFrame: DataFrame con las categorías más reseñadas y el conteo de reseñas.
        """
        print(f"Identificando las {top_n} categorías más populares...")
        categories_split = self.data["categories"].str.split(", ")
        categories_exploded = categories_split.explode()
        popular_categories = (
            categories_exploded.value_counts().head(top_n).reset_index()
        )
        popular_categories.columns = ["Category", "Review Count"]
        print(f"Categorías más populares identificadas.")
        return popular_categories