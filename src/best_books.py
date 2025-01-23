import pandas as pd
import os
from dotenv import load_dotenv


class BestBooks:
    """
    Clase para identificar y exportar los mejores libros según criterios específicos.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa la clase BestBooks con el DataFrame procesado.

        Args:
            data (pd.DataFrame): DataFrame procesado que contiene la información de los libros.
        """
        self.data = data
        load_dotenv()
        self.output_path = os.getenv("OUTPUT_PATH")
        if not self.output_path:
            raise ValueError("La ruta de salida (OUTPUT_PATH) no está definida en el archivo .env.")

    def top_books_by_reviews(self, top_n: int = 10):
        """
        Exporta los libros con más reseñas a un archivo Excel.

        Args:
            top_n (int): Número de libros a incluir.
        """
        print("Identificando los libros con más reseñas...")
        aggregated_data = self._aggregate_book_data()
        books_by_reviews = aggregated_data.sort_values(by="Review Count", ascending=False).head(top_n)
        books_by_reviews = books_by_reviews[["Title", "authors", "categories", "Review Count"]]

        output_file = os.path.join(self.output_path, "top_libros_numero_resenas.xlsx")
        books_by_reviews.to_excel(output_file, index=False)
        print(f"Archivo exportado: {output_file}")

    def top_books_by_average_rating(self, top_n: int = 10):
        """
        Exporta los libros con las mejores calificaciones promedio a un archivo Excel.

        Args:
            top_n (int): Número de libros a incluir.
        """
        print("Identificando los libros con las mejores calificaciones promedio...")
        aggregated_data = self._aggregate_book_data()
        books_by_average_rating = aggregated_data.sort_values(by="Average Rating", ascending=False).head(top_n)
        books_by_average_rating = books_by_average_rating[["Title", "authors", "categories", "Average Rating"]]

        output_file = os.path.join(self.output_path, "top_libros_calificacion_promedio.xlsx")
        books_by_average_rating.to_excel(output_file, index=False)
        print(f"Archivo exportado: {output_file}")

    def top_books_by_sentiment(self, top_n: int = 10):
        """
        Exporta los libros con el sentimiento promedio más positivo a un archivo Excel.

        Args:
            top_n (int): Número de libros a incluir.
        """
        print("Identificando los libros con el sentimiento promedio más positivo...")
        aggregated_data = self._aggregate_book_data()
        books_by_sentiment = aggregated_data.sort_values(by="Average Sentiment", ascending=False).head(top_n)
        books_by_sentiment = books_by_sentiment[["Title", "authors", "categories", "Average Sentiment"]]

        output_file = os.path.join(self.output_path, "top_libros_sentimiento_promedio.xlsx")
        books_by_sentiment.to_excel(output_file, index=False)
        print(f"Archivo exportado: {output_file}")

    def _aggregate_book_data(self) -> pd.DataFrame:
        """
        Agrega los datos por libro, calculando el conteo de reseñas, promedio de puntaje, y promedio de sentimiento.

        Returns:
            pd.DataFrame: DataFrame con las columnas 'Title', 'authors', 'categories', 'Review Count', 'Average Rating', y 'Average Sentiment'.
        """
        print("Agregando datos por libro...")
        aggregated_data = (
            self.data.groupby(["Title", "authors", "categories"])
            .agg(
                Review_Count=("review/text", "count"),
                Average_Rating=("review/score", "mean"),
                Average_Sentiment=("compound", "mean"),
            )
            .reset_index()
        )
        print("Datos agregados correctamente.")
        return aggregated_data