import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


class SentimentAnalysis:
    """
    Clase para realizar análisis de sentimientos en reseñas de libros.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa la clase SentimentAnalysis con el DataFrame procesado.

        Args:
            data (pd.DataFrame): DataFrame procesado que contiene las reseñas de libros.
        """
        self.data = data
        self.analyzer = SentimentIntensityAnalyzer()

    def preprocess_text(self):
        """
        Limpia y estandariza las reseñas para el análisis de sentimiento.
        """
        print("Preprocesando texto de las reseñas...")
        self.data["clean_reviews"] = self.data["review/text"].str.lower().fillna("")
        print("Texto preprocesado.")

    def calculate_sentiment_scores(self):
        """
        Calcula las puntuaciones de sentimiento (compound) y clasifica el sentimiento.
        """
        print("Calculando puntuaciones de sentimiento...")
        self.data["score"] = self.data["clean_reviews"].apply(
            lambda review: self.analyzer.polarity_scores(review)
        )
        self.data["compound"] = self.data["score"].apply(lambda x: x["compound"])
        self.data["Sentiment"] = self.data["compound"].apply(self._classify_sentiment)
        print("Puntuaciones de sentimiento calculadas.")

        # Retornar el DataFrame actualizado
        return self.data

    @staticmethod
    def _classify_sentiment(compound: float) -> str:
        """
        Clasifica el sentimiento basado en la puntuación compuesta.

        Args:
            compound (float): Puntuación compuesta.

        Returns:
            str: Clasificación del sentimiento ('positivo', 'neutral', 'negativo').
        """
        if compound > 0.05:
            return "positivo"
        elif compound < -0.05:
            return "negativo"
        return "neutral"

    def visualize_sentiment_distribution(self):
        """
        Genera visualizaciones para la distribución de sentimientos.
        """
        print("Generando visualizaciones de la distribución de sentimientos...")
        self._plot_sentiment_pie_chart()
        self._plot_sentiment_histogram()
        self._plot_sentiment_bar_chart()

    def _plot_sentiment_pie_chart(self):
        """
        Genera un gráfico de pastel para la distribución de sentimientos.
        """
        plt.figure(figsize=(6, 6))
        labels = ["Positivo", "Negativo", "Neutral"]
        sizes = self.data["Sentiment"].value_counts()
        colors = ["green", "red", "blue"]
        explode = (0.1, 0.1, 0.1)

        plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=True,
        )
        plt.title("Sentiment Distribution")
        plt.show()

    def _plot_sentiment_histogram(self):
        """
        Genera un histograma para la distribución de puntuaciones compuestas.
        """
        plt.figure(figsize=(8, 6))
        sentiments = {
            "Positivo": self.data[self.data["compound"] > 0]["compound"],
            "Negativo": self.data[self.data["compound"] < 0]["compound"],
            "Neutral": self.data[self.data["compound"] == 0]["compound"],
        }
        colors = ["green", "red", "orange"]

        for sentiment, color in zip(sentiments, colors):
            plt.hist(sentiments[sentiment], bins=20, alpha=0.5, label=sentiment, color=color)

        plt.title("Distribución de sentimientos")
        plt.xlabel("Puntuación compuesta")
        plt.ylabel("Cantidad")
        plt.legend()
        plt.show()

    def _plot_sentiment_bar_chart(self):
        """
        Genera un gráfico de barras para la distribución de sentimientos.
        """
        self.data["Sentiment"].value_counts().plot(kind="bar", figsize=(8, 5), color=["green", "red", "blue"])
        plt.title("Distribución de sentimientos", fontsize=15)
        plt.xlabel("Sentimiento")
        plt.ylabel("Cantidad")
        plt.grid(axis="y")
        plt.show()

    def visualize_top_books_by_sentiment(self):
        """
        Genera visualizaciones de los libros con la mayor cantidad de reseñas por tipo de sentimiento.
        """
        sentiments = ["positivo", "neutral", "negativo"]
        titles = [
            "Top 20 libros con más reseñas positivas",
            "Top 20 libros con más reseñas neutrales",
            "Top 20 libros con más reseñas negativas",
        ]
        colors = ["green", "blue", "red"]

        for sentiment, title, color in zip(sentiments, titles, colors):
            self._plot_top_books_by_sentiment(sentiment, title, color)

    def _plot_top_books_by_sentiment(self, sentiment: str, title: str, color: str):
        """
        Genera un gráfico de barras para los libros con la mayor cantidad de reseñas de un tipo de sentimiento.

        Args:
            sentiment (str): Tipo de sentimiento ('positivo', 'neutral', 'negativo').
            title (str): Título de la visualización.
            color (str): Color del gráfico.
        """
        sentiment_data = self.data[self.data["Sentiment"] == sentiment]["Title"].value_counts().head(20)
        sentiment_data.plot(kind="bar", figsize=(8, 6), color=color)
        plt.title(title, fontsize=14)
        plt.xlabel("Titulo del libro")
        plt.ylabel("Cantidad de reseñas")
        plt.xticks(rotation=90)
        plt.show()

    def visualize_top_authors_by_sentiment_score(self, top_n=20):
        """
        Muestra los 20 autores con las calificaciones promedio más altas y más bajas.

        Args:
            top_n (int): Número de autores a mostrar.
        """
        print("Generando visualización: Autores con calificaciones promedio más altas y bajas...")

        # Explota autores para desagregarlos
        authors_split = self.data["authors"].str.split(", ")
        exploded_data = self.data.assign(authors=authors_split).explode("authors")

        # Calcula calificación promedio por autor
        author_sentiment = (
            exploded_data.groupby("authors")["compound"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        # Autores con calificaciones más altas
        top_authors = author_sentiment.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_authors["authors"], top_authors["compound"], color="green")
        plt.title(f"Top {top_n} Autores con Calificaciones Promedio Más Altas")
        plt.xlabel("Calificación Promedio (Compound)")
        plt.ylabel("Autor")
        plt.gca().invert_yaxis()
        plt.show()

        # Autores con calificaciones más bajas
        bottom_authors = author_sentiment.tail(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(bottom_authors["authors"], bottom_authors["compound"], color="red")
        plt.title(f"Top {top_n} Autores con Calificaciones Promedio Más Bajas")
        plt.xlabel("Calificación Promedio (Compound)")
        plt.ylabel("Autor")
        plt.gca().invert_yaxis()
        plt.show()

    def visualize_top_authors_by_review_sentiment(self, sentiment: str, top_n=20):
        """
        Muestra los 20 autores con la mayor cantidad de reseñas positivas o negativas.

        Args:
            sentiment (str): Tipo de sentimiento ('positivo' o 'negativo').
            top_n (int): Número de autores a mostrar.
        """
        print(f"Generando visualización: Top {top_n} autores con más reseñas {sentiment}...")

        # Filtrar los datos según el sentimiento
        filtered_data = self.data[self.data["Sentiment"] == sentiment]

        # Contar la cantidad de reseñas por autor
        author_counts = (
            filtered_data["authors"]
            .str.split(", ")
            .explode()
            .value_counts()
            .head(top_n)
        )

        # Verificar si hay datos para graficar
        if author_counts.empty:
            print("No se encontraron autores para el tipo de sentimiento especificado.")
            return

        # Crear la visualización
        plt.figure(figsize=(10, 6))
        author_counts.sort_values().plot(kind="barh", color="blue" if sentiment == "positivo" else "red")
        plt.title(f"Top {top_n} Autores con Más Reseñas {sentiment.capitalize()}")
        plt.xlabel("Cantidad de Reseñas")
        plt.ylabel("Autor")
        plt.tight_layout()
        plt.show()


    def visualize_top_categories_by_review_sentiment(data: pd.DataFrame, sentiment: str, top_n=20):
        """
        Muestra las categorías con la mayor cantidad de reseñas positivas o negativas.
    
        Args:
            data (pd.DataFrame): DataFrame con los datos procesados.
            sentiment (str): Tipo de sentimiento ('positivo' o 'negativo').
            top_n (int): Número de categorías a mostrar.
        """
        print(f"Generando visualización: Top {top_n} categorías con más reseñas {sentiment}...")
    
        # Filtrar los datos por el tipo de sentimiento
        filtered_data = data[data["Sentiment"] == sentiment]
    
        # Explotar las categorías y contar reseñas
        category_counts = (
            filtered_data["categories"]
            .str.split(", ")
            .explode()
            .value_counts()
            .head(top_n)
        )
    
        if category_counts.empty:
            print("No se encontraron categorías para el tipo de sentimiento especificado.")
            return
    
        # Generar la visualización
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        category_counts.sort_values().plot(kind="barh", color="blue" if sentiment == "positivo" else "red")
        plt.title(f"Top {top_n} Categorías con Más Reseñas {sentiment.capitalize()}")
        plt.xlabel("Cantidad de Reseñas")
        plt.ylabel("Categoría")
        plt.tight_layout()
        plt.show()


    def average_sentiment_by_book(self) -> tuple:
        """
        Calcula el sentimiento promedio para un libro dado.

        Returns:
            tuple: (Promedio de puntuación compuesta, Sentimiento).
        """
        title = input("Ingrese el título del libro: ")
        filtered_data = self.data[self.data["Title"] == title]

        if filtered_data.empty:
            print(f"El libro '{title}' no fue encontrado.")
            return None

        avg_compound = filtered_data["compound"].mean()
        sentiment = self._classify_sentiment(avg_compound)
        print(f"Sentimiento promedio del libro '{title}': {sentiment} ({avg_compound:.2f})")
        return avg_compound, sentiment

    def average_sentiment_by_category(self) -> tuple:
        """
        Calcula el sentimiento promedio para una categoría dada.

        Returns:
            tuple: (Promedio de puntuación compuesta, Sentimiento).
        """
        category = input("Ingrese la categoría: ")
        filtered_data = self.data.explode("categories")
        filtered_data = filtered_data[filtered_data["categories"] == category]

        if filtered_data.empty:
            print(f"La categoría '{category}' no fue encontrada.")
            return None

        avg_compound = filtered_data["compound"].mean()
        sentiment = self._classify_sentiment(avg_compound)
        print(f"Sentimiento promedio de la categoría '{category}': {sentiment} ({avg_compound:.2f})")
        return avg_compound, sentiment
