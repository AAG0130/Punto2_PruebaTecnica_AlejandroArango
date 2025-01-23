import os
import pandas as pd
from dotenv import load_dotenv
import ast  # Para evaluar cadenas con listas como Python objects
from typing import Tuple


class DataLoader:
    """
    Clase para la carga, limpieza y procesamiento de datos desde una ubicación especificada.
    """

    def __init__(self):
        """
        Inicializa la clase DataLoader y carga la configuración desde el archivo .env.
        """
        load_dotenv()  # Carga las variables del archivo .env
        self.data_path = os.getenv("DATA_PATH")  # Ruta de los datos
        if not self.data_path:
            raise ValueError("La ruta de los datos (DATA_PATH) no está definida en el archivo .env.")

    def load_data(self) -> dict:
        """
        Carga los datos desde la ubicación especificada.

        Returns:
            dict: Un diccionario con los DataFrames cargados.
        """
        try:
            print(f"Cargando datos desde: {self.data_path}")
            files = {
                "books_data": os.path.join(self.data_path, "books_data.csv"),
                "books_rating": os.path.join(self.data_path, "books_rating.csv")
            }

            for file_name, file_path in files.items():
                print(f"Buscando {file_name} en {file_path}")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"El archivo {file_name} no se encuentra en {file_path}")

            # Cargar los datos en DataFrames
            data = {
                "books_data": pd.read_csv(files["books_data"]),
                "books_rating": pd.read_csv(files["books_rating"])
            }
            print("Datos cargados correctamente.")
            return data
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return {}

    @staticmethod
    def clean_column(value):
        """
        Limpia y une valores de una columna que contienen listas en formato string.

        Args:
            value (str): Valor de la columna.

        Returns:
            str: Cadena limpia con valores unidos por comas.
        """
        try:
            # Convierte la cadena como lista y une sus elementos
            return ", ".join(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            # Si falla, devuelve el valor original o NaN
            return value

    def process_data(self, data: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Procesa los datos combinando ambos DataFrames, limpiando columnas y eliminando duplicados y nulos.

        Args:
            data (dict): Diccionario con los DataFrames cargados.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrame combinado y limpio, y DataFrame con registros no coincidentes.
        """
        try:
            # Reducir columnas antes del merge
            print("Filtrando columnas necesarias en DataFrames originales...")
            data["books_data"] = data["books_data"][["Title", "authors", "categories", "ratingsCount"]]
            data["books_rating"] = data["books_rating"][["Title", "review/score", "review/text"]]

            # Identificar registros no coincidentes
            print("Identificando registros no coincidentes...")
            unmatched_ratings = data["books_rating"][~data["books_rating"]["Title"].isin(data["books_data"]["Title"])]

            # Unir ambos DataFrames por la columna "Title"
            print("Uniendo DataFrames por la columna 'Title'...")
            merged_df = pd.merge(
                data["books_data"],
                data["books_rating"],
                on="Title",
                how="inner"
            )

            # Limpiar columnas específicas con la función optimizada
            print("Limpiando columnas 'authors' y 'categories'...")
            merged_df["authors"] = merged_df["authors"].apply(self.clean_column)
            merged_df["categories"] = merged_df["categories"].apply(self.clean_column)

            # Eliminar registros con "review/text" vacío o NaN
            print("Eliminando registros con 'review/text' vacío o NaN...")
            merged_df = merged_df[~merged_df["review/text"].isnull() & (merged_df["review/text"] != "")]

            # Eliminar duplicados
            print("Eliminando duplicados...")
            merged_df = merged_df.drop_duplicates()

            print("Procesamiento completado.")
            return merged_df, unmatched_ratings
        except Exception as e:
            print(f"Error al procesar los datos: {e}")
            return pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    # Inicializar el cargador de datos
    loader = DataLoader()

    # Cargar los datos
    data = loader.load_data()
    if data:
        # Procesar los datos
        processed_data, unmatched_data = loader.process_data(data)

        # Verificar los resultados
        print(f"DataFrame procesado:\n{processed_data.head()}")
        print(f"Número total de registros: {len(processed_data)}")

        # Mostrar los registros no coincidentes
        print(f"Registros no coincidentes en 'books_rating':\n{unmatched_data}")
        print(f"Número total de registros no coincidentes: {len(unmatched_data)}")
