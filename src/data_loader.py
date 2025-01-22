import os
import pandas as pd
from dotenv import load_dotenv

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

    def process_data(self, data: dict) -> pd.DataFrame:
        """
        Procesa los datos combinando ambos DataFrames, limpiando columnas y eliminando duplicados y nulos.

        Args:
            data (dict): Diccionario con los DataFrames cargados.

        Returns:
            pd.DataFrame: DataFrame combinado y limpio.
        """
        try:
            # Reducir columnas antes del merge
            print("Filtrando columnas necesarias en DataFrames originales...")
            data["books_data"] = data["books_data"][["Title", "authors", "categories", "ratingsCount"]]
            data["books_rating"] = data["books_rating"][["Title", "review/score", "review/text"]]

            # Unir ambos DataFrames por la columna "Title"
            print("Uniendo DataFrames por la columna 'Title'...")
            merged_df = pd.merge(
                data["books_data"],
                data["books_rating"],
                on="Title",
                how="inner"
            )

            # Eliminar duplicados y nulos del DataFrame combinado
            print("Eliminando duplicados y valores nulos del DataFrame combinado...")
            merged_df = merged_df.drop_duplicates().dropna()

            # Limpiar columnas específicas
            print("Limpiando columnas 'authors' y 'categories'...")
            for col in ['authors', 'categories']:
                merged_df[col] = merged_df[col].str.extract(r"'(.*?)'")

            print("Procesamiento completado.")
            return merged_df
        except Exception as e:
            print(f"Error al procesar los datos: {e}")
            return pd.DataFrame()
