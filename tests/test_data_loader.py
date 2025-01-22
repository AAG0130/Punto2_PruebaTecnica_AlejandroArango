import unittest
import os
from src.data_loader import DataLoader
import pandas as pd


class TestDataLoader(unittest.TestCase):
    """
    Clase de pruebas para el módulo DataLoader.
    """

    def setUp(self):
        """
        Configuración inicial antes de cada prueba.
        """
        # Crear una instancia de DataLoader
        self.loader = DataLoader()

        # Crear datos de prueba
        self.books_data = pd.DataFrame({
            "Title": ["Book A", "Book B", "Book C"],
            "authors": ["['Author1']", "['Author2']", "['Author3']"],
            "categories": ["['Category1']", "['Category2']", "['Category3']"],
            "ratingsCount": [100, 200, 300]
        })

        self.books_rating = pd.DataFrame({
            "Title": ["Book A", "Book B", "Book D"],  # Book D no está en books_data
            "review/score": [4.5, 4.0, 3.5],
            "review/text": ["Great book!", "Good book.", "Not bad."]
        })

    def test_load_data(self):
        """
        Prueba para verificar que los datos se cargan correctamente desde el archivo.
        """
        # Simular la ruta de datos
        data_path = os.getenv("DATA_PATH")
        self.assertTrue(data_path, "La variable DATA_PATH no está configurada en el archivo .env.")

        # Verificar que las rutas de los archivos existen
        self.assertTrue(os.path.exists(os.path.join(data_path, "books_data.csv")), "El archivo books_data.csv no existe.")
        self.assertTrue(os.path.exists(os.path.join(data_path, "books_rating.csv")), "El archivo books_rating.csv no existe.")

    def test_process_data(self):
        """
        Prueba para verificar el procesamiento de los datos.
        """
        # Crear un diccionario simulado de datos cargados
        data = {
            "books_data": self.books_data,
            "books_rating": self.books_rating
        }

        # Procesar los datos
        processed_data = self.loader.process_data(data)

        # Verificar el número de registros después del merge
        self.assertEqual(len(processed_data), 2, "El número de registros después del merge no es el esperado.")

        # Verificar columnas finales
        expected_columns = ['Title', 'review/score', 'review/text', 'authors', 'categories', 'ratingsCount']
        self.assertListEqual(list(processed_data.columns), expected_columns, "Las columnas procesadas no coinciden con las esperadas.")

        # Verificar que se han limpiado correctamente las columnas 'authors' y 'categories'
        self.assertEqual(processed_data['authors'].iloc[0], "Author1", "La limpieza de 'authors' no funcionó correctamente.")
        self.assertEqual(processed_data['categories'].iloc[0], "Category1", "La limpieza de 'categories' no funcionó correctamente.")

    def test_clean_text_format(self):
        """
        Prueba para verificar el formato limpio de las columnas 'authors' y 'categories'.
        """
        # Valores de ejemplo con corchetes y comillas
        data_with_quotes = pd.DataFrame({
            "authors": ["['Author1']", "['Another Author']"],
            "categories": ["['Category1']", "['Another Category']"]
        })

        # Simular limpieza en ambas columnas
        data_with_quotes["authors"] = data_with_quotes["authors"].str.extract(r"'(.*?)'")
        data_with_quotes["categories"] = data_with_quotes["categories"].str.extract(r"'(.*?)'")

        # Verificar que el formato limpio es el esperado
        self.assertEqual(data_with_quotes["authors"].iloc[0], "Author1", "La limpieza de 'authors' no funcionó correctamente.")
        self.assertEqual(data_with_quotes["categories"].iloc[0], "Category1", "La limpieza de 'categories' no funcionó correctamente.")


if __name__ == "__main__":
    unittest.main()