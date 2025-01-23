# Análisis de Datos de Libros y Reseñas de Amazon

## Propósito del Proyecto
Este proyecto analiza un conjunto de datos de libros y reseñas de Amazon, utilizando técnicas de procesamiento de lenguaje natural (NLP), análisis exploratorio de datos (EDA) y visualización para extraer insights relevantes. Además, identifica los libros más destacados en función de criterios como número de reseñas, promedio de calificaciones y sentimiento promedio.

## Requisitos del Sistema
- **Python**: Versión 3.8 o superior
- **Dependencias adicionales**: Ver archivo `requirements.txt`

## Configuración del Entorno
1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. Crea un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configura las variables de entorno:
   - Asegúrate de que el archivo `.env` contenga las siguientes rutas:
     ```plaintext
     DATA_PATH=data/raw/
     OUTPUT_PATH=output/
     ```
   - Estas rutas definen dónde se encuentran los archivos de entrada y dónde se guardarán los resultados.

## Descarga de Datos
Los archivos insumo necesarios para el análisis están disponibles en [Amazon Books Reviews Dataset](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data?select=books_data.csv). Descarga los siguientes archivos:
- `books_data.csv`
- `books_rating.csv`

Crea las carpetas necesarias y coloca los archivos en la ubicación especificada:
```bash
mkdir -p data/raw/
mv books_data.csv data/raw/
mv books_rating.csv data/raw/
```

## Ejecución del Proyecto
1. Una vez configurados los datos y las dependencias, ejecuta el archivo `main.py` para realizar todo el flujo de análisis:
   ```bash
   python main.py
   ```

2. Esto generará:
   - **Visualizaciones** interactivas de los datos procesados.
   - **Archivos Excel** con las listas de los mejores libros, que se guardarán en la carpeta definida por `OUTPUT_PATH`.

## Estructura del Proyecto
- **`src/`**: Carpeta que contiene los módulos del proyecto:
  - `data_loader.py`: Carga, limpieza y procesamiento de datos.
  - `eda.py`: Análisis exploratorio de datos y visualizaciones.
  - `sentiment_analysis.py`: Análisis de sentimientos en las reseñas.
  - `best_books.py`: Identificación de los mejores libros.
- **`main.py`**: Script principal que ejecuta todo el flujo del proyecto.
- **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.
- **`.env`**: Archivo de configuración que define rutas para datos y salidas.