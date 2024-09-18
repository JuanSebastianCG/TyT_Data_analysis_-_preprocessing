Aquí tienes un ejemplo de cómo estructurar el archivo `README.md` para tu proyecto. Este archivo sigue buenas prácticas y proporciona a los usuarios o colaboradores una descripción clara de la estructura del proyecto, cómo configurarlo, y cómo utilizarlo.

---

# **TyT Data Analysis & Preprocessing**

Este repositorio contiene las herramientas y módulos para la extracción, preprocesamiento, análisis y visualización de datos de las pruebas Saber TyT. Los módulos están organizados para permitir una fácil modificación, escalabilidad y uso modular en diferentes etapas del pipeline de ciencia de datos.

## **Tabla de Contenidos**

1. [Estructura del Proyecto](#estructura-del-proyecto)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Descripción de los Módulos](#descripción-de-los-módulos)
---

## **Estructura del Proyecto**

```bash
TyT_carolina/
┣ .venv/                          # Entorno virtual Python
┣ .vscode/                        # Configuraciones del entorno VSCode
┃ ┗ settings.json
┣ data/                           # Almacenamiento de los datos
┃ ┣ external/                     # Datos externos o sin procesar
┃ ┣ processed/                    # Datos procesados
┃ ┃ ┣ dataPreprocessingService/   # encoders
┃ ┃ ┣ dataProcessed/              # final data
┃ ┃ ┗ features/
┃ ┗ raw/                          # Datos sin procesar del sistema TyT
┃   ┗ SaberTyT_Genericas_20221.TXT
┣ docs/                           # Documentación del proyecto
┣ model/                          # Módulos del modelo y procesamiento de datos
┃ ┣ config/                       # Configuración general del proyecto
┃ ┣ controller/                   # Controladores
┃ ┣ core/                         # Módulos de procesamiento de datos
┃ ┗ utils/                        # Utilidades generales
┣ .env                            # Variables de entorno
┣ .gitignore                      # Archivos a ignorar por git
┣ README.md                       # Este archivo
┗ requirements.txt                # Dependencias del proyecto
```

## **Requisitos**

Para poder ejecutar este proyecto localmente, necesitas tener instalado lo siguiente:

- **Python 3.11** o superior
- **pip** (gestor de paquetes de Python)
- Un entorno de trabajo como **Visual Studio Code** (opcional, pero recomendado)
  
### **Bibliotecas de Python necesarias**:

Asegúrate de que todas las dependencias estén instaladas:

```bash
pip install -r requirements.txt
```

## **Instalación**

### 1. **Clonar el Repositorio**

```bash
git clone https://github.com/tu_usuario/TyT_carolina.git
cd TyT_carolina
```

### 2. **Crear un Entorno Virtual**

```bash
python -m venv .venv
source .venv/bin/activate  # Para Linux o macOS
# Para Windows: .venv\Scripts\activate
```

### 3. **Instalar Dependencias**

```bash
pip install -r requirements.txt
```

### 4. **Configurar Variables de Entorno**

Define las variables de entorno necesarias en el archivo `.env`. Un ejemplo básico de variables de entorno podría ser:

```bash
PYTHONPATH=src/..
```

### 5. **Editar Configuraciones (opcional)**

Revisa el archivo `config.yaml` en la carpeta `model/config/` para personalizar rutas o configuraciones específicas del proyecto.

## **Uso**

### 1. **Preprocesamiento de Datos**

Carga y transforma los datos crudos del sistema de pruebas Saber TyT utilizando el siguiente comando en Jupyter Notebook o un archivo `.py`:

```python
from model.core.dataPreprocessing.dataCleaner import DataCleaner
from model.utils.dataExtractor import DataExtractor

# Cargar y limpiar los datos
data = DataExtractor.load_data_txt_to_dataframe("data/raw/SaberTyT_Genericas_20221.TXT")
cleaner = DataCleaner(X=data)
cleaner.convert_int_to_float()
```

### 2. **Visualización de los Datos**

Utiliza el módulo de visualización para crear gráficos, histogramas o diagramas de correlación:

```python
from model.utils.dataVisualization import DataVisualization

visualizer = DataVisualization()
visualizer.visualize_data(data, kind='histogram')
```

## **Descripción de los Módulos**

### 1. **model/config/**
Contiene los archivos de configuración (`config.yaml`) y los loaders correspondientes para facilitar la configuración del proyecto.

### 2. **model/core/**
Contiene las funciones centrales del preprocesamiento de datos, tales como:
- `dataCleaner.py`: Limpieza de datos, normalización, manejo de datos faltantes.
- `encoders.py`: Codificación de variables categóricas.
- `outlierDetector.py`: Detección de valores atípicos.

### 3. **model/utils/**
Aquí se incluyen utilidades como:
- `dataExtractor.py`: Carga y transformación de los datos crudos a un formato estructurado.
- `dataVisualization.py`: Herramientas para graficar y visualizar los datos.

## **Buenas Prácticas**

- **Modularización**: El proyecto está dividido en módulos específicos para garantizar su escalabilidad.
- **Documentación**: Mantén la documentación actualizada en `docs/` y asegúrate de que todas las funciones tengan comentarios claros.
- **Entorno virtual**: Asegúrate de utilizar siempre el entorno virtual (`.venv/`) para evitar conflictos de dependencias.
- **Uso de `dotenv`**: Configura correctamente las variables de entorno en el archivo `.env` para que el código sea portable y adaptable a diferentes entornos.


