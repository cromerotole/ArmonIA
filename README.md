# 🎵 armonIA: Transformer for Chord Prediction
Repositorio del Trabajo de Fin de Grado (TFG) en la UPM.  
El proyecto consiste en el **diseño de una herramienta de apoyo a la composición musical** mediante redes neuronales secuenciales (Transformer vs. LSTM).  
Incluye notebooks de preprocesamiento, entrenamiento y métricas, y una aplicación interactiva con **Gradio**.

## 📂 Estructura del repositorio
armonIA/
├─ app/
│ └─ app.py # Interfaz de usuario (Gradio UI)
├─ models/
│ ├─ checkpoint_best.pt # Modelo Transformer entrenado
│ ├─ checkpoint_last.pt # Último checkpoint del entrenamiento
│ └─ config.json # Configuración del modelo
├─ data/
│ ├─ chord_to_idx.json # Diccionario acorde→índice
│ └─ idx_to_chord.json # Diccionario índice→acorde
├─ notebooks/ # Notebooks del proyecto (PDFs exportados)
├─ requirements.txt # Dependencias necesarias
└─ README.md

## 🚀 Guía para lanzar la aplicación (UI de Gradio)
### 1. Clonar el repositorio
git clone https://github.com/usuario/armonIA.git
cd armonIA
### 2. Crear un entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
### 3. Instalar dependencias
pip install -r requirements.txt
### 4. Ejecutar la aplicación
cd app
python app.py

Esto abrirá una interfaz interactiva en tu navegador (local) donde podrás:
Construir una secuencia de acordes.
Obtener predicciones del siguiente acorde usando el modelo Transformer.
Explorar las sugerencias Top-k en tiempo real.

## 📑 Notebooks del proyecto
Los notebooks incluyen:
Preprocesamiento → limpieza, tokenización, normalización y codificación de acordes.
Modelo y métricas → entrenamiento del Transformer y evaluación cuantitativa.
Aplicación → integración del modelo con la interfaz Gradio
## ℹ️ Notas
Los datasets originales (chordonomicon.csv, all_features.csv, etc.) no están incluidos en este repositorio por superar el límite de tamaño de GitHub.
Este repositorio permite únicamente usar el modelo entrenado y lanzar la aplicación interactiva.

# 🎵 armonIA: Transformer for Chord Prediction (English)
Repository for my Bachelor’s Thesis (TFG) at UPM.
The project focuses on the design of a tool to support musical composition using sequential neural networks (Transformer vs. LSTM).
It includes preprocessing, training/evaluation notebooks, and an interactive Gradio application.

## 📂 Repository structure
armonIA/
├─ app/
│  └─ app.py              # Gradio user interface
├─ models/
│  ├─ checkpoint_best.pt  # Pretrained Transformer model
│  ├─ checkpoint_last.pt  # Last training checkpoint
│  └─ config.json         # Model configuration
├─ data/
│  ├─ chord_to_idx.json   # Chord→index dictionary
│  └─ idx_to_chord.json   # Index→chord dictionary
├─ notebooks/             # Project notebooks (exported as PDF)
├─ requirements.txt       # Dependencies
└─ README.md


---

## 🚀 Run the application (Gradio UI)
### 1. Clone the repo
git clone https://github.com/usuario/armonIA.git
cd armonIA
### 2. Create a virtual environment (optional but recommended)python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
### 3. Install dependencies
pip install -r requirements.txt
### 4. Run the app
cd app
python app.py

This will launch a local Gradio interface in your browser where you can:
Build a chord sequence.
Predict the next chord with the Transformer model.
Explore Top-k chord suggestions in real time.

## 📑 Project notebooks
The included notebooks document:
Preprocessing → chord cleaning, tokenization, normalization, encoding.
Model & metrics → Transformer training and evaluation.
Application → Gradio-based user interface.

## ℹ️ Notes
Original datasets (chordonomicon.csv, all_features.csv, etc.) are not included due to GitHub size limitations.
This repository is intended only to use the pretrained model and launch the interactive app.
