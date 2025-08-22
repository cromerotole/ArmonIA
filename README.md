# 🎵 armonIA: Transformer for Chord Prediction

Repositorio del Trabajo de Fin de Grado (TFG) en la UPM.  
El proyecto consiste en el **diseño de una herramienta de apoyo a la composición musical** mediante redes neuronales secuenciales (Transformer vs. LSTM).  
Incluye notebooks de preprocesamiento, entrenamiento y métricas, y una aplicación interactiva con **Gradio**.

---

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

---

## 🚀 Lanzar la aplicación (UI de Gradio)

### 1. Clonar el repositorio
```bash
git clone https://github.com/usuario/armonIA.git
cd armonIA
```

### 2. Crear un entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows


