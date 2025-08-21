# ArmonIA
Herramienta de predicción de acordes con modelo Transformer.

## Estructura del proyecto
- **notebooks/** → Preprocesamiento, entrenamiento y app
- **src/** → Código Python modular (a futuro)
- **app/** → Scripts para lanzar la UI de Gradio
- **data/raw/** → Dataset original (`chordonomicon.csv`)
- **data/processed/** → Archivos procesados (JSON, parquet, CSV)
- **models/** → Checkpoints y config del Transformer

## Ejecución
1. Coloca `chordonomicon.csv` en `data/raw/`
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
