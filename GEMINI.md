# AGS-S 201 "Discovery" - Hybrid Optimization Framework

## 🎯 Project Overview
Este repositorio contiene el framework de optimización híbrida para la calibración del **Modelo de Davies**, aplicado al estudio del estallido social chileno de 2019. El sistema integra ecuaciones diferenciales en tiempo discreto con algoritmos de optimización global y local para modelar la interacción entre manifestantes y fuerzas policiales.

### Key Technologies
- **Simulation Engine**: PyTorch (aceleración por GPU A100/L4).
- **Optimization**: Híbrido (Global: DE/PSO, Local: L-BFGS-B).
- **Data Processing**: SciPy, Statsmodels (LOWESS), Scikit-learn (Scaling).
- **Analysis**: Kolmogorov-Smirnov, MAE, RMSE.

---

## 🚀 Building and Running

### Prerequisites
El proyecto utiliza `uv` para la gestión de entornos y dependencias.
```bash
# Crear entorno e instalar dependencias
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Execution Protocols
El punto de entrada principal es `src/main.py`. El sistema utiliza `rich` para reportes estructurados en consola.

**1. Calibración con Differential Evolution (DE):**
```bash
python src/main.py --optimizer de --max_iter 50 --pop_size 15 --seed 42 --save --plot
```

**2. Calibración con Particle Swarm Optimization (PSO):**
```bash
python src/main.py --optimizer pso --max_iter 100 --pop_size 20 --seed 42 --save --plot
```

**3. Linting con Ruff:**
```bash
ruff check src/
```

**4. Benchmark de Rendimiento:**
```bash
python src/main.py --benchmark
```

**5. Flags Disponibles:**
- `--data_dir`: Ruta a los archivos `.dat` y `.npy` (default: `data`).
- `--method`: Método de suavizado (`linear` o `lowess`).
- `--Nt` / `--Ntt`: Parámetros de pasos de tiempo de la simulación.
- `--seed`: Semilla aleatoria para reproducibilidad (default: 42).
- `--save`: Flag para persistir resultados en disco (JSON y NPY).
- `--output_dir`: Directorio para guardar resultados (default: `results`).

---

## 📂 Directory Structure & Roles

- **`src/engine.py`**: Núcleo matemático del Modelo de Davies. Implementa la lógica de atracción, difusión y captura en PyTorch.
- **`src/optimizers.py`**: Implementación de `DifferentialEvolutionOptimizer` (scipy wrapper) y `ParticleSwarmOptimizer` (custom PyTorch).
- **`src/preprocessing.py`**: Pipeline de limpieza de datos SOSAFE y generación de targets.
- **`src/main.py`**: Orquestador de la calibración y evaluación de métricas.
- **`data/`**: Contiene densidades de origen/destino, targets de SOSAFE y matrices de distancias (`rij`).
- **`notebooks/`**: Historial de experimentos y validaciones estadísticas.

---

## 🛠️ Development Conventions

### Coding Standards
- **Tensor Ops**: Priorizar operaciones vectorizadas en PyTorch para mantener el speedup de 112x. Evitar loops explícitos de Python dentro de `run_simulation`.
- **Device Agnostic**: Asegurar que los tensores se muevan al dispositivo correcto (`cuda` o `cpu`) usando el parámetro `device` en las clases.
- **Reproducibilidad**: Mantener las semillas aleatorias si se realizan comparaciones de optimizadores.

### Testing
Existen tests unitarios en la carpeta `tests/` para validar el preprocesamiento y utilitarios.
```bash
pytest tests/
```

### Contribution Guidelines
1. Las nuevas heurísticas de optimización deben implementarse en `src/optimizers.py` siguiendo la interfaz de clase establecida.
2. Cualquier modificación en el motor físico (`engine.py`) debe ser validada contra los resultados de los notebooks de referencia.

---

## 📡 System Status (2026)
*Discovery operando a capacidad nominal. Los modelos de calibración están listos para ser adaptados a desafíos de movilidad urbana y logística industrial.*
