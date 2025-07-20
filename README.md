# Análisis Comparativo de Encodings Posicionales para Transformers en Visión

Este repositorio contiene el código y los resultados de un benchmark comparativo que evalúa el impacto de diferentes estrategias de encoding posicional en el rendimiento, la complejidad y la eficiencia de un modelo de auto-atención para la clasificación de imágenes.

El trabajo se basa en el paper **"On the Relationship between Self-Attention and Convolutional Layers"** de Cordonnier et al. (ICLR 2020), utilizando su implementación de un modelo Transformer para visión como base para los experimentos.

## Resumen del Proyecto

La efectividad de las arquitecturas Transformer en visión depende críticamente de cómo se codifica la información espacial. Este proyecto implementa un benchmark sistemático para comparar cinco estrategias de encoding en un entorno con recursos computacionales controlados (GPU de 8GB VRAM).

Los resultados de nuestros experimentos, realizados sobre un subconjunto del dataset CIFAR-10, revelan una compleja compensación entre la sofisticación del encoding, la facilidad de optimización y el costo computacional, ofreciendo una guía práctica para el desarrollo en escenarios con hardware limitado.

## Características Principales

- **Benchmark de 5 Encodings**: Compara sistemáticamente los métodos de encoding **Absoluto**, **Relativo con Contenido**, **Relativo solo Posición**, **Cuadrático Gaussiano** y un modelo **Sin Encoding**.
- **Métricas Detalladas**: Para cada experimento, se generan y guardan automáticamente:
    - Métricas de rendimiento (Precisión, Pérdida de Test).
    - Métricas de eficiencia (Tiempo de Ejecución, Número de Parámetros).
    - Gráficos comparativos de resumen.
    - Matrices de confusión por cada experimento.
    - Logs detallados para TensorBoard.
- **Entorno Reproducible**: Se proporciona un `Dockerfile` para encapsular el entorno y todas las dependencias, garantizando una replicación perfecta del experimento en cualquier máquina.

## Estructura del Proyecto

```plaintext
attention-cnn/
│
├── benchmark_results/              # Carpeta de resultados de un benchmark de prueba
├── benchmark_results_10_epochs/    # Carpeta con los resultados del informe final
├── data/                           # Datos del dataset CIFAR-10
├── models/                         # Definiciones de las arquitecturas (Transformer, ResNet)
├── runs/                           # Scripts de ejecución originales del paper
├── utils/                          # Scripts auxiliares (config, plotting, logging)
│
├── benchmark_final.py              # NUESTRO SCRIPT PRINCIPAL para ejecutar el benchmark
├── Dockerfile                      # Receta para construir el entorno reproducible
├── README.md                       # Este archivo
└── train.py                        # Script de entrenamiento original del paper


## Despliegue y Ejecución

Existen dos métodos para configurar el entorno y ejecutar el benchmark. El método con Docker es el más recomendado por su simplicidad y reproducibilidad.

### Método 1: Usando Docker (Recomendado)

Este método construye una imagen con todas las dependencias correctas y ejecuta el benchmark en un contenedor aislado.

**Prerrequisitos:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución.
- Para usuarios de GPU NVIDIA: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (generalmente se integra con Docker Desktop).

**Pasos:**

1.  **Construir la imagen de Docker**:
    Abre una terminal en la raíz del proyecto y ejecuta:
    ```bash
    docker build -t atencion-benchmark .
    ```
    *(Este proceso tardará varios minutos la primera vez).*

2.  **Ejecutar el benchmark**:
    Una vez construida la imagen, ejecuta el siguiente comando:
    ```bash
    docker run --gpus all -v "%cd%\benchmark_results_docker:/app/benchmark_results_10_epochs" atencion-benchmark
    ```
    - `--gpus all`: Permite que el contenedor use tu GPU.
    - `-v ...`: Mapea la carpeta de resultados del contenedor a una carpeta en tu PC (`benchmark_results_docker`), asegurando que los resultados se guarden permanentemente.

### Método 2: Configuración Manual con Conda

Este método recrea el entorno que depuramos manualmente.

1.  **Crear el entorno de Conda**:
    ```bash
    conda create -n atencion_env python=3.12 -y
    ```

2.  **Activar el entorno**:
    ```bash
    conda activate atencion_env
    ```

3.  **Instalar dependencias clave**:
    ```bash
    # Instalar PyTorch con soporte para CUDA
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    # Instalar las librerías con versiones específicas que solucionamos
    pip install "protobuf==3.20.3" "Pillow<10.0.0"
    pip install --upgrade tabulate boto3 urllib3
    
    # Instalar el resto de dependencias
    pip install -r requirements.txt
    pip install scikit-learn seaborn matplotlib pyyaml tensorboardX tqdm
    ```

4.  **Ejecutar el benchmark**:
    ```bash
    python benchmark_final.py
    ```

## Resultados del Benchmark

A continuación se presentan los resultados finales obtenidos tras un entrenamiento de 10 épocas sobre un subconjunto de 1000 imágenes de CIFAR-10.

| Tipo de Encoding | Precisión Final | Pérdida de Test | # Parámetros | Tiempo de Ejecución |
| :--- | :--- | :--- | :--- | :--- |
| relativo_aprendido_contenido | 24.80% | 2.013 | 29M | 16351.29 s |
| relativo_aprendido_solo_posicion | 30.00% | 1.9431 | 20.5M | 8644.62 s |
| cuadratico_gaussiano | 28.40% | 1.9303 | 11.9M | 6198.98 s |
| **absoluto** | **32.80%** | **1.9538** | **6.63M** | **15492.13 s** |
| sin_encoding | 27.20% | 2.051 | 6.23M | 17637.15 s |

El análisis completo, la discusión y las conclusiones de estos resultados se encuentran detallados en el informe técnico adjunto.

## Agradecimientos

Este trabajo es una extensión y un análisis basado en el código y los conceptos presentados por Cordonnier, Loukas y Jaggi en su paper "On the Relationship between Self-Attention and Convolutional Layers". Se agradece a los autores por hacer su código público.