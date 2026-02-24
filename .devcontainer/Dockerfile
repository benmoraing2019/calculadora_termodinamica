FROM python:3.11-slim

# Evita que Python genere archivos .pyc y fuerza a que la salida de la consola no se bloquee
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /workspace

# Copia el archivo de requerimientos primero (para aprovechar el caché de Docker)
COPY requirements.txt .

# Actualiza pip e instala las dependencias
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt