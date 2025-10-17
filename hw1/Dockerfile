FROM python:3.11-slim

WORKDIR /data

# Создание директории для логов
RUN mkdir -p /data/logs && \
    touch /data/logs/service.log && \
    chmod -R 777 /data/logs  # Права на запись для всех пользователей

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Точки монтирования
VOLUME /data/input
VOLUME /data/output

CMD ["python", "app.py"]