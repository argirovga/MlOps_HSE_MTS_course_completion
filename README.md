# HW's by Argirov Georgy

В данном репозитории можно найти все дз выполненые мною на курсе MLops

---

# HW 1️⃣

## 📂 Структура дз

```bash
hw1/
│
├── data/
│   ├── logs/                   # логи работы
│   ├── input/                  # входные CSV-файлы для скоринга
│   ├── output/                 # результаты предсказаний
│   └── model_weights/          # веса модели
│       └──best_hist_gb_model.pkl
├── model_training/
│   └── simple_training.ipynb   # быстрые экспериментами  
├── src/
│   └── model.py                # обертка над моделью
├── app.py                      # основной сервис
├── Dockerfile                  # docker-образ
└── requirements.txt            # реки
```

---

## Запуск

### 1. Подготовка окружения

Перейдите в директорию дз1:

```bash
cd hw1
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

---

### 2. Запуск через Docker

Соберите docker-образ:

```bash
docker build -t hw1-ml-service .
```

Запустите контейнер:

```bash
docker run -d \
  -v $(pwd)/data/input:/data/input \
  -v $(pwd)/data/output:/data/output \
  -v $(pwd)/data/logs:/data/logs \
  -v $(pwd)/data/model_weights:/data/model_weights \
  --name hw1_service \
  hw1-ml-service
```

Сервис начнет мониторить папку `data/input` на появление новых файлов.

---

### 3. Использование

Чтобы запустить скоринг, просто положите CSV-файл в директорию:

```
hw1/data/input/
```

Сервис автоматически:
- обнаружит новый файл,
- прогонит его через модель,
- сохранит результат в `hw1/data/output/`  
  под именем  
  `predictions_<timestamp>_<filename>.csv`.

---

### 4. Логи

Все логи работы сервиса сохраняются в:

```
hw1/data/logs/service.log
```

Посмотреть в реальном времени:

```bash
tail -f hw1/data/logs/service.log
```

---

### 5. Формат входного CSV

Ожидаемые колонки:

```
amount, lat, lon, population_city, cat_id, us_state
```

---

### 6. Остановка контейнера

```bash
docker stop hw1_service
docker rm hw1_service
```

---

### 7. Альтернатива — запуск без Docker

```bash
cd hw1
python app.py
```

После запуска можно также просто добавлять CSV-файлы в `data/input`, и сервис будет их автоматически обрабатывать.

---