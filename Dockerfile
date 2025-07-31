# Використовуємо офіційний Python образ
FROM python:3.11-slim

# Встановлюємо системні залежності і TA-Lib
RUN apt-get update && \
    apt-get install -y build-essential wget gcc make && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib* && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Створюємо робочу директорію
WORKDIR /app

# Копіюємо файли проєкту
COPY . .

# Встановлюємо Python залежності
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Запускаємо програму
CMD ["python", "main.py"]
