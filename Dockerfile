FROM python:3.11-slim

# Встановлюємо необхідні пакети
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Копіюємо проєкт
WORKDIR /app
COPY . .

# Встановлюємо Python-залежності
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
