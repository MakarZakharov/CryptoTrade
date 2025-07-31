# Deployment на Render.com

## 🚀 Швидке розгортання браузерного бота

### Крок 1: Підготовка репозиторію

1. **Створіть новий GitHub репозиторій** тільки для браузерного бота:
   ```
   binance-browser-automation/
   ├── package.json
   ├── bot.js
   ├── start.js
   └── .env.example
   ```

2. **Завантажте тільки папку browser-automation**:
   ```bash
   # Створіть новий репозиторій
   git init
   git add .
   git commit -m "Initial browser bot setup"
   git remote add origin https://github.com/username/binance-browser-bot.git
   git push -u origin main
   ```

### Крок 2: Налаштування Render.com

1. **Зайдіть на render.com** → **New** → **Web Service**

2. **Підключіть GitHub** та оберіть репозиторій

3. **Налаштування сервісу**:
   ```
   Name: binance-browser-bot
   Environment: Node
   Branch: main
   Build Command: npm install && npx playwright install chromium
   Start Command: npm start
   ```

4. **Environment Variables** (додайте через Render dashboard):
   ```
   NODE_ENV=production
   HEADLESS=true
   
   # Ваші проксі
   PROXY_1=http://user:pass@proxy1.com:8080
   PROXY_2=http://user:pass@proxy2.com:8080
   
   # Binance облікові дані
   BINANCE_EMAIL=your_email@example.com
   BINANCE_PASSWORD=your_password
   
   # Торгові налаштування
   FROM_CURRENCY=USDT
   TO_CURRENCY=BTC
   TRADE_AMOUNT=10
   
   # Затримки
   MIN_DELAY=3000
   MAX_DELAY=7000
   RUN_INTERVAL=30
   ```

### Крок 3: Додаткові налаштування

1. **План**: Start з **Free tier** для тестування
2. **Auto-Deploy**: Увімкнено
3. **Health Checks**: Автоматично

## 🐳 Альтернатива: Docker на Render

1. **Створіть Dockerfile** в корені репозиторію:
   ```dockerfile
   FROM node:18-slim
   
   # Playwright dependencies
   RUN apt-get update && apt-get install -y \
       libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
       libgbm1 libxss1 libasound2 && rm -rf /var/lib/apt/lists/*
   
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci --only=production
   RUN npx playwright install chromium
   
   COPY . .
   CMD ["npm", "start"]
   ```

2. **Render налаштування**:
   - Environment: **Docker**
   - Dockerfile Path: **Dockerfile**

## 🆘 Якщо все ще є проблеми з Python

### Варіант 1: Розділіть проекти

**Створіть окремі репозиторії**:
- `crypto-trading-python` - ваш основний Python код
- `binance-browser-bot` - тільки браузерна автоматизація

### Варіант 2: Виключіть TA-Lib з requirements.txt

В головному Python проекті:
```txt
# requirements.txt (без TA-Lib)
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
requests>=2.25.0
# TA-Lib>=0.4.28  # Закоментовано для cloud deployment
```

Використайте альтернативи:
```bash
pip install pandas-ta  # Альтернатива TA-Lib
```

## 🎯 Рекомендований workflow

1. **Локальна розробка**: Повний проект з TA-Lib
2. **Cloud deployment**: Тільки браузерний бот (Node.js)
3. **Інтеграція**: JSON файли та API між сервісами

Ваш браузерний бот буде працювати незалежно від Python проблем! 🚀