const { chromium } = require('playwright-extra');
const stealth = require('playwright-extra-plugin-stealth')();
const axios = require('axios');
const winston = require('winston');
const cron = require('node-cron');
const path = require('path');
require('dotenv').config();

// Налаштування stealth
chromium.use(stealth);

// Налаштування логування
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ timestamp, level, message }) => {
      return `[${timestamp}] ${level.toUpperCase()}: ${message}`;
    })
  ),
  transports: [
    new winston.transports.Console(),
    ...(process.env.LOG_TO_FILE === 'true' ? [
      new winston.transports.File({ 
        filename: path.join(__dirname, 'logs', 'bot.log'),
        maxsize: 10 * 1024 * 1024, // 10MB
        maxFiles: 5
      })
    ] : [])
  ]
});

class BinanceBot {
  constructor() {
    this.browser = null;
    this.page = null;
    this.context = null;
    this.proxyList = [
      process.env.PROXY_1,
      process.env.PROXY_2,
      process.env.PROXY_3,
      process.env.PROXY_4,
      process.env.PROXY_5
    ].filter(Boolean);
    
    this.config = {
      minDelay: parseInt(process.env.MIN_DELAY) || 2000,
      maxDelay: parseInt(process.env.MAX_DELAY) || 5000,
      mouseDelayMin: parseInt(process.env.MOUSE_DELAY_MIN) || 100,
      mouseDelayMax: parseInt(process.env.MOUSE_DELAY_MAX) || 500,
      viewportWidth: parseInt(process.env.VIEWPORT_WIDTH) || 1920,
      viewportHeight: parseInt(process.env.VIEWPORT_HEIGHT) || 1080,
      headless: process.env.HEADLESS === 'true',
      devtools: process.env.DEVTOOLS === 'true'
    };

    // Створення папки для логів
    this.ensureLogsDirectory();
  }

  // Створення директорії для логів
  ensureLogsDirectory() {
    const fs = require('fs');
    const logsDir = path.join(__dirname, 'logs');
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
  }

  // Генерація випадкової затримки
  randomDelay(min = this.config.minDelay, max = this.config.maxDelay) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  // Вибір випадкового проксі
  getRandomProxy() {
    if (this.proxyList.length === 0) {
      logger.warn('Проксі не налаштовані, використовуємо прямий зв\'язок');
      return null;
    }
    const proxy = this.proxyList[Math.floor(Math.random() * this.proxyList.length)];
    logger.info(`Вибрано проксі: ${proxy.replace(/\/\/.*@/, '//***@')}`);
    return proxy;
  }

  // Генерація реалістичного User-Agent
  getRandomUserAgent() {
    const userAgents = [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
    ];
    return userAgents[Math.floor(Math.random() * userAgents.length)];
  }

  // Інтеграція з існуючим Python кодом
  async integrateWithPythonBot() {
    try {
      // Можна читати конфігурацію з Python проекту
      const fs = require('fs');
      const pythonConfigPath = path.join(__dirname, '..', 'config.json');
      
      if (fs.existsSync(pythonConfigPath)) {
        const pythonConfig = JSON.parse(fs.readFileSync(pythonConfigPath, 'utf8'));
        logger.info('Завантажено конфігурацію з Python проекту');
        return pythonConfig;
      }
    } catch (error) {
      logger.debug(`Не вдалося завантажити Python конфігурацію: ${error.message}`);
    }
    return null;
  }

  // Імітація людського руху миші
  async humanMouseMove(page) {
    try {
      const moves = Math.floor(Math.random() * 3) + 2;
      logger.debug(`Виконуємо ${moves} рухів миші`);
      
      for (let i = 0; i < moves; i++) {
        const x = Math.floor(Math.random() * (this.config.viewportWidth - 200)) + 100;
        const y = Math.floor(Math.random() * (this.config.viewportHeight - 200)) + 100;
        
        await page.mouse.move(x, y, {
          steps: Math.floor(Math.random() * 5) + 1
        });
        
        await page.waitForTimeout(this.randomDelay(
          this.config.mouseDelayMin, 
          this.config.mouseDelayMax
        ));
      }
    } catch (error) {
      logger.error(`Помилка руху миші: ${error.message}`);
    }
  }

  // Імітація людського скролінгу
  async humanScroll(page) {
    try {
      const scrolls = Math.floor(Math.random() * 3) + 1;
      logger.debug(`Виконуємо ${scrolls} скролів`);
      
      for (let i = 0; i < scrolls; i++) {
        const scrollDelta = Math.floor(Math.random() * 300) + 100;
        await page.mouse.wheel(0, scrollDelta);
        await page.waitForTimeout(this.randomDelay(500, 1500));
      }
    } catch (error) {
      logger.error(`Помилка скролінгу: ${error.message}`);
    }
  }

  // Ініціалізація браузера
  async initBrowser() {
    try {
      const proxy = this.getRandomProxy();
      
      const browserOptions = {
        headless: this.config.headless,
        devtools: this.config.devtools,
        args: [
          '--no-sandbox',
          '--disable-setuid-sandbox',
          '--disable-dev-shm-usage',
          '--disable-accelerated-2d-canvas',
          '--no-first-run',
          '--no-zygote',
          '--disable-gpu',
          '--disable-blink-features=AutomationControlled',
          '--disable-features=VizDisplayCompositor'
        ]
      };

      // Налаштування контексту
      const contextOptions = {
        viewport: { 
          width: this.config.viewportWidth, 
          height: this.config.viewportHeight 
        },
        userAgent: this.getRandomUserAgent(),
        locale: 'en-US',
        timezoneId: 'America/New_York',
        permissions: ['geolocation'],
        geolocation: { latitude: 40.7128, longitude: -74.0060 } // Нью-Йорк
      };

      // Додавання проксі якщо доступний
      if (proxy) {
        const proxyUrl = new URL(proxy);
        contextOptions.proxy = {
          server: `${proxyUrl.protocol}//${proxyUrl.host}`,
          username: proxyUrl.username || process.env.PROXY_USERNAME,
          password: proxyUrl.password || process.env.PROXY_PASSWORD
        };
      }

      this.browser = await chromium.launch(browserOptions);
      this.context = await this.browser.newContext(contextOptions);
      
      // Додаткові налаштування для маскування автоматизації
      await this.context.addInitScript(() => {
        Object.defineProperty(navigator, 'webdriver', {
          get: () => undefined,
        });
        
        Object.defineProperty(navigator, 'plugins', {
          get: () => [1, 2, 3, 4, 5],
        });
        
        Object.defineProperty(navigator, 'languages', {
          get: () => ['en-US', 'en'],
        });
        
        window.chrome = {
          runtime: {},
        };
      });
      
      this.page = await this.context.newPage();
      
      logger.info(`Браузер ініціалізовано${proxy ? ` з проксі` : ''}`);
      return true;
      
    } catch (error) {
      logger.error(`Помилка ініціалізації браузера: ${error.message}`);
      return false;
    }
  }

  // Перевірка на капчу або блокування
  async checkForBlocking() {
    try {
      const captchaSelectors = [
        '[data-testid="captcha"]',
        '.captcha',
        '#captcha',
        '[class*="captcha"]',
        '[id*="captcha"]',
        '.grecaptcha-badge',
        '#cf-challenge-running'
      ];

      for (const selector of captchaSelectors) {
        const element = await this.page.$(selector);
        if (element) {
          logger.warn(`Виявлена капча або блокування: ${selector}`);
          return true;
        }
      }

      // Перевірка на блокування по тексту
      const blockingTexts = [
        'Access denied',
        'Blocked',
        'Security check',
        'Please verify',
        'Captcha',
        'Robot verification'
      ];

      const pageContent = await this.page.textContent('body');
      for (const text of blockingTexts) {
        if (pageContent && pageContent.toLowerCase().includes(text.toLowerCase())) {
          logger.warn(`Виявлено блокування по тексту: ${text}`);
          return true;
        }
      }

      return false;
    } catch (error) {
      logger.error(`Помилка перевірки блокування: ${error.message}`);
      return false;
    }
  }

  // Основна логіка роботи з Binance
  async runBinanceAutomation() {
    try {
      logger.info('Початок автоматизації Binance');
      
      // Інтеграція з Python конфігурацією
      const pythonConfig = await this.integrateWithPythonBot();
      
      // Перехід на головну сторінку Binance
      await this.page.goto('https://www.binance.com/en', {
        waitUntil: 'networkidle',
        timeout: 30000
      });

      logger.info('Завантажена головна сторінка Binance');
      
      // Затримка для завантаження
      await this.page.waitForTimeout(this.randomDelay(3000, 6000));
      
      // Перевірка на блокування
      if (await this.checkForBlocking()) {
        logger.error('Виявлено блокування або капчу');
        await this.sendNotification('🚫 Виявлено блокування на Binance');
        return false;
      }

      // Імітація людської поведінки
      await this.humanMouseMove(this.page);
      await this.humanScroll(this.page);

      // Перехід до сторінки конвертації
      logger.info('Переходимо на сторінку конвертації');
      await this.page.goto('https://www.binance.com/en/convert', {
        waitUntil: 'networkidle',
        timeout: 30000
      });

      await this.page.waitForTimeout(this.randomDelay(2000, 4000));

      // Перевірка необхідності входу
      const loginButton = await this.page.$('text="Log In"');
      if (loginButton) {
        logger.info('Потрібен вхід в акаунт');
        const loginSuccess = await this.performLogin();
        if (!loginSuccess) {
          return false;
        }
      }

      // Виконання торгових дій
      const tradeSuccess = await this.performTradeActions();
      
      if (tradeSuccess) {
        logger.info('Автоматизація завершена успішно');
        await this.sendNotification('✅ Автоматизація Binance завершена успішно');
        await this.saveResults();
        return true;
      } else {
        logger.warn('Автоматизація завершена з помилками');
        return false;
      }

    } catch (error) {
      logger.error(`Помилка під час автоматизації: ${error.message}`);
      await this.sendNotification(`❌ Помилка автоматизації: ${error.message}`);
      return false;
    }
  }

  // Виконання входу (якщо потрібно)
  async performLogin() {
    try {
      logger.info('Спроба автоматичного входу');
      
      // Натискання кнопки Log In
      await this.page.click('text="Log In"');
      await this.page.waitForTimeout(this.randomDelay(2000, 3000));

      // Введення email
      const emailInput = await this.page.$('input[type="email"], input[name="email"]');
      if (emailInput && process.env.BINANCE_EMAIL) {
        await this.humanMouseMove(this.page);
        await emailInput.click();
        await this.page.waitForTimeout(this.randomDelay(500, 1000));
        await emailInput.type(process.env.BINANCE_EMAIL, { delay: 100 });
        await this.page.waitForTimeout(this.randomDelay(1000, 2000));
      }

      // Введення пароля
      const passwordInput = await this.page.$('input[type="password"]');
      if (passwordInput && process.env.BINANCE_PASSWORD) {
        await passwordInput.click();
        await this.page.waitForTimeout(this.randomDelay(500, 1000));
        await passwordInput.type(process.env.BINANCE_PASSWORD, { delay: 100 });
        await this.page.waitForTimeout(this.randomDelay(1000, 2000));
      }

      // Натискання кнопки входу
      const submitButton = await this.page.$('button[type="submit"], button:has-text("Log In")');
      if (submitButton) {
        await submitButton.click();
        await this.page.waitForTimeout(this.randomDelay(3000, 5000));
      }

      // Перевірка успішного входу
      await this.page.waitForSelector('[data-testid="header-account-menu"], .account-menu', {
        timeout: 10000
      });

      logger.info('Вхід виконано успішно');
      return true;

    } catch (error) {
      logger.error(`Помилка входу: ${error.message}`);
      return false;
    }
  }

  // Виконання торгових дій
  async performTradeActions() {
    try {
      logger.info('Виконання торгових дій');

      // Очікування завантаження сторінки конвертації
      await this.page.waitForSelector('[data-testid="from-input"], .from-input', {
        timeout: 15000
      });

      // Імітація людської поведінки
      await this.humanMouseMove(this.page);
      await this.page.waitForTimeout(this.randomDelay(2000, 4000));

      // Вибір валюти FROM
      if (process.env.FROM_CURRENCY) {
        const fromCurrencyButton = await this.page.$('[data-testid="from-currency"], .from-currency');
        if (fromCurrencyButton) {
          await fromCurrencyButton.click();
          await this.page.waitForTimeout(this.randomDelay(1000, 2000));
          
          // Пошук валюти
          const searchInput = await this.page.$('input[placeholder*="Search"]');
          if (searchInput) {
            await searchInput.type(process.env.FROM_CURRENCY, { delay: 100 });
            await this.page.waitForTimeout(this.randomDelay(1000, 2000));
            
            // Вибір першого результату
            const firstResult = await this.page.$('.currency-option:first-child, [data-testid="currency-option"]:first-child');
            if (firstResult) {
              await firstResult.click();
            }
          }
        }
      }

      // Введення суми
      if (process.env.TRADE_AMOUNT) {
        const amountInput = await this.page.$('[data-testid="from-input"], .amount-input input');
        if (amountInput) {
          await this.humanMouseMove(this.page);
          await amountInput.click();
          await this.page.waitForTimeout(this.randomDelay(500, 1000));
          await amountInput.fill(process.env.TRADE_AMOUNT);
          await this.page.waitForTimeout(this.randomDelay(1000, 2000));
        }
      }

      // Вибір валюти TO
      if (process.env.TO_CURRENCY) {
        const toCurrencyButton = await this.page.$('[data-testid="to-currency"], .to-currency');
        if (toCurrencyButton) {
          await toCurrencyButton.click();
          await this.page.waitForTimeout(this.randomDelay(1000, 2000));
          
          const searchInput = await this.page.$('input[placeholder*="Search"]');
          if (searchInput) {
            await searchInput.type(process.env.TO_CURRENCY, { delay: 100 });
            await this.page.waitForTimeout(this.randomDelay(1000, 2000));
            
            const firstResult = await this.page.$('.currency-option:first-child, [data-testid="currency-option"]:first-child');
            if (firstResult) {
              await firstResult.click();
            }
          }
        }
      }

      // Очікування розрахунку конвертації
      await this.page.waitForTimeout(this.randomDelay(3000, 5000));

      logger.info('Торгові дії виконано (без фактичної конвертації)');
      
      // УВАГА: Тут НЕ виконується фактична конвертація для безпеки
      // Розкоментуйте наступні рядки тільки після ретельного тестування
      /*
      const convertButton = await this.page.$('[data-testid="convert-button"], button:has-text("Convert")');
      if (convertButton) {
        await convertButton.click();
        await this.page.waitForTimeout(this.randomDelay(2000, 3000));
        
        // Підтвердження конвертації
        const confirmButton = await this.page.$('button:has-text("Confirm")');
        if (confirmButton) {
          await confirmButton.click();
        }
      }
      */

      return true;

    } catch (error) {
      logger.error(`Помилка під час торгових дій: ${error.message}`);
      return false;
    }
  }

  // Збереження результатів для інтеграції з Python
  async saveResults() {
    try {
      const fs = require('fs');
      const resultsPath = path.join(__dirname, 'data', 'last_execution.json');
      
      // Створити папку data якщо не існує
      const dataDir = path.dirname(resultsPath);
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
      }

      const results = {
        timestamp: new Date().toISOString(),
        success: true,
        fromCurrency: process.env.FROM_CURRENCY,
        toCurrency: process.env.TO_CURRENCY,
        amount: process.env.TRADE_AMOUNT,
        executionTime: Date.now()
      };

      fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
      logger.info('Результати збережено для Python інтеграції');
    } catch (error) {
      logger.error(`Помилка збереження результатів: ${error.message}`);
    }
  }

  // Відправка сповіщень
  async sendNotification(message) {
    try {
      if (process.env.ENABLE_NOTIFICATIONS !== 'true' || !process.env.WEBHOOK_URL) {
        return;
      }

      await axios.post(process.env.WEBHOOK_URL, {
        text: `🤖 Binance Browser Bot: ${message}`,
        timestamp: new Date().toISOString()
      });

      logger.info('Сповіщення відправлено');
    } catch (error) {
      logger.error(`Помилка відправки сповіщення: ${error.message}`);
    }
  }

  // Закриття браузера
  async close() {
    try {
      if (this.browser) {
        await this.browser.close();
        logger.info('Браузер закрито');
      }
    } catch (error) {
      logger.error(`Помилка закриття браузера: ${error.message}`);
    }
  }

  // Обробка сигналів завершення
  setupGracefulShutdown() {
    const shutdown = async (signal) => {
      logger.info(`Отримано сигнал ${signal}, завершення роботи...`);
      await this.close();
      process.exit(0);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
  }
}

// Головна функція
async function main() {
  const bot = new BinanceBot();
  bot.setupGracefulShutdown();
  
  try {
    const initSuccess = await bot.initBrowser();
    if (!initSuccess) {
      logger.error('Не вдалося ініціалізувати браузер');
      return false;
    }

    const success = await bot.runBinanceAutomation();
    
    if (success) {
      logger.info('Автоматизація завершена успішно');
    } else {
      logger.warn('Автоматизація завершена з помилками');
    }
    
    return success;
    
  } catch (error) {
    logger.error(`Критична помилка: ${error.message}`);
    return false;
  } finally {
    await bot.close();
  }
}

// Планування виконання
function scheduleBot() {
  const interval = parseInt(process.env.RUN_INTERVAL) || 30;
  logger.info(`Планування запуску кожні ${interval} хвилин`);
  
  // Запуск кожні N хвилин
  cron.schedule(`*/${interval} * * * *`, async () => {
    logger.info('🕐 Запуск запланованої автоматизації...');
    await main();
  });

  // Перший запуск одразу
  main();
}

// Точка входу
if (require.main === module) {
  if (process.env.SCHEDULED === 'true') {
    scheduleBot();
  } else {
    main();
  }
}

module.exports = { BinanceBot, main };