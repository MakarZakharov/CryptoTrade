#!/usr/bin/env node

/**
 * Стартовий скрипт для Binance Browser Bot
 * Інтегрований з Python CryptoTrade проектом
 */

const { program } = require('commander');
const readline = require('readline');
const { BinanceBot, main } = require('./bot');
const winston = require('winston');
const fs = require('fs');
const path = require('path');

// Налаштування логування
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.colorize(),
    winston.format.printf(({ timestamp, level, message }) => {
      return `[${timestamp}] ${level}: ${message}`;
    })
  ),
  transports: [new winston.transports.Console()]
});

class BotStarter {
  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    this.projectRoot = path.resolve(__dirname, '../../../..');
  }

  async askQuestion(question) {
    return new Promise((resolve) => {
      this.rl.question(question, (answer) => {
        resolve(answer.trim());
      });
    });
  }

  async checkEnvironment() {
    logger.info('🔍 Перевірка середовища браузерного бота...');
    
    // Перевірка .env файлу
    const envPath = path.join(__dirname, '.env');
    if (!fs.existsSync(envPath)) {
      logger.warn('⚠️  Файл .env не знайдено');
      const createEnv = await this.askQuestion('Створити .env файл з прикладу? (y/n): ');
      
      if (createEnv.toLowerCase() === 'y') {
        const examplePath = path.join(__dirname, '.env.example');
        if (fs.existsSync(examplePath)) {
          fs.copyFileSync(examplePath, envPath);
          logger.info('✅ Файл .env створено з прикладу');
          logger.info('⚠️  Будь ласка, відредагуйте .env файл з вашими налаштуваннями');
          return false;
        }
      }
    }

    // Перевірка node_modules
    const nodeModulesPath = path.join(__dirname, 'node_modules');
    if (!fs.existsSync(nodeModulesPath)) {
      logger.error('❌ Залежності не встановлені');
      logger.info('🔧 Запустіть: npm install');
      return false;
    }

    // Перевірка інтеграції з Python проектом
    const pythonSignalPath = path.join(__dirname, '..', 'binance_signal.py');
    if (fs.existsSync(pythonSignalPath)) {
      logger.info('✅ Python Binance Signal знайдено');
    } else {
      logger.warn('⚠️  Python Binance Signal не знайдено в очікуваному місці');
    }

    // Перевірка важливих змінних середовища
    require('dotenv').config({ path: envPath });
    
    const warnings = [];
    
    if (!process.env.PROXY_1 && !process.env.PROXY_2) {
      warnings.push('Проксі не налаштовані (можливе блокування)');
    }
    
    if (!process.env.BINANCE_EMAIL || !process.env.BINANCE_PASSWORD) {
      warnings.push('Облікові дані Binance не налаштовані');
    }

    if (warnings.length > 0) {
      logger.warn('⚠️  Попередження:');
      warnings.forEach(warning => logger.warn(`   - ${warning}`));
    }

    logger.info('✅ Середовище браузерного бота готове');
    return true;
  }

  async checkPythonIntegration() {
    logger.info('🐍 Перевірка інтеграції з Python проектом...');
    
    try {
      // Пошук конфігураційних файлів Python проекту
      const possibleConfigs = [
        path.join(this.projectRoot, 'config.json'),
        path.join(__dirname, '..', 'config.json'),
        path.join(__dirname, '..', '..', 'config.json')
      ];

      for (const configPath of possibleConfigs) {
        if (fs.existsSync(configPath)) {
          logger.info(`✅ Знайдено конфіг Python: ${configPath}`);
          return true;
        }
      }

      logger.warn('⚠️  Конфігурація Python проекту не знайдена');
      return false;
    } catch (error) {
      logger.error(`❌ Помилка перевірки Python інтеграції: ${error.message}`);
      return false;
    }
  }

  async showMainMenu() {
    console.log('\n🤖 Binance Browser Bot (інтегрований з CryptoTrade)');
    console.log('=====================================================');
    console.log('1. Запустити браузерний бот (одноразово)');
    console.log('2. Запустити браузерний бот (заплановано)');
    console.log('3. Тестування браузерної системи');
    console.log('4. Перевірка конфігурації');
    console.log('5. Демо режим (без торгівлі)');
    console.log('6. Інтеграція з Python сигналами');
    console.log('7. Моніторинг логів');
    console.log('0. Вихід');
    console.log('=====================================================');
    
    const choice = await this.askQuestion('Оберіть опцію (0-7): ');
    return choice;
  }

  async runSingleBot() {
    logger.info('🚀 Запуск браузерного бота (одноразово)');
    
    const confirm = await this.askQuestion('⚠️  Це запустить реальний браузерний бот. Продовжити? (y/n): ');
    if (confirm.toLowerCase() !== 'y') {
      logger.info('❌ Скасовано користувачем');
      return;
    }

    try {
      const success = await main();
      if (success) {
        logger.info('✅ Браузерний бот завершив роботу успішно');
      } else {
        logger.warn('⚠️  Браузерний бот завершив роботу з помилками');
      }
    } catch (error) {
      logger.error(`❌ Помилка виконання: ${error.message}`);
    }
  }

  async runScheduledBot() {
    logger.info('⏰ Запуск запланованого браузерного бота');
    
    const interval = await this.askQuestion('Інтервал запуску (хвилини, за замовчуванням 30): ');
    const intervalMinutes = parseInt(interval) || 30;
    
    process.env.SCHEDULED = 'true';
    process.env.RUN_INTERVAL = intervalMinutes.toString();
    
    logger.info(`🕐 Браузерний бот буде запускатися кожні ${intervalMinutes} хвилин`);
    logger.info('💡 Натисніть Ctrl+C для зупинки');
    
    // Імпорт та запуск запланованого режиму
    require('./bot');
  }

  async runTests() {
    logger.info('🧪 Запуск тестування браузерного бота');
    
    try {
      // Простий тест без повного test.js файлу
      const bot = new BinanceBot();
      
      logger.info('Тест 1: Ініціалізація браузера...');
      const initSuccess = await bot.initBrowser();
      if (initSuccess) {
        logger.info('✅ Браузер ініціалізовано успішно');
        await bot.close();
      } else {
        logger.error('❌ Помилка ініціалізації браузера');
      }
      
    } catch (error) {
      logger.error(`❌ Тестування провалено: ${error.message}`);
    }
  }

  async checkConfiguration() {
    logger.info('⚙️  Перевірка конфігурації браузерного бота');
    
    require('dotenv').config({ path: path.join(__dirname, '.env') });
    
    const configs = [
      { name: 'Проксі', check: () => process.env.PROXY_1 || process.env.PROXY_2 },
      { name: 'Email Binance', check: () => process.env.BINANCE_EMAIL },
      { name: 'Пароль Binance', check: () => process.env.BINANCE_PASSWORD },
      { name: 'Валюта FROM', check: () => process.env.FROM_CURRENCY },
      { name: 'Валюта TO', check: () => process.env.TO_CURRENCY },
      { name: 'Сума торгівлі', check: () => process.env.TRADE_AMOUNT },
      { name: 'Затримки', check: () => process.env.MIN_DELAY && process.env.MAX_DELAY },
      { name: 'Розмір вікна', check: () => process.env.VIEWPORT_WIDTH && process.env.VIEWPORT_HEIGHT }
    ];

    console.log('\n📋 Стан конфігурації браузерного бота:');
    configs.forEach(config => {
      const status = config.check() ? '✅' : '❌';
      console.log(`${status} ${config.name}`);
    });

    const envPath = path.join(__dirname, '.env');
    if (fs.existsSync(envPath)) {
      const stats = fs.statSync(envPath);
      console.log(`\n📁 Файл .env: ${stats.size} байт, оновлено ${stats.mtime.toLocaleString()}`);
    }

    // Перевірка Python інтеграції
    await this.checkPythonIntegration();
  }

  async runDemoMode() {
    logger.info('🎭 Запуск демо режиму браузерного бота');
    
    // Тимчасово відключаємо реальну торгівлю
    const originalAmount = process.env.TRADE_AMOUNT;
    process.env.TRADE_AMOUNT = '0';
    
    logger.info('⚠️  Демо режим: торгівля відключена');
    
    try {
      const success = await main();
      if (success) {
        logger.info('✅ Демо режим завершено успішно');
      }
    } catch (error) {
      logger.error(`❌ Помилка демо режиму: ${error.message}`);
    } finally {
      // Відновлюємо оригінальні налаштування
      if (originalAmount) {
        process.env.TRADE_AMOUNT = originalAmount;
      }
    }
  }

  async integratePythonSignals() {
    logger.info('🐍 Інтеграція з Python сигналами');
    
    // Пошук файлу Python сигналів
    const pythonSignalPath = path.join(__dirname, '..', 'binance_signal.py');
    
    if (!fs.existsSync(pythonSignalPath)) {
      logger.error('❌ binance_signal.py не знайдено');
      return;
    }

    logger.info('✅ binance_signal.py знайдено');
    
    // Створення мосту між Python та Node.js
    const bridgeConfig = {
      pythonSignalPath,
      browserBotPath: __dirname,
      lastUpdate: new Date().toISOString(),
      integration: {
        enabled: true,
        syncData: true,
        sharedConfig: true
      }
    };

    const configPath = path.join(__dirname, 'python_integration.json');
    fs.writeFileSync(configPath, JSON.stringify(bridgeConfig, null, 2));
    
    logger.info(`✅ Конфіг інтеграції створено: ${configPath}`);
    
    // Пояснення використання
    console.log('\n📋 Інтеграція налаштована:');
    console.log('1. Python сигнали можуть читати: python_integration.json');
    console.log('2. Браузерний бот зберігає результати в: data/last_execution.json');
    console.log('3. Спільні налаштування через: .env файл');
  }

  async monitorLogs() {
    const logPath = path.join(__dirname, 'logs', 'bot.log');
    
    if (!fs.existsSync(logPath)) {
      logger.warn('📝 Файл логів браузерного бота не знайдено');
      
      // Перевірка логів Python бота
      const pythonLogPath = path.join(__dirname, '..', 'logs');
      if (fs.existsSync(pythonLogPath)) {
        logger.info('📊 Знайдено логи Python бота');
        console.log(`Python логи: ${pythonLogPath}`);
      }
      return;
    }

    logger.info('📊 Моніторинг логів браузерного бота (натисніть Ctrl+C для виходу)');
    
    // Показати останні 20 рядків
    try {
      const content = fs.readFileSync(logPath, 'utf8');
      const lines = content.split('\n').slice(-20);
      console.log('\n📝 Останні 20 рядків логу:');
      lines.forEach(line => {
        if (line.trim()) console.log(line);
      });
    } catch (error) {
      logger.error(`Помилка читання логів: ${error.message}`);
    }
  }

  async start() {
    try {
      logger.info('🎯 Браузерний бот для CryptoTrade проекту');
      
      const envReady = await this.checkEnvironment();
      if (!envReady) {
        this.rl.close();
        return;
      }

      while (true) {
        const choice = await this.showMainMenu();
        
        switch (choice) {
          case '1':
            await this.runSingleBot();
            break;
          case '2':
            await this.runScheduledBot();
            return;
          case '3':
            await this.runTests();
            break;
          case '4':
            await this.checkConfiguration();
            break;
          case '5':
            await this.runDemoMode();
            break;
          case '6':
            await this.integratePythonSignals();
            break;
          case '7':
            await this.monitorLogs();
            break;
          case '0':
            logger.info('👋 До побачення!');
            this.rl.close();
            return;
          default:
            logger.warn('❌ Невірний вибір');
        }
        
        if (choice !== '2') {
          await this.askQuestion('\nНатисніть Enter для продовження...');
        }
      }
    } catch (error) {
      logger.error(`💥 Критична помилка: ${error.message}`);
    } finally {
      this.rl.close();
    }
  }
}

// CLI інтерфейс
program
  .name('binance-browser-bot')
  .description('Binance Browser Automation Bot (інтегрований з CryptoTrade)')
  .version('1.0.0');

program
  .command('start')
  .description('Запустити інтерактивне меню')
  .action(async () => {
    const starter = new BotStarter();
    await starter.start();
  });

program
  .command('run')
  .description('Запустити браузерний бот одноразово')
  .action(async () => {
    const success = await main();
    process.exit(success ? 0 : 1);
  });

// Якщо скрипт запущений безпосередньо без аргументів
if (require.main === module) {
  if (process.argv.length === 2) {
    // Інтерактивний режим
    const starter = new BotStarter();
    starter.start();
  } else {
    // CLI режим
    program.parse();
  }
}

module.exports = { BotStarter };