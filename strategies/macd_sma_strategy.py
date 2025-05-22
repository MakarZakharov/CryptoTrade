# Add these imports to your existing Binance bot
import pandas as pd
from trading_strategy import TradingStrategy  # Import your strategy module
import time
from threading import Thread
from datetime import datetime, timedelta

from signal import logger


# Add this class to your existing BinanceBot class or create a new trading bot class
class TradingBot(BinanceBot):
    """
    Extended Binance bot with automated trading strategy
    """

    def __init__(self, api_key: str, api_secret: str, use_testnet: bool = False):
        super().__init__(api_key, api_secret, use_testnet)

        # Trading strategy
        self.strategy = TradingStrategy(
            macd_fast=12,
            macd_slow=26,
            macd_signal_period=9,
            sma_fast=10,
            sma_slow=50
        )

        # Trading state
        self.is_trading = False
        self.current_position = None  # 'buy', 'sell', or None
        self.entry_price = None
        self.position_size = None
        self.trading_pair = 'BTC/USDT'  # Default trading pair
        self.check_interval = 300  # Check every 5 minutes

        # Risk management
        self.max_position_size = 0.1  # Max 10% of balance per trade
        self.stop_loss_percent = 5.0  # 5% stop loss

    def get_ohlc_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Get OHLC data for the trading pair

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '1h', '1d')
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLC data
        """
        try:
            if self.use_testnet:
                # Generate sample data for testnet
                import numpy as np
                dates = pd.date_range(end=datetime.now(), periods=limit, freq='H')
                np.random.seed(int(time.time()) % 1000)

                prices = []
                price = 50000.0  # Starting BTC price
                for _ in range(limit):
                    price += np.random.normal(0, price * 0.02)  # 2% volatility
                    prices.append(max(price, 1000))  # Minimum price

                return pd.DataFrame({
                    'timestamp': dates,
                    'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
                    'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                    'close': prices,
                    'volume': [np.random.randint(100, 1000) for _ in prices]
                })
            else:
                # Fetch real data from exchange
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df

        except Exception as e:
            logger.error(f"Error fetching OHLC data: {e}")
            return pd.DataFrame()

    def calculate_position_size(self, signal: str, base_currency: str = 'USDT') -> float:
        """
        Calculate position size based on available balance and risk management

        Args:
            signal: Trading signal ('buy' or 'sell')
            base_currency: Base currency for calculation

        Returns:
            Position size in base currency
        """
        try:
            balance = self.get_balance(base_currency)['free']
            max_position = float(balance) * self.max_position_size

            return max_position

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def execute_trade(self, signal: str, symbol: str) -> bool:
        """
        Execute a trade based on the signal

        Args:
            signal: Trading signal ('buy' or 'sell')
            symbol: Trading pair symbol

        Returns:
            True if trade executed successfully
        """
        try:
            logger.info(f"Executing {signal.upper()} order for {symbol}")

            # Calculate position size
            if signal == 'buy':
                position_size = self.calculate_position_size('buy', 'USDT')
                if position_size < 10:  # Minimum $10 trade
                    logger.warning("Position size too small, skipping trade")
                    return False

                # Execute buy order (using quote currency amount)
                order = self.place_micro_order(symbol, 'buy', Decimal(str(position_size)), is_quote=True)

            else:  # sell
                # For sell, we need to sell all available base currency
                base_curr = symbol.split('/')[0]  # e.g., 'BTC' from 'BTC/USDT'
                balance = self.get_balance(base_curr)['free']

                if balance <= 0:
                    logger.warning(f"No {base_curr} balance to sell")
                    return False

                order = self.place_micro_order(symbol, 'sell', balance)

            if order:
                # Update position tracking
                current_price = float(self.get_ohlc_data(symbol, limit=1)['close'].iloc[-1])
                self.current_position = signal
                self.entry_price = current_price
                self.position_size = position_size if signal == 'buy' else float(balance)

                logger.info(f"Trade executed: {signal.upper()} at {current_price}")
                return True
            else:
                logger.error("Failed to execute trade")
                return False

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def check_signals(self):
        """
        Check for trading signals and execute trades
        """
        try:
            logger.info(f"Checking signals for {self.trading_pair}")

            # Get market data
            df = self.get_ohlc_data(self.trading_pair, timeframe='1h', limit=100)
            if df.empty:
                logger.warning("No market data available")
                return

            # Get signal strength information
            signal_info = self.strategy.get_signal_strength(df)
            signal = signal_info.get('signal')
            strength = signal_info.get('strength', 0)
            confidence = signal_info.get('confidence', 'low')

            logger.info(f"Signal: {signal}, Strength: {strength}, Confidence: {confidence}")

            # Only trade on medium/high confidence signals
            if confidence not in ['medium', 'high']:
                logger.info("Signal confidence too low, skipping")
                return

            # Check if we should enter a new trade
            if self.strategy.should_enter_trade(df, self.current_position):
                if self.execute_trade(signal, self.trading_pair):
                    logger.info(f"Entered {signal.upper()} position")

            # Check if we should exit current trade
            elif self.current_position and self.entry_price:
                if self.strategy.should_exit_trade(df, self.current_position, self.entry_price):
                    exit_signal = 'sell' if self.current_position == 'buy' else 'buy'
                    if self.execute_trade(exit_signal, self.trading_pair):
                        logger.info(f"Exited {self.current_position.upper()} position")
                        self.current_position = None
                        self.entry_price = None
                        self.position_size = None

        except Exception as e:
            logger.error(f"Error checking signals: {e}")

    def trading_loop(self):
        """
        Main trading loop that runs in a separate thread
        """
        logger.info("Starting trading loop")

        while self.is_trading:
            try:
                self.check_signals()
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Trading loop interrupted")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def start_trading(self, trading_pair: str = 'BTC/USDT', check_interval: int = 300):
        """
        Start automated trading

        Args:
            trading_pair: Trading pair to trade
            check_interval: Check interval in seconds
        """
        if self.is_trading:
            print("Trading is already running")
            return

        self.trading_pair = trading_pair
        self.check_interval = check_interval
        self.is_trading = True

        # Reset strategy signals for new session
        self.strategy.reset_signals()

        # Start trading in a separate thread
        trading_thread = Thread(target=self.trading_loop, daemon=True)
        trading_thread.start()

        print(f"Started automated trading for {trading_pair}")
        print(f"Check interval: {check_interval} seconds")

    def stop_trading(self):
        """Stop automated trading"""
        self.is_trading = False
        print("Stopping automated trading...")

    def get_trading_status(self):
        """Get current trading status"""
        print(f"\n--- Trading Status ---")
        print(f"Trading Active: {self.is_trading}")
        print(f"Trading Pair: {self.trading_pair}")
        print(f"Current Position: {self.current_position}")

        if self.current_position:
            print(f"Entry Price: {self.entry_price}")
            print(f"Position Size: {self.position_size}")

            # Calculate current P&L
            try:
                current_data = self.get_ohlc_data(self.trading_pair, limit=1)
                if not current_data.empty:
                    current_price = float(current_data['close'].iloc[-1])
                    if self.current_position == 'buy':
                        pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
                    else:
                        pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100

                    print(f"Current Price: {current_price}")
                    print(f"Unrealized P&L: {pnl_percent:.2f}%")
            except Exception:
                pass

    def run_enhanced(self):
        """Enhanced main program loop with trading features"""
        print("\n=== Mini Binance Trading Bot ===")
        if self.use_testnet:
            print("⚠️ TESTNET MODE ACTIVE ⚠️")

        if not self.connect():
            sys.exit(1)

        while True:
            print("\n1. Convert Currency")
            print("2. Show Balance")
            print("3. Start Auto Trading")
            print("4. Stop Auto Trading")
            print("5. Trading Status")
            print("6. Manual Signal Check")
            print("7. Exit")

            try:
                choice = input("Choice (1-7): ").strip()

                if choice == '1':
                    # Original convert functionality
                    self.show_balance()
                    from_c = input("From currency: ").upper().strip()
                    to_c = input("To currency: ").upper().strip()

                    balance = self.get_balance(from_c)['free']
                    if balance <= 0:
                        print(f"No {from_c} available")
                        continue

                    print(f"Available: {balance:.8f} {from_c}")
                    amount_input = input(f"Amount to convert (or 'max'): ").strip().lower()

                    if amount_input == 'max':
                        if self.convert(from_c, to_c):
                            print("✅ Conversion complete")
                            time.sleep(1)
                            self.show_balance()
                        else:
                            print("❌ Conversion failed")
                    else:
                        try:
                            amount = Decimal(amount_input.replace(',', '.'))
                            if self.convert(from_c, to_c, amount):
                                print("✅ Conversion complete")
                                time.sleep(1)
                                self.show_balance()
                            else:
                                print("❌ Conversion failed")
                        except Exception as e:
                            print(f"Invalid amount: {e}")

                elif choice == '2':
                    self.show_balance()

                elif choice == '3':
                    if self.is_trading:
                        print("Trading is already active")
                    else:
                        pair = input("Trading pair (default BTC/USDT): ").strip() or 'BTC/USDT'
                        interval = input("Check interval in seconds (default 300): ").strip()
                        interval = int(interval) if interval.isdigit() else 300
                        self.start_trading(pair, interval)

                elif choice == '4':
                    self.stop_trading()

                elif choice == '5':
                    self.get_trading_status()

                elif choice == '6':
                    pair = input("Trading pair (default BTC/USDT): ").strip() or 'BTC/USDT'
                    print(f"Checking signals for {pair}...")

                    df = self.get_ohlc_data(pair)
                    if not df.empty:
                        signal_info = self.strategy.get_signal_strength(df)
                        print(f"Signal Analysis: {signal_info}")
                    else:
                        print("Failed to get market data")

                elif choice == '7':
                    self.stop_trading()
                    print("Exiting")
                    break

            except KeyboardInterrupt:
                print("\nExiting")
                self.stop_trading()
                break
            except ValueError:
                print("Invalid input")
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")


# Usage example
if __name__ == "__main__":
    # Use the enhanced trading bot instead of the original
    try:
        use_testnet = USE_TESTNET
        if len(sys.argv) > 1:
            if sys.argv[1].lower() in ('--testnet', '-t'):
                use_testnet = True
            elif sys.argv[1].lower() in ('--live', '-l'):
                use_testnet = False

        # Create the enhanced trading bot
        bot = TradingBot(API_KEY, API_SECRET, use_testnet)
        bot.run_enhanced()  # Use enhanced run method

    except KeyboardInterrupt:
        print("\nProgram terminated")
    except Exception as e:
        logger.critical(f"Critical error: {e}")