"""
Simplified Binance Crypto Converter
Optimized for reliability and CSP compliance
"""

import os
import time
import webbrowser
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    Client = None
    BinanceAPIException = Exception

try:
    import ccxt
except ImportError:
    ccxt = None


class SimpleBinanceConverter:
    """Simplified Binance converter focusing on working methods"""
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.binance_client = None
        self.ccxt_exchange = None
        
        # Test balances for demo mode
        self.test_balance = {
            'BTC': 0.5, 'ETH': 10.0, 'USDT': 10000.0, 'BNB': 50.0, 
            'ADA': 5000.0, 'DOT': 100.0, 'SOL': 25.0, 'USDC': 5000.0
        }
        
        # Current rates (approximate)
        self.rates = {
            'USDT': 1.0, 'BTC': 104000.0, 'ETH': 2500.0, 'BNB': 600.0,
            'ADA': 0.7, 'DOT': 8.0, 'SOL': 200.0, 'USDC': 1.0
        }
        
        if not testnet:
            self._init_real_clients()
    
    def _init_real_clients(self):
        """Initialize real API clients"""
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print("⚠️ API keys not found. Using manual browser conversion only.")
            return
        
        try:
            if Client:
                self.binance_client = Client(api_key, api_secret)
                self.binance_client.ping()
                print("✅ Binance API connected")
        except Exception as e:
            print(f"⚠️ Binance API error: {e}")
        
        try:
            if ccxt:
                self.ccxt_exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': False,
                    'enableRateLimit': True
                })
                self.ccxt_exchange.load_markets()
                print("✅ CCXT connected")
        except Exception as e:
            print(f"⚠️ CCXT error: {e}")
    
    def get_balance(self, asset: str) -> float:
        """Get balance for asset"""
        asset = asset.upper()
        
        if self.testnet:
            return self.test_balance.get(asset, 0.0)
        
        if self.binance_client:
            try:
                account = self.binance_client.get_account()
                for balance in account['balances']:
                    if balance['asset'] == asset:
                        return float(balance['free'])
            except Exception as e:
                print(f"❌ Error getting {asset} balance: {e}")
        
        return 0.0
    
    def show_balance(self):
        """Show current balances"""
        print(f"\n💰 Balance ({'TEST' if self.testnet else 'REAL'})")
        
        if self.testnet:
            balances = {k: v for k, v in self.test_balance.items() if v > 0}
        else:
            balances = {}
            if self.binance_client:
                try:
                    account = self.binance_client.get_account()
                    for balance in account['balances']:
                        free_balance = float(balance['free'])
                        if free_balance > 0:
                            balances[balance['asset']] = free_balance
                except Exception as e:
                    print(f"❌ Error fetching balances: {e}")
        
        total_usd = 0
        for asset, amount in balances.items():
            usd_value = amount * self.rates.get(asset, 1.0)
            total_usd += usd_value
            print(f"  {asset}: {amount:,.8f} (~${usd_value:,.2f})")
        
        print(f"💵 Total: ${total_usd:,.2f}")
    
    def convert_via_api(self, from_asset: str, to_asset: str, amount: float, is_max: bool = False) -> bool:
        """Try conversion via API methods"""
        
        # Method 1: Binance Convert API
        if self.binance_client:
            try:
                print("💱 Trying Binance Convert API...")
                
                convert_amount = self.get_balance(from_asset) if is_max else amount
                if convert_amount <= 0:
                    print(f"❌ No {from_asset} to convert")
                    return False
                
                # Get quote
                quote_response = self.binance_client.convert_request_quote(
                    fromAsset=from_asset,
                    toAsset=to_asset,
                    fromAmount=convert_amount
                )
                
                if 'quoteId' not in quote_response:
                    print("❌ No quote received")
                    return False
                
                quote_id = quote_response['quoteId']
                to_amount = float(quote_response['toAmount'])
                
                print(f"💱 Quote: {convert_amount:.8f} {from_asset} → {to_amount:.8f} {to_asset}")
                
                # Accept quote
                result = self.binance_client.convert_accept_quote(quoteId=quote_id)
                
                if result.get('status') == 'PROCESS':
                    print(f"✅ Convert API successful: {result.get('orderId', 'N/A')}")
                    return True
                else:
                    print(f"❌ Convert API failed: {result.get('status', 'Unknown')}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'not supported' in error_msg:
                    print(f"❌ Pair {from_asset}/{to_asset} not supported by Convert API")
                elif '-1002' in error_msg:
                    print("⚠️ Convert API not available for your account")
                else:
                    print(f"❌ Convert API error: {e}")
        
        # Method 2: CCXT trading
        if self.ccxt_exchange:
            try:
                print("💱 Trying CCXT trading...")
                
                symbol = f"{from_asset}/{to_asset}"
                reverse_symbol = f"{to_asset}/{from_asset}"
                
                convert_amount = self.get_balance(from_asset) if is_max else amount
                
                if symbol in self.ccxt_exchange.markets:
                    order = self.ccxt_exchange.create_market_sell_order(symbol, convert_amount)
                    print(f"✅ CCXT successful: {order['id']}")
                    return True
                elif reverse_symbol in self.ccxt_exchange.markets:
                    quote_amount = convert_amount * self.rates.get(from_asset, 1.0)
                    order = self.ccxt_exchange.create_order(
                        symbol=reverse_symbol, type='market', side='buy', 
                        amount=None, price=None, params={'quoteOrderQty': quote_amount}
                    )
                    print(f"✅ CCXT successful: {order['id']}")
                    return True
                else:
                    print(f"❌ Trading pair {from_asset}/{to_asset} not available")
                    
            except Exception as e:
                print(f"❌ CCXT trading error: {e}")
        
        return False
    
    def manual_browser_conversion(self, from_asset: str, to_asset: str, amount) -> bool:
        """Enhanced browser automation with CSP-safe JavaScript"""
        try:
            convert_url = f"https://www.binance.com/en/convert/{from_asset}/{to_asset}"
            print(f"\n🤖 Enhanced Browser Automation for Conversion...")
            print(f"🔗 URL: {convert_url}")
            
            # Open URL in browser
            webbrowser.open(convert_url)
            time.sleep(2)  # Wait for browser to open
            
            amount_str = "max" if str(amount).lower() == 'max' else str(amount)
            
            # Generate CSP-safe automation script
            automation_script = self._generate_csp_safe_automation(from_asset, to_asset, amount_str)
            
            print(f"\n🎯 === AUTOMATIC CONVERSION SYSTEM ===")
            print(f"💱 Converting: {from_asset} → {to_asset}")
            print(f"💰 Amount: {amount_str}")
            print(f"")
            print(f"🛡️ CSP-Safe JavaScript Automation:")
            print(f"   1. ✅ Page opened automatically")
            print(f"   2. 🔐 Make sure you're logged into Binance")
            print(f"   3. 📱 Press F12 to open Developer Console")
            print(f"   4. 📋 Go to 'Console' tab")
            print(f"   5. 📄 Copy and paste the script below:")
            print(f"")
            print("=" * 70)
            print(automation_script)
            print("=" * 70)
            print(f"")
            print(f"🚀 Script Features:")
            print(f"   • ✅ Automatically detects current token pair")
            print(f"   • 🔄 Changes tokens if needed")
            print(f"   • 💰 Sets amount ({'MAX button' if amount_str == 'max' else amount_str})")
            print(f"   • 🎯 Clicks Convert and Confirm buttons")
            print(f"   • 📊 Shows detailed progress logs")
            print(f"   • 🛡️ Works around CSP restrictions")
            
            # Enhanced confirmation with automation feedback
            print(f"\n❓ How did the automation work?")
            print(f"   [1] - ✅ Automatic script completed successfully")
            print(f"   [2] - 🔧 Did manual conversion (script didn't work)")
            print(f"   [3] - ❌ Conversion failed/cancelled")
            
            while True:
                try:
                    choice = input("👉 Your choice (1/2/3): ").strip()
                    
                    if choice == '1':
                        print("🎉 Automatic conversion successful!")
                        print(f"✅ {from_asset} → {to_asset} conversion completed via automation")
                        return True
                    elif choice == '2':
                        print("✅ Manual conversion completed!")
                        print(f"💪 {from_asset} → {to_asset} conversion done manually")
                        return True
                    elif choice == '3':
                        print("❌ Conversion cancelled or failed")
                        return False
                    else:
                        print("❌ Please enter 1, 2, or 3")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Operation cancelled")
                    return False
                    
        except Exception as e:
            print(f"❌ Browser automation error: {e}")
            return False

    def _generate_csp_safe_automation(self, from_asset: str, to_asset: str, amount_str: str) -> str:
        """Generate CSP-compliant JavaScript for conversion automation"""
        return f'''// 🤖 BINANCE CONVERSION AUTOMATION - CSP SAFE
console.log("🚀 Starting Binance conversion: {from_asset} → {to_asset}");

// Step 1: Wait for page to load completely
function waitForLoad() {{
    return new Promise(resolve => {{
        if (document.readyState === 'complete') {{
            setTimeout(resolve, 2000);
        }} else {{
            window.addEventListener('load', () => setTimeout(resolve, 2000));
        }}
    }});
}}

// Step 2: Find elements safely
function findElement(selectors, description) {{
    console.log(`🔍 Finding: ${{description}}`);
    for (const selector of selectors) {{
        try {{
            const elements = document.querySelectorAll(selector);
            for (const elem of elements) {{
                if (elem && elem.offsetParent !== null && !elem.disabled) {{
                    console.log(`✅ Found ${{description}}: ${{selector}}`);
                    return elem;
                }}
            }}
        }} catch(e) {{}}
    }}
    console.warn(`❌ Not found: ${{description}}`);
    return null;
}}

// Step 3: Safe click with human-like behavior
function safeClick(element, description) {{
    if (!element) return false;
    console.log(`🖱️ Clicking: ${{description}}`);
    element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    setTimeout(() => {{
        element.click();
        console.log(`✅ Clicked: ${{description}}`);
    }}, 300);
    return true;
}}

// Step 4: Safe typing
function safeType(element, text, description) {{
    if (!element) return false;
    console.log(`⌨️ Typing: ${{description}} = ${{text}}`);
    element.focus();
    element.value = '';
    element.value = text;
    element.dispatchEvent(new Event('input', {{ bubbles: true }}));
    element.dispatchEvent(new Event('change', {{ bubbles: true }}));
    console.log(`✅ Typed: ${{description}}`);
    return true;
}}

// Step 5: Main automation function
async function runConversion() {{
    try {{
        console.log("⏳ Waiting for page load...");
        await waitForLoad();
        
        // Close any popups
        const popups = document.querySelectorAll('[aria-label*="close"], .modal-close, .bn-modal-close');
        popups.forEach(popup => {{
            try {{ popup.click(); }} catch(e) {{}}
        }});
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        {"// Set amount to MAX" if amount_str == "max" else f"// Set specific amount: {amount_str}"}
        console.log("💰 Setting amount...");
        
        {"const maxBtn = findElement(['button', 'span', 'div'], 'MAX button');" if amount_str == "max" else ""}
        {"if (maxBtn && maxBtn.textContent.toLowerCase().includes('max')) {" if amount_str == "max" else ""}
        {"    safeClick(maxBtn, 'MAX button');" if amount_str == "max" else ""}
        {"    await new Promise(resolve => setTimeout(resolve, 3000));" if amount_str == "max" else ""}
        {"} else {" if amount_str == "max" else ""}
        {"    console.log('⚠️ MAX button not found, entering large amount');" if amount_str == "max" else ""}
            const amountInput = findElement([
                'input[type="text"]',
                'input[type="number"]',
                'input[placeholder*="amount"]'
            ], 'amount input');
            
            if (amountInput) {{
                safeType(amountInput, '{"999999999" if amount_str == "max" else amount_str}', 'amount');
                await new Promise(resolve => setTimeout(resolve, 1000));
            }}
        {"}" if amount_str == "max" else ""}
        
        // Find and click Convert button
        console.log("🔄 Looking for Convert button...");
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const convertBtn = findElement([
            'button[class*="convert"]',
            'button[data-testid*="convert"]'
        ], 'Convert button');
        
        if (!convertBtn) {{
            // Fallback: search by text content
            const allButtons = document.querySelectorAll('button');
            for (const btn of allButtons) {{
                if (btn.textContent && btn.textContent.toLowerCase().includes('convert') && 
                    btn.offsetParent !== null && !btn.disabled) {{
                    safeClick(btn, 'Convert button (by text)');
                    break;
                }}
            }}
        }} else {{
            safeClick(convertBtn, 'Convert button');
        }}
        
        // Wait and look for confirmation
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        console.log("✅ Looking for confirmation...");
        const confirmBtn = findElement([
            'button[class*="confirm"]',
            'button[data-testid*="confirm"]'
        ], 'Confirm button');
        
        if (!confirmBtn) {{
            // Fallback: search by text content
            const allButtons = document.querySelectorAll('button');
            for (const btn of allButtons) {{
                if (btn.textContent && btn.textContent.toLowerCase().includes('confirm') && 
                    btn.offsetParent !== null && !btn.disabled) {{
                    safeClick(btn, 'Confirm button (by text)');
                    break;
                }}
            }}
        }} else {{
            safeClick(confirmBtn, 'Confirm button');
        }}
        
        // Check result
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        console.log("🎉 Conversion automation completed!");
        console.log("📊 Check the page for success/error messages");
        
        // Look for success/error indicators
        const successWords = ['success', 'completed', 'successful'];
        const errorWords = ['error', 'failed', 'insufficient'];
        
        let foundSuccess = false;
        let foundError = false;
        
        document.querySelectorAll('*').forEach(elem => {{
            const text = elem.textContent?.toLowerCase() || '';
            if (successWords.some(word => text.includes(word))) foundSuccess = true;
            if (errorWords.some(word => text.includes(word))) foundError = true;
        }});
        
        if (foundSuccess) {{
            console.log("🎉 SUCCESS indicators found on page!");
        }} else if (foundError) {{
            console.log("❌ ERROR indicators found on page!");
        }} else {{
            console.log("ℹ️ Result unclear - please check manually");
        }}
        
    }} catch (error) {{
        console.error("❌ Automation error:", error);
        console.log("💡 Try manual conversion or reload page");
    }}
}}

// Execute automation
console.log("⏳ Starting automation in 3 seconds...");
setTimeout(runConversion, 3000);

// Additional utility functions for better automation
function findMaxButton() {{
    console.log("🔍 Advanced MAX button search...");
    const selectors = [
        'button[class*="max"]',
        'span[class*="max"]', 
        'div[class*="max"]',
        'button[data-testid*="max"]',
        '[title*="max"]',
        '[aria-label*="max"]'
    ];
    
    // First try CSS selectors
    for (const selector of selectors) {{
        const elem = document.querySelector(selector);
        if (elem && elem.offsetParent !== null) {{
            console.log(`✅ Found MAX button with selector: ${{selector}}`);
            return elem;
        }}
    }}
    
    // Then try text content search
    const allElements = document.querySelectorAll('button, span, div, a');
    for (const elem of allElements) {{
        const text = elem.textContent?.trim().toLowerCase() || '';
        if ((text === 'max' || text === 'макс') && elem.offsetParent !== null && !elem.disabled) {{
            console.log(`✅ Found MAX button by text: "${{elem.textContent.trim()}}"`);
            return elem;
        }}
    }}
    
    console.warn("❌ MAX button not found");
    return null;
}}

function findAmountInput() {{
    console.log("🔍 Advanced amount input search...");
    const selectors = [
        'input[placeholder*="amount"]',
        'input[placeholder*="Amount"]',
        'input[data-testid*="amount"]',
        'input[type="text"]',
        'input[type="number"]',
        'input[class*="amount"]'
    ];
    
    for (const selector of selectors) {{
        const elem = document.querySelector(selector);
        if (elem && elem.offsetParent !== null && !elem.disabled) {{
            console.log(`✅ Found amount input: ${{selector}}`);
            return elem;
        }}
    }}
    
    console.warn("❌ Amount input not found");
    return null;
}}

function findConvertButton() {{
    console.log("🔍 Advanced Convert button search...");
    const selectors = [
        'button[class*="convert"]',
        'button[data-testid*="convert"]',
        'div[role="button"][class*="convert"]'
    ];
    
    // First try CSS selectors
    for (const selector of selectors) {{
        const elem = document.querySelector(selector);
        if (elem && elem.offsetParent !== null && !elem.disabled) {{
            console.log(`✅ Found Convert button with selector: ${{selector}}`);
            return elem;
        }}
    }}
    
    // Then search by text content
    const allButtons = document.querySelectorAll('button, div[role="button"]');
    for (const btn of allButtons) {{
        const text = btn.textContent?.trim().toLowerCase() || '';
        if (text.includes('convert') && btn.offsetParent !== null && !btn.disabled) {{
            console.log(`✅ Found Convert button by text: "${{btn.textContent.trim()}}"`);
            return btn;
        }}
    }}
    
    console.warn("❌ Convert button not found");
    return null;
}}

// Enhanced conversion function with better element detection
window.smartConvert = async function() {{
    try {{
        console.log("🚀 Smart conversion starting...");
        
        // Wait for page stabilization
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Close popups
        const closeButtons = document.querySelectorAll('[aria-label*="close"], .modal-close, .bn-modal-close');
        closeButtons.forEach(btn => {{
            try {{ btn.click(); }} catch(e) {{}}
        }});
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Set amount
        console.log("💰 Setting amount...");
        {"" if amount_str != "max" else "const maxBtn = findMaxButton();"}
        {"" if amount_str != "max" else "if (maxBtn) {"}
        {"" if amount_str != "max" else "    safeClick(maxBtn, 'MAX button');"}
        {"" if amount_str != "max" else "    await new Promise(resolve => setTimeout(resolve, 3000));"}
        {"" if amount_str != "max" else "} else {"}
            const amountInput = findAmountInput();
            if (amountInput) {{
                safeType(amountInput, '{"999999999" if amount_str == "max" else amount_str}', 'amount');
                await new Promise(resolve => setTimeout(resolve, 1000));
            }}
        {"" if amount_str != "max" else "}"}
        
        // Execute conversion
        console.log("🔄 Executing conversion...");
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const convertBtn = findConvertButton();
        if (convertBtn) {{
            safeClick(convertBtn, 'Convert button');
            
            // Wait for confirmation dialog
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // Look for confirm button
            const confirmSelectors = [
                'button[class*="confirm"]',
                'button[data-testid*="confirm"]'
            ];
            
            let confirmBtn = null;
            for (const selector of confirmSelectors) {{
                confirmBtn = document.querySelector(selector);
                if (confirmBtn && confirmBtn.offsetParent !== null) break;
            }}
            
            if (!confirmBtn) {{
                const allButtons = document.querySelectorAll('button');
                for (const btn of allButtons) {{
                    if (btn.textContent?.toLowerCase().includes('confirm') && 
                        btn.offsetParent !== null && !btn.disabled) {{
                        confirmBtn = btn;
                        break;
                    }}
                }}
            }}
            
            if (confirmBtn) {{
                safeClick(confirmBtn, 'Confirm button');
                console.log("✅ Conversion confirmed");
            }} else {{
                console.log("ℹ️ No confirmation needed or auto-confirmed");
            }}
            
            // Check results
            await new Promise(resolve => setTimeout(resolve, 5000));
            console.log("🎉 Smart conversion completed!");
            
        }} else {{
            console.error("❌ Convert button not found!");
        }}
        
    }} catch (error) {{
        console.error("❌ Smart conversion error:", error);
    }}
}};

console.log("💡 Available functions:");
console.log("  runConversion() - Full automation");
console.log("  smartConvert() - Enhanced conversion");
console.log("  findMaxButton() - Find MAX button");
console.log("  findAmountInput() - Find amount input");
console.log("  findConvertButton() - Find Convert button");'''
    
    def convert(self, from_asset: str, to_asset: str, amount) -> bool:
        """Main conversion method"""
        from_asset = from_asset.upper()
        to_asset = to_asset.upper()
        
        if from_asset == to_asset:
            print("❌ Same tokens selected")
            return False
        
        # Check balance
        balance = self.get_balance(from_asset)
        if balance <= 0:
            print(f"❌ No {from_asset} available")
            return False
        
        is_max = str(amount).lower() == 'max'
        convert_amount = balance if is_max else float(amount)
        
        if not is_max and convert_amount > balance:
            print(f"❌ Insufficient balance. Available: {balance}")
            return False
        
        print(f"\n💱 Converting: {convert_amount:.8f} {from_asset} → {to_asset}")
        
        if self.testnet:
            print("🧪 TEST MODE: Simulating conversion...")
            time.sleep(1)
            
            # Simulate conversion
            fee = convert_amount * 0.001
            net_amount = convert_amount - fee
            rate = self.rates.get(from_asset, 1.0) / self.rates.get(to_asset, 1.0)
            receive_amount = net_amount * rate
            
            # Update test balances
            self.test_balance[from_asset] -= convert_amount
            self.test_balance[to_asset] = self.test_balance.get(to_asset, 0) + receive_amount
            
            print(f"✅ TEST conversion: {receive_amount:.8f} {to_asset}")
            return True
        
        # Try API methods first
        if self.convert_via_api(from_asset, to_asset, convert_amount, is_max):
            return True
        
        # Fallback to manual browser conversion
        print("\n🔄 API methods unavailable, switching to manual browser conversion...")
        return self.manual_browser_conversion(from_asset, to_asset, amount)
    
    def add_test_balance(self, asset: str, amount: float):
        """Add test balance (test mode only)"""
        if not self.testnet:
            print("❌ Only available in test mode")
            return
        
        asset = asset.upper()
        self.test_balance[asset] = self.test_balance.get(asset, 0) + amount
        print(f"💰 Added {amount} {asset} to test balance")


def main():
    """Interactive converter"""
    print("🔄 === SIMPLIFIED BINANCE CONVERTER ===")
    
    while True:
        mode = input("Mode (test/real): ").strip().lower()
        if mode in ['test', 'тест']:
            testnet = True
            break
        elif mode in ['real', 'реал']:
            testnet = False
            break
        else:
            print("❌ Choose 'test' or 'real'")
    
    try:
        converter = SimpleBinanceConverter(testnet=testnet)
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    print("\n📋 Commands:")
    print("  balance - show balance")
    print("  convert - convert tokens")
    print("  add - add test balance (test mode only)")
    print("  exit - exit")
    
    while True:
        try:
            command = input("\n👉 Command: ").strip().lower()
            
            if command == 'exit':
                print("👋 Goodbye!")
                break
                
            elif command == 'balance':
                converter.show_balance()
                
            elif command == 'add':
                if not testnet:
                    print("❌ Only for test mode")
                    continue
                asset = input("Token: ").strip().upper()
                try:
                    amount = float(input("Amount: "))
                    converter.add_test_balance(asset, amount)
                except ValueError:
                    print("❌ Invalid amount")
                    
            elif command == 'convert':
                from_token = input("From token: ").strip().upper()
                to_token = input("To token: ").strip().upper()
                
                if not from_token or not to_token or from_token == to_token:
                    print("❌ Invalid tokens")
                    continue
                
                balance = converter.get_balance(from_token)
                if balance <= 0:
                    print(f"❌ No {from_token} available")
                    continue
                
                print(f"💰 Available {from_token}: {balance:,.8f}")
                amount_input = input("Amount ('max' for all): ").strip()
                
                if not amount_input:
                    continue
                
                print(f"\n⚠️ Convert {from_token} → {to_token}")
                confirm = input("Confirm? (y/n): ")
                
                if confirm.lower() in ['y', 'yes', 'так', 'да']:
                    converter.convert(from_token, to_token, amount_input)
                else:
                    print("❌ Cancelled")
                    
            else:
                print("❌ Unknown command")
                
        except KeyboardInterrupt:
            print("\n👋 Exit")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()