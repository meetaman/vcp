import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio
import aiohttp
from tqdm import tqdm  # For progress bar

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Load stock symbols from a file
def load_stock_list(filename):
    try:
        with open(filename, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logging.error(f"Stock list file {filename} not found")
        return []

# Fetch stock data asynchronously
async def get_stock_data(session, symbol, retries=3, delay=2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            if df.empty:
                logging.warning(f"No data found for {symbol}")
                return None
            return df
        except Exception as e:
            if attempt < retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logging.error(f"Failed to fetch data for {symbol} after {retries} attempts: {str(e)}")
                return None

# Calculate metrics for VCP pattern detection
def calculate_vcp_metrics(df):
    if len(df) < 50:  # Need sufficient data for analysis
        return None
    
    # Calculate key metrics
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume_MA'] = df['Volume'].rolling(window=50).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df

# Check for VCP pattern
def check_vcp_pattern(df):
    if df is None or len(df) < 50:
        return {'vcp_found': False}
    
    recent_df = df.tail(60)
    criteria = {
        'price_above_mas': False,
        'decreasing_volatility': False,
        'higher_lows': False,
        'volume_dry_up': False,
    }
    
    last_price = recent_df['Close'].iloc[-1]
    criteria['price_above_mas'] = (
        last_price > recent_df['20_MA'].iloc[-1] and 
        last_price > recent_df['50_MA'].iloc[-1]
    )
    
    vol_change = recent_df['Volatility'].iloc[-1] / recent_df['Volatility'].iloc[-20]
    criteria['decreasing_volatility'] = vol_change < 0.8
    
    lows = recent_df['Low'].rolling(window=5).min()
    criteria['higher_lows'] = (
        lows.iloc[-1] > lows.iloc[-20] and 
        lows.iloc[-20] > lows.iloc[-40]
    )
    
    recent_volume_avg = recent_df['Volume_Ratio'].tail(10).mean()
    criteria['volume_dry_up'] = recent_volume_avg < 0.8
    
    pattern_score = sum([
        criteria['price_above_mas'] * 30,
        criteria['decreasing_volatility'] * 25,
        criteria['higher_lows'] * 25,
        criteria['volume_dry_up'] * 20
    ])
    
    vcp_found = pattern_score >= 75
    
    # Prepare remarks for criteria met
    remarks = []
    for key, value in criteria.items():
        if value:
            remarks.append(key.replace('_', ' ').title())
    remarks = ", ".join(remarks) if remarks else "No triggers met"
    
    return {
        'vcp_found': vcp_found,
        'pattern_score': pattern_score,
        'criteria_met': criteria,
        'last_price': last_price,
        'current_volatility': recent_df['Volatility'].iloc[-1],
        'volume_ratio': recent_volume_avg,
        'remarks': remarks,
        'trigger_date': recent_df.index[-1].strftime('%Y-%m-%d')  # Date of the last data point
    }

# Main function to scan stocks
async def scan_stocks(input_file, output_file):
    setup_logging()
    logging.info("Starting VCP pattern scan...")
    
    symbols = load_stock_list(input_file)
    if not symbols:
        logging.error("No symbols loaded. Exiting...")
        return
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [get_stock_data(session, symbol) for symbol in symbols]
        stock_data = await asyncio.gather(*tasks)
        
        for symbol, df in tqdm(zip(symbols, stock_data), total=len(symbols), desc="Scanning stocks"):
            if df is None:
                continue
            
            df = calculate_vcp_metrics(df)
            pattern_results = check_vcp_pattern(df)
            
            if pattern_results['vcp_found']:
                results.append({
                    'Symbol': symbol,
                    'Pattern_Score': pattern_results['pattern_score'],
                    'Last_Price': pattern_results['last_price'],
                    'Volatility': pattern_results['current_volatility'],
                    'Volume_Ratio': pattern_results['volume_ratio'],
                    'Scan_Date': datetime.now().strftime('%Y-%m-%d'),
                    'Price_Above_MAs': pattern_results['criteria_met']['price_above_mas'],
                    'Decreasing_Volatility': pattern_results['criteria_met']['decreasing_volatility'],
                    'Higher_Lows': pattern_results['criteria_met']['higher_lows'],
                    'Volume_Dry_Up': pattern_results['criteria_met']['volume_dry_up'],
                    'Remarks': pattern_results['remarks'],  # New column for remarks
                    'Trigger_Date': pattern_results['trigger_date']  # New column for trigger date
                })
    
    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)
        logging.info(f"Found {len(results)} stocks with VCP patterns. Results saved to {output_file}")
    else:
        logging.info("No stocks matching VCP pattern criteria found")

if __name__ == "__main__":
    asyncio.run(scan_stocks("watchlist.txt", "vcp_results.csv"))