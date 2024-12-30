import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def setup_logging():
    """Configure logging for the scanner"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_stock_list(filename):
    """Load stock symbols from a text file"""
    try:
        with open(filename, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logging.error(f"Stock list file {filename} not found")
        return []

def get_stock_data(symbol, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_vcp_metrics(df):
    """Calculate metrics needed for VCP pattern identification"""
    if len(df) < 50:  # Need sufficient data for analysis
        return None
    
    # Calculate key metrics
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    
    # Calculate volatility (20-day standard deviation)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Calculate volume metrics
    df['Volume_MA'] = df['Volume'].rolling(window=50).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df

def check_vcp_pattern(df):
    """
    Check if a stock exhibits VCP pattern characteristics
    Returns: dict with pattern details and boolean indicating if pattern found
    """
    if df is None or len(df) < 50:
        return {'vcp_found': False}
    
    # Get recent data (last 3 months)
    recent_df = df.tail(60)
    
    # VCP Criteria
    criteria = {
        'price_above_mas': False,  # Price above major moving averages
        'decreasing_volatility': False,  # Contracting volatility
        'higher_lows': False,  # Series of higher lows
        'volume_dry_up': False,  # Decreasing volume
    }
    
    # Check if price is above moving averages
    last_price = recent_df['Close'].iloc[-1]
    criteria['price_above_mas'] = (
        last_price > recent_df['20_MA'].iloc[-1] and 
        last_price > recent_df['50_MA'].iloc[-1]
    )
    
    # Check for decreasing volatility
    vol_change = (
        recent_df['Volatility'].iloc[-1] / 
        recent_df['Volatility'].iloc[-20]
    )
    criteria['decreasing_volatility'] = vol_change < 0.8
    
    # Check for higher lows in the last 3 months
    lows = recent_df['Low'].rolling(window=5).min()
    criteria['higher_lows'] = (
        lows.iloc[-1] > lows.iloc[-20] and 
        lows.iloc[-20] > lows.iloc[-40]
    )
    
    # Check for volume dry-up
    recent_volume_avg = recent_df['Volume_Ratio'].tail(10).mean()
    criteria['volume_dry_up'] = recent_volume_avg < 0.8
    
    # Calculate pattern strength score (0-100)
    pattern_score = sum([
        criteria['price_above_mas'] * 30,
        criteria['decreasing_volatility'] * 25,
        criteria['higher_lows'] * 25,
        criteria['volume_dry_up'] * 20
    ])
    
    # VCP pattern is considered valid if score is above 75
    vcp_found = pattern_score >= 75
    
    return {
        'vcp_found': vcp_found,
        'pattern_score': pattern_score,
        'criteria_met': criteria,
        'last_price': last_price,
        'current_volatility': recent_df['Volatility'].iloc[-1],
        'volume_ratio': recent_volume_avg
    }

def scan_stocks(input_file, output_file):
    """
    Main function to scan stocks for VCP patterns
    """
    setup_logging()
    logging.info("Starting VCP pattern scan...")
    
    # Load stock list
    symbols = load_stock_list(input_file)
    if not symbols:
        logging.error("No symbols loaded. Exiting...")
        return
    
    # Store results
    results = []
    
    # Scan each stock
    for symbol in symbols:
        logging.info(f"Scanning {symbol}")
        
        # Get stock data
        df = get_stock_data(symbol)
        if df is None:
            continue
            
        # Calculate metrics
        df = calculate_vcp_metrics(df)
        
        # Check for VCP pattern
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
                'Volume_Dry_Up': pattern_results['criteria_met']['volume_dry_up']
            })
    
    # Save results to CSV
    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)
        logging.info(f"Found {len(results)} stocks with VCP patterns. Results saved to {output_file}")
    else:
        logging.info("No stocks matching VCP pattern criteria found")

if __name__ == "__main__":
    scan_stocks("watchlist.txt", "vcp_results.csv")