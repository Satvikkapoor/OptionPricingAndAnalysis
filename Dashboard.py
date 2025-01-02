import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from OptionPricing import (
    OptionParams, MonteCarloEngine, GreeksCalculator, 
    VolatilityCalculator, PriceComparison, Visualizer
)

st.set_page_config(page_title="Options Pricing Dashboard", layout="wide")

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="1d")
    current_price = history['Close'].iloc[-1]  
    info = stock.info
    # current_price = info.get('regularMarketPrice') or info.get('previousClose', 0)
    return stock, current_price, info

def create_option_params(ticker, strike, expiry_date, option_type):
    stock = yf.Ticker(ticker)
    vol_calculator = VolatilityCalculator()
    historical_vol = vol_calculator.calculate_historical_volatility(ticker)
    
    days_to_expiry = (pd.to_datetime(expiry_date) - pd.Timestamp.now()).days
    T = days_to_expiry / 365
    
    return OptionParams(
        S0=stock.history(period='1d')['Close'].iloc[-1],
        K=strike,
        T=T,
        r=0.05, 
        sigma=historical_vol,
        option_type=option_type.lower()
    )

def plot_price_paths(paths, title):
    fig = go.Figure()
    for path in paths.T:
        fig.add_trace(go.Scatter(y=path, mode='lines', opacity=0.3))
    fig.update_layout(title=title, xaxis_title="Time Steps", yaxis_title="Price")
    return fig

def main():
    st.title("Options Pricing Dashboard")
    
    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    
    try:
        stock, current_price, info = fetch_stock_data(ticker)
        
        st.sidebar.metric("Current Price", f"${current_price:.2f}")
        
        option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
        strike = st.sidebar.number_input("Strike Price", 
                                       min_value=0.0, 
                                       value=float(current_price),
                                       step=1.0)
        
        expiry_dates = stock.options
        expiry = st.sidebar.selectbox("Expiry Date", expiry_dates)
        
        engine_type = st.sidebar.selectbox("Pricing Engine", ["Monte Carlo"])
        
        n_simulations = st.sidebar.slider("Number of Simulations", 
                                        min_value=100, 
                                        max_value=10000, 
                                        value=1000)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Price Analysis", "Greeks", "Market Comparison"])
        
        # Calculate option parameters
        params = create_option_params(ticker, strike, expiry, option_type)
        engine = MonteCarloEngine(params)
        
        with tab1:
            st.header("Option Price Analysis")
            
            # Monte Carlo price calculation
            price, std_error = engine.price_option(n_simulations=n_simulations)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Option Price", f"${price:.2f}")
            with col2:
                st.metric("Standard Error", f"${std_error:.4f}")
            
            # Generate and plot price paths
            paths = engine.generate_paths(10, 252)  # 10 sample paths
            st.plotly_chart(plot_price_paths(paths, "Sample Price Paths"))
        
        with tab2:
            st.header("Greeks Analysis")
            
            # Calculate Greeks
            greeks = GreeksCalculator(params)
            delta = greeks.calculate_delta()
            gamma = greeks.calculate_gamma()
            theta = greeks.calculate_theta()
            vega = greeks.calculate_vega()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Delta", f"{delta:.4f}")
            col2.metric("Gamma", f"{gamma:.4f}")
            col3.metric("Theta", f"{theta:.4f}")
            col4.metric("Vega", f"{vega:.4f}")
            
            st.markdown("""
            - **Delta**: Measure of the change in option price for a $1 change in underlying price
            - **Gamma**: Rate of change in delta for a $1 change in underlying price
            - **Theta**: Time decay; change in option price as time passes
            - **Vega**: Change in option price for a 1% change in volatility
            """)
        
        with tab3:
            st.header("Market Comparison")
            
            comparison = PriceComparison(ticker, engine)
            results = comparison.compare_prices(expiry)
            
            # Plot comparison
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['strike'], 
                                   y=results['simulated_price'],
                                   name='Simulated Price'))
            fig.add_trace(go.Scatter(x=results['strike'], 
                                   y=results['market_price'],
                                   name='Market Price'))
            fig.update_layout(title='Simulated vs Market Prices',
                            xaxis_title='Strike Price',
                            yaxis_title='Option Price')
            st.plotly_chart(fig)
            
            # Display results table
            st.dataframe(results)
            
            # Add download button
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Comparison Data",
                data=csv,
                file_name=f"{ticker}_option_comparison.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()