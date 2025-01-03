import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List
import yfinance as yf
from scipy.stats import norm
import logging
import matplotlib.pyplot as plt
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for parameter validation"""
    pass

@dataclass
class OptionParams:
    """Data class for option parameters"""
    S0: float  # Current stock price
    K: float   # Strike price (price at which buy or sell)
    T: float   # Time to maturity (years)
    r: float   # Risk-free rate (based on the assumption of zero loss)
    sigma: float  # Volatility (calculated by something called annulized volitility)
    option_type: str  # 'call' or 'put'
    
    def validate(self):
        """Validate option parameters"""
        try:
            if not all(isinstance(x, (int, float)) for x in [self.S0, self.K, self.T, self.r, self.sigma]):
                raise ValidationError("All numerical parameters must be numbers")
            if self.S0 <= 0 or self.K <= 0:
                raise ValidationError("Stock and strike prices must be positive")
            if self.T <= 0:
                raise ValidationError("Time to maturity must be positive")
            if self.sigma <= 0:
                raise ValidationError("Volatility must be positive")
            if self.option_type not in ['call', 'put']:
                raise ValidationError("Option type must be 'call' or 'put'")
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise

class VolatilityCalculator:
    """Calculate historical volatility from stock data"""
    
    @staticmethod
    def calculate_historical_volatility(ticker: str, lookback_days: int = 252) -> float:
        """Calculate historical volatility using daily returns"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            returns = np.log(hist['Close'] / hist['Close'].shift(1))
            return np.sqrt(252) * returns.std()
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            raise

class MonteCarloEngine:
    """Monte Carlo simulation engine for option pricing"""
    
    def __init__(self, params: OptionParams):
        self.params = params
        self.params.validate()
    
    def generate_paths(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Generate stock price paths using geometric Brownian motion"""
        dt = self.params.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.params.S0
        
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.params.r - 0.5 * self.params.sigma**2) * dt + 
                self.params.sigma * np.sqrt(dt) * z
            )
        
        return paths
    
    def calculate_payoff(self, final_prices: np.ndarray) -> np.ndarray:
        """Calculate option payoff at maturity"""
        if self.params.option_type == 'call':
            return np.maximum(final_prices - self.params.K, 0)
        else:
            return np.maximum(self.params.K - final_prices, 0)
    
    def price_option(self, n_simulations: int = 10000, n_steps: int = 252) -> Tuple[float, float]:
        """Price the option using Monte Carlo simulation"""
        try:
            paths = self.generate_paths(n_simulations, n_steps)
            payoffs = self.calculate_payoff(paths[:, -1])
            
            option_price = np.exp(-self.params.r * self.params.T) * np.mean(payoffs)
            std_error = np.std(payoffs) / np.sqrt(n_simulations)
            
            return option_price, std_error
        except Exception as e:
            logger.error(f"Error in pricing option: {str(e)}")
            raise

class GreeksCalculator:
    """Calculate option Greeks using finite differences"""
    
    def __init__(self, params: OptionParams):
        self.params = params
        self.base_engine = MonteCarloEngine(params)
    
    def calculate_delta(self, h: float = 0.01) -> float:
        """Calculate Delta using finite differences"""
        try:
            up_params = OptionParams(**self.params.__dict__)
            down_params = OptionParams(**self.params.__dict__)
            
            up_params.S0 *= (1 + h)
            down_params.S0 *= (1 - h)
            
            up_price, _ = MonteCarloEngine(up_params).price_option()
            down_price, _ = MonteCarloEngine(down_params).price_option()
            
            return (up_price - down_price) / (2 * h * self.params.S0)
        except Exception as e:
            logger.error(f"Error calculating Delta: {str(e)}")
            raise
    
    def calculate_gamma(self, h: float = 0.01) -> float:
        """Calculate Gamma using finite differences"""
        try:
            center_price, _ = self.base_engine.price_option()
            
            up_params = OptionParams(**self.params.__dict__)
            down_params = OptionParams(**self.params.__dict__)
            
            up_params.S0 *= (1 + h)
            down_params.S0 *= (1 - h)
            
            up_price, _ = MonteCarloEngine(up_params).price_option()
            down_price, _ = MonteCarloEngine(down_params).price_option()
            
            return (up_price - 2*center_price + down_price) / (h * self.params.S0)**2
        except Exception as e:
            logger.error(f"Error calculating Gamma: {str(e)}")
            raise
    
    def calculate_theta(self, h: float = 0.01) -> float:
        """Calculate Theta using finite differences"""
        try:
            up_params = OptionParams(**self.params.__dict__)
            up_params.T += h
            
            current_price, _ = self.base_engine.price_option()
            future_price, _ = MonteCarloEngine(up_params).price_option()
            
            return -(future_price - current_price) / h
        except Exception as e:
            logger.error(f"Error calculating Theta: {str(e)}")
            raise
    
    def calculate_vega(self, h: float = 0.01) -> float:
        """Calculate Vega using finite differences"""
        try:
            up_params = OptionParams(**self.params.__dict__)
            down_params = OptionParams(**self.params.__dict__)
            
            up_params.sigma += h
            down_params.sigma -= h
            
            up_price, _ = MonteCarloEngine(up_params).price_option()
            down_price, _ = MonteCarloEngine(down_params).price_option()
            
            return (up_price - down_price) / (2 * h)
        except Exception as e:
            logger.error(f"Error calculating Vega: {str(e)}")
            raise
        
class MarketDataAnalyzer:
    """Analyze and compare market vs theoretical prices"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
    
    def get_market_option_price(self, strike: float, expiry_date: str, option_type: str) -> float:
        """Get market price for specific option"""
        try:
            options = self.stock.option_chain(expiry_date)
            chain = options.calls if option_type == 'call' else options.puts
            return float(chain[chain['strike'] == strike]['lastPrice'].iloc[0])
        except Exception as e:
            logger.error(f"Error fetching market price: {e}")
            raise

class PriceComparison:
    """Compare simulated vs market option prices"""
    
    def __init__(self, ticker: str, monte_carlo_engine: 'MonteCarloEngine'):
        self.ticker = ticker
        self.mc_engine = monte_carlo_engine
        self.stock = yf.Ticker(ticker)
    
    def get_market_prices(self, expiry_date: str) -> pd.DataFrame:
        """Get market option chain for comparison"""
        options = self.stock.option_chain(expiry_date)
        if self.mc_engine.params.option_type == 'call':
            return options.calls
        return options.puts
    
    def compare_prices(self, expiry_date: str) -> pd.DataFrame:
        """Compare simulated vs market prices for different strikes"""
        market_prices = self.get_market_prices(expiry_date)
        results = []
        
        for strike in market_prices['strike']:
            self.mc_engine.params.K = strike
            sim_price, _ = self.mc_engine.price_option()
            market_price = float(market_prices[market_prices['strike'] == strike]['lastPrice'].iloc[0])
            
            results.append({
                'strike': strike,
                'simulated_price': sim_price,
                'market_price': market_price,
                'difference': sim_price - market_price
            })
        
        return pd.DataFrame(results)


class Visualizer:
    """Create visualizations for analysis"""
    
    @staticmethod
    def plot_price_distribution(price_history: np.ndarray, bins: int = 50):
        plt.figure(figsize=(10, 6))
        plt.hist(price_history, bins=bins, density=True, alpha=0.7)
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_price_comparison(results: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        plt.plot(results['strike'], results['simulated_price'], 'b-', label='Simulated Price')
        plt.plot(results['strike'], results['market_price'], 'r--', label='Market Price')
        plt.title('Option Prices: Simulated vs Market')
        plt.xlabel('Strike Price')
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_risk_metrics(var_values: List[float], es_values: List[float], 
                         confidence_levels: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(confidence_levels, var_values, label='VaR')
        plt.plot(confidence_levels, es_values, label='Expected Shortfall')
        plt.title('Risk Metrics vs Confidence Level')
        plt.xlabel('Confidence Level')
        plt.ylabel('Risk Measure')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    """Example usage of the options pricing engine"""
    try:
        ticker = "AAPL"
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        
        vol_calculator = VolatilityCalculator()
        historical_vol = vol_calculator.calculate_historical_volatility(ticker)
        
        # Get available expiration dates
        expiry_dates = stock.options
        nearest_expiry = expiry_dates[0]  # Get first available expiry
        
        # Calculate time to expiry in years
        days_to_expiry = (pd.to_datetime(nearest_expiry) - pd.Timestamp.now()).days
        T = days_to_expiry / 365
        
        # Set up option parameters with actual stock price
        params = OptionParams(
            S0=current_price,
            K=current_price,  # ATM option
            T=T,
            r=0.05,
            sigma=historical_vol,
            option_type='call'
        )
        
        # Price the option
        engine = MonteCarloEngine(params)
        price, std_error = engine.price_option()
        
        # Calculate Greeks
        greeks = GreeksCalculator(params)
        delta = greeks.calculate_delta()
        gamma = greeks.calculate_gamma()
        theta = greeks.calculate_theta()
        vega = greeks.calculate_vega()
        
        # Print results
        print(f"Option Price: ${price:.2f} (±${std_error:.2f})")
        print(f"Greeks:")
        print(f"Delta: {delta:.4f}")
        print(f"Gamma: {gamma:.4f}")
        print(f"Theta: {theta:.4f}")
        print(f"Vega: {vega:.4f}")
        
        comparison = PriceComparison('AAPL', engine)
        results = comparison.compare_prices(nearest_expiry)
        Visualizer.plot_price_comparison(results)   
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()