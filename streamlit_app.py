import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# Title and Introduction
st.title("Black-Scholes Option Pricing Dashboard")
st.markdown("""
### Quick Guide for Beginners
An **option** is a contract giving you the right (but not obligation) to buy or sell a stock at a fixed price by a certain date.
- **Call Option**: Right to buy at the strike price.
- **Put Option**: Right to sell at the strike price.
- **Spot Price (S)**: Current stock price (e.g., Apple's price today).
- **Strike Price (K)**: Fixed price to buy/sell.
- **Time to Maturity (T)**: Years until option expires.
- **Risk-Free Rate (r)**: Safe interest rate (e.g., 0.05 for 5%).
- **Volatility (σ)**: How much the stock price swings (e.g., 0.2 for 20%).
- **Dividend Yield (q)**: Annual dividend rate (e.g., 0.02 for 2%).
This app uses the **Black-Scholes-Merton model** to calculate prices for European options (exercisable only at expiration).
Click below to see the formulas.
""")

with st.expander("Black-Scholes-Merton Formulas"):
    st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)")
    st.latex(r"P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)")
    st.latex(r"d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2) T}{\sigma \sqrt{T}}")
    st.latex(r"d_2 = d_1 - \sigma \sqrt{T}")

# Black-Scholes-Merton Functions
def black_scholes_call(S, K, T, r, sigma, q):
    d1 = (np.log(S/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma, q):
    d1 = (np.log(S/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

def black_scholes_greeks(S, K, T, r, sigma, q):
    d1 = (np.log(S/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta_call = np.exp(-q * T) * norm.cdf(d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_call = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r * T) * norm.cdf(d2) 
                  + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365  # Per day
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% volatility
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% rate
    return {
        "Delta": round(delta_call, 4),
        "Gamma": round(gamma, 4),
        "Theta (per day)": round(theta_call, 4),
        "Vega (per 1%)": round(vega, 4),
        "Rho (per 1%)": round(rho_call, 4)
    }

# Sidebar Inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL")
try:
    S_default = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
except:
    S_default = 100.0
    st.sidebar.warning("Invalid ticker. Using default spot price.")
S = st.sidebar.slider("Spot Price (S)", min_value=1.0, max_value=500.0, value=float(S_default), step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T, years)", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
r = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.1, value=0.05, step=0.01)
sigma = st.sidebar.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
q = st.sidebar.slider("Dividend Yield (q)", min_value=0.0, max_value=0.1, value=0.0, step=0.01)

# Error Handling
if T <= 0 or sigma <= 0:
    st.error("Invalid inputs: Time to Maturity and Volatility must be positive.")
else:
    # Calculate Single-Point Prices
    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)
    st.subheader("Option Prices")
    st.write(f"**Call Price**: ${call_price:.2f}")
    st.write(f"**Put Price**: ${put_price:.2f}")

    # Calculate Greeks
    greeks = black_scholes_greeks(S, K, T, r, sigma, q)
    st.subheader("Option Greeks (for Call)")
    st.table(greeks)

    # Heatmap Visualization
    st.subheader("Option Price Heatmaps")
    spot_range = np.linspace(max(1, S-50), S+50, 50)
    vol_range = np.linspace(0.05, 0.5, 50)
    S_grid, sigma_grid = np.meshgrid(spot_range, vol_range)
    call_grid = np.vectorize(black_scholes_call)(S_grid, K, T, r, sigma_grid, q)
    put_grid = np.vectorize(black_scholes_put)(S_grid, K, T, r, sigma_grid, q)

    # Call Heatmap
    plt.figure(figsize=(8, 6))
    cax = plt.pcolormesh(S_grid, sigma_grid, call_grid, cmap="RdYlGn", shading="auto")
    plt.xlabel("Spot Price ($)")
    plt.ylabel("Volatility")
    plt.title("Call Option Prices")
    plt.colorbar(cax, label="Price ($)")
    plt.grid(True, alpha=0.3)
    st.pyplot(plt.gcf())
    plt.close()

    # Put Heatmap
    plt.figure(figsize=(8, 6))
    cax = plt.pcolormesh(S_grid, sigma_grid, put_grid, cmap="RdYlGn", shading="auto")
    plt.xlabel("Spot Price ($)")
    plt.ylabel("Volatility")
    plt.title("Put Option Prices")
    plt.colorbar(cax, label="Price ($)")
    plt.grid(True, alpha=0.3)
    st.pyplot(plt.gcf())
    plt.close()

    # Line Plot
    st.subheader("Option Price vs Spot Price")
    spots = np.linspace(max(1, S-50), S+50, 100)
    calls = [black_scholes_call(s, K, T, r, sigma, q) for s in spots]
    puts = [black_scholes_put(s, K, T, r, sigma, q) for s in spots]
    plt.figure(figsize=(8, 6))
    plt.plot(spots, calls, label="Call", color="blue")
    plt.plot(spots, puts, label="Put", color="red")
    plt.xlabel("Spot Price ($)")
    plt.ylabel("Option Price ($)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()