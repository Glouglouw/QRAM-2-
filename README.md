# QRAM-2-
This project integrates the Black-Litterman model with sentiment analysis, adjusting forecasts using Monte Carlo simulations. It features sentiment-aware portfolio optimization, personalized risk profiling, and performance visualization via a Streamlit app, blending behavioral finance with modern portfolio theory.

Black-Litterman Portfolio Optimization with Sentiment Analysis
This repository presents a quantitative finance project that combines the Black-Litterman model with sentiment analysis to develop a sentiment-aware portfolio optimization framework. The model leverages insights from behavioral finance and integrates them into a robust quantitative framework for strategic asset allocation. The accompanying web application, built with Streamlit, provides an interactive platform for users to explore portfolio optimization, adjust parameters, and visualize results.

Project Overview
The goal of this project is to enhance traditional portfolio optimization techniques by incorporating social media sentiment data into return forecasts. By using Monte Carlo simulations, sentiment scores are transformed into forward-looking views, which are integrated into the Black-Litterman framework. This approach provides a novel way to account for behavioral factors and market sentiment in portfolio allocation.

Key Features
1. Sentiment-Adjusted Return Forecasting
Sentiment scores are derived from social media and financial news, reflecting market sentiment for individual assets.
Monte Carlo simulations are used to adjust return forecasts based on sentiment polarity and volatility.
2. Black-Litterman Model
Combines market equilibrium returns (implied returns) with sentiment-driven views to create an optimized portfolio.
Allows for constraints such as entropy-based diversification and short selling.
3. Personalized Risk Profiling
A Know-Your-Customer (KYC) questionnaire assesses user-specific risk aversion (gamma), enabling tailored portfolio strategies.
4. Portfolio Rebalancing and Visualization
Simulates monthly rebalancing of the portfolio, incorporating updated sentiment scores.
Provides detailed visualizations, including portfolio weights, cumulative returns, and risk statistics.
5. Interactive Web Application
Built with Streamlit, the app offers an intuitive interface for users to interact with the model, visualize results, and customize inputs.
Methodology
Data Collection and Processing:

Sentiment data is aggregated from social media posts and news articles.
Asset price data is sourced from financial APIs (e.g., Yahoo Finance).
Monthly sentiment and price data are synchronized for analysis.
Monte Carlo Sentiment Adjustment:

Simulates potential future price movements based on historical returns and volatility.
Adjusts expected returns based on the sentiment polarity, amplifying or mitigating forecasts.
Portfolio Optimization:

Implied Returns: Derived from a market-cap-weighted portfolio using equilibrium assumptions.
Sentiment Views: Constructed using the sentiment-adjusted return forecasts.
Black-Litterman Optimization: Balances implied returns and sentiment views with constraints to produce optimal weights.
Performance Evaluation:

Rebalances portfolios monthly and calculates cumulative returns.
Visualizes portfolio composition and performance metrics.
Technologies Used
Python: Core language for data processing and analysis.
Streamlit: Interactive web application framework for visualization.
NumPy & Pandas: Data manipulation and numerical computations.
SciPy: Portfolio optimization via constrained minimization.
yFinance: Asset price data extraction.
Matplotlib: Portfolio weight and performance visualizations.
Applications
Behavioral Finance: Integrating sentiment to capture market psychology.
Risk Management: Adapting portfolios based on user-specific risk aversion.
Investment Strategy: Dynamic allocation strategies for improved risk-adjusted returns.
This project bridges traditional portfolio theory with innovative sentiment analysis techniques, providing a unique perspective on investment strategy optimization. It showcases the potential of combining data-driven finance with behavioral insights to achieve enhanced portfolio performance.
