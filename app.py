import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---- Fonction utilitaires ----
def calculate_monthly_log_returns(prices_df):
    log_returns = np.log(prices_df / prices_df.shift(1))
    log_returns = log_returns.apply(lambda x: x.fillna(x.mean()), axis=0)
    return log_returns

def calculate_implied_returns(rf, gamma, cov_matrix, initial_allocation):
    if gamma is None or gamma == 0:
        raise ValueError("Gamma (risk aversion) must be defined and non-zero.")
    return rf + (1 / gamma) * cov_matrix @ initial_allocation

def optimize_black_litterman(cov_matrix, implied_returns, tau, P, Q, omega):
    Gamma_matrix = tau * cov_matrix
    mu_bar = implied_returns + Gamma_matrix @ P.T @ np.linalg.inv(P @ Gamma_matrix @ P.T + omega) @ (Q - P @ implied_returns)

    n_assets = len(mu_bar)
    bounds = [(0, 1) for _ in range(n_assets)]
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    result = minimize(lambda weights: -weights @ mu_bar / portfolio_volatility(weights),
                      x0=np.ones(n_assets) / n_assets,
                      bounds=bounds, constraints=constraints)
    return result.x

def calculate_sentiment_adjusted_forecast(prices, polarities, tickers, num_simulations=10000):
    Q = []
    for ticker in tickers:
        if ticker not in prices.columns or ticker not in polarities.columns:
            Q.append(0)
            continue

        ticker_prices = prices[ticker].dropna()
        ticker_polarities = polarities[ticker].dropna()

        if len(ticker_prices) < 2:
            Q.append(0)
            continue

        mean_return = ticker_prices.pct_change().mean()
        std_return = ticker_prices.pct_change().std()
        initial_price = ticker_prices.iloc[-1]

        random_shocks = np.random.normal(mean_return, std_return, num_simulations)
        simulated_prices = initial_price * np.exp(random_shocks)

        S_MC_plus = simulated_prices.max()
        S_MC_minus = simulated_prices.min()

        sentiment_score = ticker_polarities.iloc[-1]
        if sentiment_score > 0:
            S_T_adjusted = initial_price + (S_MC_plus - initial_price) * sentiment_score
        else:
            S_T_adjusted = initial_price - (initial_price - S_MC_minus) * abs(sentiment_score)

        Q.append(S_T_adjusted / initial_price - 1)
    return np.array(Q)

def plot_portfolio_weights_bar_vertical(weights, tickers, date):
    """Graphique des poids du portefeuille."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tickers, weights, color="skyblue")
    ax.set_ylabel("Portfolio Weights")
    ax.set_xlabel("Tickers")
    ax.set_title(f"Portfolio Weights for {date}")

    # Annotations des poids au-dessus des barres
    for i, v in enumerate(weights):
        ax.text(i, v + 0.01, f"{v:.2%}", color='black', ha='center')
    st.pyplot(fig)

# ---- Chargement des données ----
data = pd.read_pickle('/Users/new/Desktop/QARM II/Project/Projet perso/sentratio_and_price_closetoclose_adj2.pkl')
data['date'] = pd.to_datetime(data['date']).dt.date  # Conversion en date uniquement
def get_symbols(symbols, data_source, ohlc, begin_date=None, end_date=None):
    out = []
    new_symbols = []

    for symbol in symbols:
        df = web.DataReader(symbol, data_source, begin_date, end_date) \
            [['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
        new_symbols.append(symbol)
        out.append(df[ohlc].astype('float'))
        data = pd.concat(out, axis=1)
        data.columns = new_symbols

    return data


symbols = ['AAPL', 'AMD', 'AMRN', 'AMZN', 'BABA', 'BAC', 'BB', 'META', 'GLD', 'IWM', 'JNUG', 'MNKD', 'NFLX', 'PLUG',
           'QQQ', 'SPY', 'TSLA', 'X', 'UVXY']
df = yf.download(tickers=symbols,
                 start="2014-09-22",
                 end="2020-03-23",
                 interval="1mo",  # Monthly Data
                 auto_adjust=True)

df = df[['Close']]
df.columns = df.columns.droplevel()
df = df.reset_index()

df['Date'] = pd.to_datetime(df['Date']).dt.date
df = df.set_index('Date')
monthly_prices = df

data['date'] = pd.to_datetime(data['date'])
data['Date'] = data['date'].dt.to_period('M')
monthly_polarity = data.groupby(['Date', 'ticker']).apply(
    lambda x: (x['Nbullish'].sum() - x['Nbearish'].sum()) /
              (x['Nbullish'].sum() + x['Nbearish'].sum() + 10)
).reset_index()

monthly_polarity.columns = ['Date', 'ticker', 'monthly_polarity']
pivoted_polarity = monthly_polarity.pivot(index='Date', columns='ticker', values='monthly_polarity')

pivoted_polarity = pivoted_polarity.sort_index(axis=1)
monthly_polarity = pivoted_polarity

monthly_polarity.index = monthly_polarity.index.to_timestamp()
monthly_polarity = monthly_polarity.loc['2014-10-01':]
monthly_polarity.index = monthly_polarity.index.strftime('%Y-%m-%d')

# Convertir les indices des deux DataFrames en datetime
monthly_prices.index = pd.to_datetime(monthly_prices.index)
monthly_polarity.index = pd.to_datetime(monthly_polarity.index)

# Rennomer Twitter et Facebook dans le data frame monthly polarity
monthly_polarity = monthly_polarity.rename(columns={'FB': 'META', 'TWTR': 'X'})
monthly_polarity = monthly_polarity[monthly_prices.columns]


# Configuration de la page
st.set_page_config(page_title="Black-Litterman Optimization", layout="wide")

# En-tête visuel
st.markdown(
    """
    <style>
    .header-banner {
        background-color: #f5f5f5;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    <div class="header-banner">
        <h1 style="color: #FF5733;">Welcome to our Project Team</h1>
        <p>Meet the professionals behind the optimization project!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Création des onglets ----
tabs = st.tabs(["Team", "Data", "KYC", "Optimization"])

# Onglet 1 : Présentation de l'équipe
with tabs[0]:
    st.title("Your team")
    st.markdown("Here are the 5 members in charge of your project:")

st.markdown(
    """
    <style>
    .team-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
        justify-items: center;
        text-align: center;
        margin-top: 20px;
    }
    .team-member {
        font-size: 16px;
        font-weight: bold;
        margin-top: 10px;
    }
    .bio {
        font-size: 14px;
        color: gray;
        margin-top: 5px;
    }
    .social-icons {
        margin-top: 10px;
    }
    .social-icons a {
        margin: 0 5px;
        color: #FF5733;
        font-size: 20px;
        text-decoration: none;
    }
    .social-icons a:hover {
        color: #900C3F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Informations des membres de l'équipe
team_members = [
    {
        "name": "Ludovik Thorens",
        "role": "Community manager",
        "bio": "Community manager with expertise in audience engagement and communication strategies.",
        "image": "/Users/new/Desktop/Ludo.jpeg",
        "socials": {
            "LinkedIn": "https://www.linkedin.com/in/ludovik-thorens-584b81221/",
            "Twitter": "https://twitter.com"
        }
    },
    {
        "name": "Niels Scharwatt",
        "role": "Software engineer",
        "bio": "Software engineer specializing in backend systems and data processing pipelines.",
        "image": "/Users/new/Desktop/Niels.jpeg",
        "socials": {
            "LinkedIn": "https://www.linkedin.com/in/niels-scharwatt/",
            "GitHub": "https://github.com"
        }
    },
    {
        "name": "Enzo Rua",
        "role": "Portfolio manager",
        "bio": "Portfolio manager with a focus on risk management and financial optimization.",
        "image": "/Users/new/Desktop/Enzo.jpeg",
        "socials": {
            "LinkedIn": "https://linkedin.com",
            "Instagram": "https://instagram.com"
        }
    },
    {
        "name": "Guillaume Granger",
        "role": "Data scientist",
        "bio": "Data scientist passionate about machine learning and predictive analytics.",
        "image": "/Users/new/Desktop/Guillaume.jpeg",
        "socials": {
            "LinkedIn": "https://www.linkedin.com/in/guillaumelgranger/",
            "Kaggle": "https://kaggle.com"
        }
    },
    {
        "name": "Bruno Oliveira",
        "role": "Data scientist",
        "bio": "Data scientist with experience in sentiment analysis and NLP.",
        "image": "/Users/new/Desktop/Bruno.jpeg",
        "socials": {
            "LinkedIn": "https://www.linkedin.com/in/bruno-oliveira-da-rocha-979a43326/",
            "Medium": "https://medium.com"
        }
    },
]

# Afficher les membres de l'équipe dans une grille
st.markdown('<div class="team-grid">', unsafe_allow_html=True)

for member in team_members:
    st.markdown(
        f"""
           <div>
               <img src="{member['image']}" alt="{member['name']}" width="150">
               <div class="team-member">{member['name']}</div>
               <div class="bio">{member['bio']}</div>
               <div class="social-icons">
                   {"".join(f'<a href="{link}" target="_blank">{icon}</a>' for icon, link in zip(['🔗', '🌐'], member['socials'].values()))}
               </div>
           </div>
           """,
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)
