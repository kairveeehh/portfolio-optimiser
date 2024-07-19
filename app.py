pip install yfinance
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os
import google.generativeai as genai

# Set up your Google API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyA6e6zHO19FS8T06wU4Wly4d2vY68GRQqM'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def optimize_portfolio(tickers, start_date, end_date):
    df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    df.index = pd.to_datetime(df.index)
    
    cov_matrix = df.pct_change().apply(lambda x: np.log(1 + x)).cov()
    ind_er = df.resample('Y').last().pct_change().mean()
    ann_sd = df.pct_change().apply(lambda x: np.log(1 + x)).std().apply(lambda x: x * np.sqrt(252))
    assets = pd.concat([ind_er, ann_sd], axis=1)
    assets.columns = ['Returns', 'Volatility']
    p_ret = []
    p_vol = []
    p_weights = []
    num_assets = len(df.columns)
    num_portfolios = 10000
    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er)
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd * np.sqrt(250)
        p_vol.append(ann_sd)
    
    data = {'Returns': p_ret, 'Volatility': p_vol}
    for counter, symbol in enumerate(df.columns.tolist()):
        data[symbol + ' weight'] = [w[counter] for w in p_weights]
    
    portfolios = pd.DataFrame(data)
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    rf = 0.01
    optimal_risky_port = portfolios.iloc[((portfolios['Returns'] - rf) / portfolios['Volatility']).idxmax()]
    
    return df, portfolios, min_vol_port, optimal_risky_port

def explain_portfolio_allocation(portfolio_weights, company_list):
    allocation_str = ", ".join([f"{company}: {weight:.2%}" for company, weight in zip(company_list, portfolio_weights)])

    explanation_prompt = f"""
    As a financial advisor expert in portfolio management, explain the rationale behind the following portfolio allocation:
    {allocation_str}

    Consider the following factors in your explanation:
    1. Diversification strategy
    2. Risk management
    3. Potential for returns
    4. Industry representation
    5. Any notable overweight or underweight positions

    Provide a concise, clear explanation suitable for an investor to understand the strategy behind this allocation.
    """

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(explanation_prompt)
        explanation = response.text
        return explanation
    except Exception as e:
        return f"An error occurred while generating the explanation: {str(e)}"

st.title('Portfolio Optimizer')

tickers = st.text_input('Enter Stock Tickers (comma-separated)', 'AAPL,NKE,GOOGL,AMZN')
start_date = st.date_input('Start Date', pd.to_datetime('2019-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'))

if st.button('Optimize Portfolio'):
    tickers_list = tickers.split(',')
    df, portfolios, min_vol_port, optimal_risky_port = optimize_portfolio(tickers_list, start_date, end_date)

    optimal_weights = optimal_risky_port[df.columns + ' weight'].values
    explanation = explain_portfolio_allocation(optimal_weights, df.columns.tolist())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3)
    ax.scatter(min_vol_port['Volatility'], min_vol_port['Returns'], color='r', marker='*', s=500)
    ax.scatter(optimal_risky_port['Volatility'], optimal_risky_port['Returns'], color='b', marker='*', s=500)
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Returns')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    st.image(img, caption='Optimal Portfolio', use_column_width=True)
    
    # Plot the pie chart of the optimal portfolio allocation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(optimal_weights, labels=df.columns, autopct='%1.1f%%')
    ax.set_title("Optimal Portfolio Allocation")
    st.pyplot(fig)

    st.markdown("### Explanation of the Optimal Portfolio Allocation")
    st.markdown(explanation)
    
    # Explanation for the minimum volatility portfolio if desired
    min_vol_weights = min_vol_port[df.columns + ' weight'].values
    min_vol_explanation = explain_portfolio_allocation(min_vol_weights, df.columns.tolist())
    st.markdown("### Explanation of the Minimum Volatility Portfolio Allocation")
    st.markdown(min_vol_explanation)
