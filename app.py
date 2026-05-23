import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('data/synthetic_fraud_data.csv')

# =========================
# KPI
# =========================
total_transactions = len(df)
total_fraud = df['is_fraud'].sum()
fraud_rate = round((total_fraud / total_transactions) * 100, 2)
avg_amount = round(df['amount'].mean(), 2)

# =========================
# DEVICE ANALYSIS
# =========================
device_counts = df['device'].value_counts().reset_index()
device_counts.columns = ['device', 'count']

fig_device = px.bar(
    device_counts,
    x='device',
    y='count',
    title='Transaction Frequency by Device',
    text_auto=True
)

fig_device.update_layout(
    template='plotly_white'
)

# =========================
# CHANNEL ANALYSIS
# =========================
channel_counts = df['channel'].value_counts().reset_index()
channel_counts.columns = ['channel', 'count']

fig_channel = px.pie(
    channel_counts,
    names='channel',
    values='count',
    title='Transaction Channel Distribution',
    hole=0.4
)

# =========================
# DEVICE FRAUD RATIO
# =========================
device_fraud = df.groupby('device')['is_fraud'].mean().reset_index()
device_fraud['is_fraud'] = device_fraud['is_fraud'] * 100

fig_device_fraud = px.bar(
    device_fraud,
    x='device',
    y='is_fraud',
    title='Fraud Ratio by Device (%)',
    text_auto='.2f'
)

# =========================
# HOURLY ANALYSIS
# =========================
hourly_summary = df.groupby('transaction_hour').agg({
    'transaction_id': 'count',
    'is_fraud': 'sum'
}).reset_index()

hourly_summary.rename(columns={
    'transaction_id': 'total_transactions'
}, inplace=True)

fig_hourly = go.Figure()

fig_hourly.add_trace(
    go.Scatter(
        x=hourly_summary['transaction_hour'],
        y=hourly_summary['total_transactions'],
        mode='lines+markers',
        name='Total Transactions'
    )
)

fig_hourly.add_trace(
    go.Scatter(
        x=hourly_summary['transaction_hour'],
        y=hourly_summary['is_fraud'],
        mode='lines+markers',
        name='Fraud Transactions'
    )
)

fig_hourly.update_layout(
    title='Transaction and Fraud Pattern by Hour',
    xaxis_title='Hour',
    yaxis_title='Transactions',
    template='plotly_white'
)

# =========================
# MERCHANT ANALYSIS
# =========================
merchant_summary = df.groupby('merchant_category').agg({
    'transaction_id': 'count',
    'amount': 'mean',
    'is_fraud': 'mean'
}).reset_index()

merchant_summary.rename(columns={
    'transaction_id': 'total_transactions',
    'amount': 'avg_spending'
}, inplace=True)

merchant_summary['is_fraud'] = merchant_summary['is_fraud'] * 100

fig_merchant = go.Figure()

fig_merchant.add_trace(
    go.Bar(
        x=merchant_summary['merchant_category'],
        y=merchant_summary['total_transactions'],
        name='Total Transactions'
    )
)

fig_merchant.add_trace(
    go.Scatter(
        x=merchant_summary['merchant_category'],
        y=merchant_summary['is_fraud'],
        mode='lines+markers',
        name='Fraud Ratio (%)',
        yaxis='y2'
    )
)

fig_merchant.update_layout(
    title='Merchant Category Analysis',
    yaxis=dict(title='Transactions'),
    yaxis2=dict(
        title='Fraud Ratio (%)',
        overlaying='y',
        side='right'
    ),
    template='plotly_white'
)

# =========================
# FRAUD VS NON FRAUD PIE
# =========================
fraud_distribution = pd.DataFrame({
    'Category': ['Non Fraud', 'Fraud'],
    'Count': [
        len(df[df['is_fraud'] == 0]),
        len(df[df['is_fraud'] == 1])
    ]
})

fig_fraud_pie = px.pie(
    fraud_distribution,
    names='Category',
    values='Count',
    title='Fraud vs Non Fraud Transactions',
    hole=0.5
)

# =========================
# DASH APP
# =========================
app = dash.Dash(__name__)
app.title = 'Fraud Detection Dashboard'

# =========================
# LAYOUT
# =========================
app.layout = html.Div([

    html.Div([
        html.H1(
            'Fraud Detection Dashboard',
            className='dashboard-title'
        ),

        html.P(
            'Interactive Dashboard for Transaction Fraud Analysis',
            className='dashboard-subtitle'
        )

    ], className='header'),

    # KPI
    html.Div([

        html.Div([
            html.H3('Total Transactions'),
            html.H1(f'{total_transactions:,}')
        ], className='card'),

        html.Div([
            html.H3('Fraud Transactions'),
            html.H1(f'{total_fraud:,}')
        ], className='card'),

        html.Div([
            html.H3('Fraud Rate'),
            html.H1(f'{fraud_rate}%')
        ], className='card'),

        html.Div([
            html.H3('Average Amount'),
            html.H1(f'${avg_amount}')
        ], className='card')

    ], className='card-container'),

    # ROW 1
    html.Div([

        html.Div([
            dcc.Graph(figure=fig_fraud_pie)
        ], className='graph-box'),

        html.Div([
            dcc.Graph(figure=fig_channel)
        ], className='graph-box')

    ], className='row'),

    # ROW 2
    html.Div([

        html.Div([
            dcc.Graph(figure=fig_device)
        ], className='graph-box-full')

    ], className='row'),

    # ROW 3
    html.Div([

        html.Div([
            dcc.Graph(figure=fig_hourly)
        ], className='graph-box-full')

    ], className='row'),

    # ROW 4
    html.Div([

        html.Div([
            dcc.Graph(figure=fig_merchant)
        ], className='graph-box-full')

    ], className='row')

], className='main-container')
# =========================
# RUN SERVER
# =========================
if __name__ == '__main__':
    app.run(debug=True)