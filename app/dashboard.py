"""
Streamlit Dashboard
Interactive dashboard for Portfolio & Risk Management Engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from regime.regime_detector import RegimeDetector
from allocation.allocator import Allocator
from risk.risk_manager import RiskManager
from backtest.backtester import Backtester
from stress_test.stress_tester import StressTester
from explainability.explainer import Explainer
from utils.constants import TICKERS, REGIME_LABELS, ASSET_NAMES
from utils.helpers import calculate_metrics


# Page configuration
st.set_page_config(
    page_title="Adaptive Portfolio Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .regime-badge {
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(start_date, end_date):
    """Load and process all data."""
    loader = DataLoader(start_date=start_date, end_date=end_date)
    prices, returns, features = loader.get_data()
    return prices, returns, features


@st.cache_data
def run_full_analysis(prices, returns, features):
    """Run complete analysis pipeline."""
    # Regime detection
    regime_detector = RegimeDetector()
    regimes = regime_detector.detect_regimes(returns, features, method='rule_based')
    
    # Allocation (without risk management)
    allocator = Allocator()
    allocations_no_risk = allocator.calculate_allocations(
        regimes, returns, features, method='risk_parity'
    )
    
    # Allocation (with risk management)
    risk_manager = RiskManager()
    allocations_with_risk, exposure_scalars = risk_manager.apply_risk_controls(
        allocations_no_risk, returns
    )
    
    # Backtesting
    backtester = Backtester()
    results_no_risk = backtester.run_backtest(
        allocations_no_risk, returns, 'Without Risk Management'
    )
    results_with_risk = backtester.run_backtest(
        allocations_with_risk, returns, 'With Risk Management'
    )
    
    # Stress testing
    stress_tester = StressTester()
    stress_results = stress_tester.run_comprehensive_stress_tests(
        allocations_with_risk, allocations_no_risk, returns
    )
    
    return {
        'regimes': regimes,
        'allocations_no_risk': allocations_no_risk,
        'allocations_with_risk': allocations_with_risk,
        'exposure_scalars': exposure_scalars,
        'results_no_risk': results_no_risk,
        'results_with_risk': results_with_risk,
        'stress_results': stress_results,
        'backtester': backtester,
        'stress_tester': stress_tester
    }


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Autonomous Adaptive Portfolio & Risk Management Engine</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2015, 1, 1)
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now()
    )
    
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    if run_analysis:
        with st.spinner("Loading and processing data..."):
            prices, returns, features = load_and_process_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            st.success(f"Loaded {len(prices)} days of data for {len(TICKERS)} assets")
        
        with st.spinner("Running full analysis..."):
            analysis = run_full_analysis(prices, returns, features)
        
        st.success("Analysis complete!")
        
        # Store in session state
        st.session_state['analysis'] = analysis
        st.session_state['prices'] = prices
        st.session_state['returns'] = returns
        st.session_state['features'] = features
    
    # Display results if available
    if 'analysis' in st.session_state:
        display_dashboard(st.session_state['analysis'], st.session_state['prices'])


def display_dashboard(analysis, prices):
    """Display main dashboard with results."""
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎯 Regime & Allocation",
        "📈 Performance",
        "⚠️ Stress Tests",
        "📝 Explainability"
    ])
    
    # Tab 1: Overview
    with tab1:
        display_overview(analysis)
    
    # Tab 2: Regime & Allocation
    with tab2:
        display_regime_allocation(analysis, prices)
    
    # Tab 3: Performance
    with tab3:
        display_performance(analysis)
    
    # Tab 4: Stress Tests
    with tab4:
        display_stress_tests(analysis)
    
    # Tab 5: Explainability
    with tab5:
        display_explainability(analysis)


def display_overview(analysis):
    """Display overview metrics."""
    st.header("Portfolio Overview")
    
    # Current regime
    regimes = analysis['regimes']
    current_regime_id = int(regimes.iloc[-1])
    current_regime = REGIME_LABELS[current_regime_id]
    
    regime_colors = {
        0: '#28a745',  # Green - Trending Up
        1: '#ffc107',  # Yellow - Trending Down
        2: '#fd7e14',  # Orange - High Volatility
        3: '#dc3545'   # Red - Crash
    }
    
    st.markdown(f"""
    <div style='background-color: {regime_colors[current_regime_id]}; 
                color: white; 
                padding: 1rem; 
                border-radius: 0.5rem; 
                text-align: center;
                font-size: 1.5rem;
                font-weight: bold;'>
        Current Market Regime: {current_regime}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 With Risk Management")
        metrics_with = analysis['results_with_risk']['metrics']
        display_metric_cards(metrics_with)
    
    with col2:
        st.subheader("📉 Without Risk Management")
        metrics_without = analysis['results_no_risk']['metrics']
        display_metric_cards(metrics_without)
    
    # Current allocation
    st.markdown("---")
    st.subheader("Current Portfolio Allocation")
    
    current_alloc = analysis['allocations_with_risk'].iloc[-1]
    
    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=[ASSET_NAMES.get(t, t) for t in current_alloc.index],
        values=current_alloc.values,
        hole=0.4
    )])
    
    fig.update_layout(
        title="Current Asset Allocation",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_metric_cards(metrics):
    """Display metrics in cards."""
    for metric, value in metrics.items():
        if metric in ['CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
            st.metric(metric, f"{value:.2f}" if 'Ratio' in metric else f"{value:.2%}")


def display_regime_allocation(analysis, prices):
    """Display regime detection and allocation over time."""
    st.header("Regime Detection & Allocation Strategy")
    
    regimes = analysis['regimes']
    allocations = analysis['allocations_with_risk']
    
    # Regime over time
    st.subheader("Market Regime Over Time")
    
    fig = go.Figure()
    
    regime_numeric = regimes.values
    regime_names = [REGIME_LABELS[int(r)] for r in regime_numeric]
    
    fig.add_trace(go.Scatter(
        x=regimes.index,
        y=regime_numeric,
        mode='lines+markers',
        name='Regime',
        text=regime_names,
        hovertemplate='Date: %{x}<br>Regime: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Market Regime Evolution",
        xaxis_title="Date",
        yaxis_title="Regime",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['Trending Up', 'Trending Down', 'High Vol', 'Crash']
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Allocation over time (stacked area)
    st.subheader("Portfolio Allocation Over Time")
    
    fig = go.Figure()
    
    for ticker in allocations.columns:
        fig.add_trace(go.Scatter(
            x=allocations.index,
            y=allocations[ticker],
            mode='lines',
            name=ASSET_NAMES.get(ticker, ticker),
            stackgroup='one',
            groupnorm='percent'
        ))
    
    fig.update_layout(
        title="Dynamic Allocation Strategy",
        xaxis_title="Date",
        yaxis_title="Allocation (%)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_performance(analysis):
    """Display performance comparison."""
    st.header("Performance Analysis")
    
    # Equity curves
    st.subheader("Equity Curves Comparison")
    
    fig = go.Figure()
    
    pv_with = analysis['results_with_risk']['portfolio_value']
    pv_without = analysis['results_no_risk']['portfolio_value']
    
    fig.add_trace(go.Scatter(
        x=pv_with.index,
        y=pv_with.values,
        mode='lines',
        name='With Risk Management',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=pv_without.index,
        y=pv_without.values,
        mode='lines',
        name='Without Risk Management',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown comparison
    st.subheader("Drawdown Comparison")
    
    fig = go.Figure()
    
    dd_with = analysis['results_with_risk']['drawdown']
    dd_without = analysis['results_no_risk']['drawdown']
    
    fig.add_trace(go.Scatter(
        x=dd_with.index,
        y=dd_with.values * 100,
        mode='lines',
        name='With Risk Management',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=dd_without.index,
        y=dd_without.values * 100,
        mode='lines',
        name='Without Risk Management',
        line=dict(color='red', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics comparison table
    st.subheader("Metrics Comparison")
    
    comparison_df = analysis['backtester'].compare_strategies()
    st.dataframe(comparison_df.style.format("{:.2%}"), use_container_width=True)


def display_stress_tests(analysis):
    """Display stress test results."""
    st.header("Stress Test Results")
    
    stress_tester = analysis['stress_tester']
    
    st.markdown("""
    The portfolio is tested under three crisis scenarios:
    - **Market Crash**: -5% daily shock for 5 days
    - **Volatility Spike**: 3x volatility increase for 10 days
    - **Correlation Spike**: All assets move together downward
    """)
    
    # Summary table
    st.subheader("Stress Test Summary")
    report_df = stress_tester.generate_stress_report()
    st.dataframe(report_df.style.format({
        'Max Drawdown': '{:.2%}',
        'Final Value': '${:.2f}',
        'Sharpe Ratio': '{:.2f}',
        'Total Return': '{:.2%}'
    }), use_container_width=True)
    
    # Visual comparison
    st.subheader("Impact Comparison")
    
    scenario_type = st.selectbox(
        "Select Scenario",
        ['Crash', 'Vol Spike', 'Corr Spike']
    )
    
    scenarios = [k for k in analysis['stress_results'].keys() if scenario_type in k]
    
    if len(scenarios) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**With Risk Management**")
            scenario_with = [s for s in scenarios if 'With' in s][0]
            pv_with = analysis['stress_results'][scenario_with]['portfolio_value']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pv_with.index, y=pv_with.values, mode='lines'))
            fig.update_layout(title="Portfolio Value", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Without Risk Management**")
            scenario_without = [s for s in scenarios if 'Without' in s][0]
            pv_without = analysis['stress_results'][scenario_without]['portfolio_value']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pv_without.index, y=pv_without.values, mode='lines'))
            fig.update_layout(title="Portfolio Value", height=300)
            st.plotly_chart(fig, use_container_width=True)


def display_explainability(analysis):
    """Display explainability insights."""
    st.header("Decision Explainability")
    
    st.markdown("""
    This section provides human-readable explanations for portfolio decisions.
    All decisions are rule-based and deterministic.
    """)
    
    regimes = analysis['regimes']
    allocations = analysis['allocations_with_risk']
    
    # Sample explanations
    st.subheader("Sample Decision Explanations")
    
    explainer = Explainer()
    
    # Get regime change dates
    regime_changes = regimes[regimes != regimes.shift(1)].index[:5]
    
    for date in regime_changes:
        idx = regimes.index.get_loc(date)
        
        if idx > 0:
            regime = int(regimes.iloc[idx])
            prev_regime = int(regimes.iloc[idx - 1])
            alloc = allocations.iloc[idx].to_dict()
            prev_alloc = allocations.iloc[idx - 1].to_dict()
            
            with st.expander(f"📅 {date.strftime('%Y-%m-%d')} - {REGIME_LABELS[regime]}"):
                st.markdown(f"**Regime Change**: {REGIME_LABELS[prev_regime]} → {REGIME_LABELS[regime]}")
                
                st.markdown("**Allocation Adjustment**:")
                for asset in TICKERS:
                    change = alloc[asset] - prev_alloc[asset]
                    if abs(change) > 0.01:
                        direction = "🔺" if change > 0 else "🔻"
                        st.write(f"{direction} {ASSET_NAMES.get(asset, asset)}: {prev_alloc[asset]:.1%} → {alloc[asset]:.1%}")
                
                # Explanation
                regime_exp = explainer.explain_regime(
                    regime,
                    pd.Series({'portfolio_volatility': 0.15, 'market_drawdown': -0.05, 'SPY_rolling_return': 0.02}),
                    prev_regime
                )
                st.info(regime_exp)


if __name__ == "__main__":
    main()
