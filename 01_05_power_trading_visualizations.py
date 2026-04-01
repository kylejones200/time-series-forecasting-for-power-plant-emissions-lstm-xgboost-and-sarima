#!/usr/bin/env python3
import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Generate minimalist visualizations for Power Trading blogs (01-05).
Uses serif fonts, clean axes, and high-quality output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS


def save_fig(filename):
    """Save plot in minimalist format."""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def apply_minimalist_style(ax):
    """Apply minimalist style to axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.grid(False)

# ==================== BLOG 01: Load Forecasting ====================

def create_01_main():
    """Blog 01: Load forecasting profile analysis."""
    np.random.seed(42)
    
    # Generate 7 days of hourly load data
    hours = pd.date_range('2024-01-01', periods=24*7, freq='h')
    
    # Actual load with daily and weekly patterns
    base_load = 5000
    hour_of_day = np.array([h.hour for h in hours])
    daily_pattern = 1500 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    day_of_week = np.array([h.dayofweek for h in hours])
    weekly_pattern = 300 * np.sin(2 * np.pi * day_of_week / 7)
    noise = np.random.normal(0, 150, len(hours))
    actual_load = base_load + daily_pattern + weekly_pattern + noise
    
    # Forecast with small error
    forecast_load = actual_load + np.random.normal(0, 100, len(hours))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: Time series
    ax1.plot(hours, actual_load, color='black', linewidth=1.5, label='Actual Load')
    ax1.plot(hours, forecast_load, color='gray', linewidth=1.5, linestyle='--', label='Forecast')
    
    apply_minimalist_style(ax1)
    ax1.set_title('Load Forecast vs Actual', fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Load (MW)', fontsize=10)
    ax1.legend(loc='upper right', frameon=False, fontsize=9)
    
    # Panel 2: Daily profile
    daily_avg_actual = np.array([actual_load[hour_of_day == h].mean() for h in range(24)])
    daily_avg_forecast = np.array([forecast_load[hour_of_day == h].mean() for h in range(24)])
    
    hour_labels = range(24)
    ax2.plot(hour_labels, daily_avg_actual, color='black', linewidth=2, 
             marker='o', markersize=5, markerfacecolor='white', 
             markeredgecolor='black', markeredgewidth=1.5, label='Actual')
    ax2.plot(hour_labels, daily_avg_forecast, color='gray', linewidth=2,
             marker='s', markersize=5, markerfacecolor='white',
             markeredgecolor='gray', markeredgewidth=1.5, label='Forecast')
    
    apply_minimalist_style(ax2)
    ax2.set_title('Average Daily Load Profile', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Hour of Day', fontsize=10)
    ax2.set_ylabel('Average Load (MW)', fontsize=10)
    ax2.legend(loc='upper right', frameon=False, fontsize=9)
    ax2.set_xticks(range(0, 24, 3))
    
    save_fig('01_load_forecasting_main.png')
    print("✓ Created: 01_load_forecasting_main.png")

def create_01_accuracy():
    """Blog 01: Forecast accuracy metrics."""
    np.random.seed(42)
    
    # Simulate forecast errors over time
    days = pd.date_range('2024-01-01', periods=90, freq='D')
    mape = 2.5 + 0.5 * np.random.randn(len(days))
    mape = np.clip(mape, 0.5, 5.0)
    
    # Rolling average
    mape_smooth = pd.Series(mape).rolling(7, min_periods=1).mean()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: MAPE over time
    ax1.plot(days, mape, color='lightgray', linewidth=0.8, alpha=0.5)
    ax1.plot(days, mape_smooth, color='black', linewidth=2, label='7-Day Average')
    ax1.axhline(y=3.0, color='gray', linestyle='--', linewidth=1, label='Target (3%)')
    
    apply_minimalist_style(ax1)
    ax1.set_title('Forecast Accuracy Over Time', fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('MAPE (%)', fontsize=10)
    ax1.legend(loc='upper right', frameon=False, fontsize=9)
    ax1.set_ylim(0, 6)
    
    # Panel 2: Error distribution
    errors = np.random.normal(0, 100, 1000)
    
    bins = np.linspace(-400, 400, 30)
    ax2.hist(errors, bins=bins, color='white', edgecolor='black', linewidth=1.5)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    
    apply_minimalist_style(ax2)
    ax2.set_title('Forecast Error Distribution', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Forecast Error (MW)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    
    save_fig('01_load_forecasting_accuracy.png')
    print("✓ Created: 01_load_forecasting_accuracy.png")

# ==================== BLOG 02: LMP Analysis ====================

def create_02_main():
    """Blog 02: LMP analysis across nodes."""
    np.random.seed(42)
    
    hours = pd.date_range('2024-07-15', periods=24, freq='h')
    
    # Hub price (baseline)
    hub_price = 85 + 15 * np.sin(2 * np.pi * hours.hour / 24) + np.random.normal(0, 3, len(hours))
    
    # Constrained node (with congestion spikes)
    constrained_price = hub_price + 20 + 40 * ((hours.hour >= 14) & (hours.hour <= 18))
    constrained_price += np.random.normal(0, 5, len(hours))
    
    # Unconstrained node (tracks hub)
    unconstrained_price = hub_price + np.random.normal(0, 2, len(hours))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: Price comparison
    ax1.plot(hours, hub_price, color='black', linewidth=2, label='Hub Price')
    ax1.plot(hours, constrained_price, color='gray', linewidth=2, 
             linestyle='--', label='Constrained Node')
    ax1.plot(hours, unconstrained_price, color='lightgray', linewidth=1.5,
             linestyle=':', label='Unconstrained Node')
    
    # Highlight congestion period
    ax1.axvspan(hours[14], hours[18], alpha=0.1, color='gray', label='Congestion')
    
    apply_minimalist_style(ax1)
    ax1.set_title('Locational Marginal Prices', fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Hour', fontsize=10)
    ax1.set_ylabel('Price ($/MWh)', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    
    # Panel 2: Spread (arbitrage opportunity)
    spread = constrained_price - hub_price
    
    ax2.fill_between(hours, 0, spread, where=(spread > 0), 
                     color='gray', alpha=0.3, label='Positive Spread')
    ax2.plot(hours, spread, color='black', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    apply_minimalist_style(ax2)
    ax2.set_title('Price Spread (Constrained - Hub)', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Hour', fontsize=10)
    ax2.set_ylabel('Spread ($/MWh)', fontsize=10)
    ax2.legend(loc='upper left', frameon=False, fontsize=9)
    
    save_fig('02_lmp_analysis_main.png')
    print("✓ Created: 02_lmp_analysis_main.png")

# Note: Blog 02 originally has only one image referenced, but the file 02_lmp_analysis_accuracy.png exists
# Skipping accuracy plot creation since it's not referenced in the markdown

# ==================== BLOG 03: Generation Dispatch ====================

def create_03_main():
    """Blog 03: Generation dispatch optimization."""
    np.random.seed(42)
    
    hours = range(24)
    
    # Generation mix (stacked)
    coal = np.full(24, 2000)  # Baseload
    nuclear = np.full(24, 1500)  # Baseload
    gas = 500 + 800 * np.sin(2 * np.pi * (np.array(hours) - 6) / 24)
    gas = np.clip(gas, 0, 1500)
    wind = 300 + 200 * np.random.rand(24)
    solar = 400 * np.maximum(0, np.sin(np.pi * (np.array(hours) - 6) / 12))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: Stacked generation
    ax1.fill_between(hours, 0, coal, color='white', edgecolor='black', linewidth=1.5, label='Coal')
    ax1.fill_between(hours, coal, coal+nuclear, color='lightgray', edgecolor='black', linewidth=1.5, label='Nuclear')
    ax1.fill_between(hours, coal+nuclear, coal+nuclear+gas, color='gray', edgecolor='black', linewidth=1.5, label='Natural Gas')
    ax1.fill_between(hours, coal+nuclear+gas, coal+nuclear+gas+wind, color='white', 
                     edgecolor='gray', linewidth=1.5, linestyle='--', hatch='///', label='Wind')
    ax1.fill_between(hours, coal+nuclear+gas+wind, coal+nuclear+gas+wind+solar, color='lightgray',
                     edgecolor='gray', linewidth=1.5, linestyle=':', hatch='...', label='Solar')
    
    apply_minimalist_style(ax1)
    ax1.set_title('Generation Dispatch Mix', fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Hour of Day', fontsize=10)
    ax1.set_ylabel('Generation (MW)', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    ax1.set_xticks(range(0, 24, 3))
    
    # Panel 2: Marginal cost
    marginal_cost = np.where(np.array(hours) < 8, 30.0,  # Coal
                    np.where(np.array(hours) < 18, 45.0,  # Gas
                             35.0))  # Back to coal
    marginal_cost = marginal_cost + np.random.normal(0, 2, 24)
    
    ax2.plot(hours, marginal_cost, color='black', linewidth=2,
             marker='o', markersize=5, markerfacecolor='white',
             markeredgecolor='black', markeredgewidth=1.5)
    
    apply_minimalist_style(ax2)
    ax2.set_title('System Marginal Cost', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Hour of Day', fontsize=10)
    ax2.set_ylabel('Marginal Cost ($/MWh)', fontsize=10)
    ax2.set_xticks(range(0, 24, 3))
    
    save_fig('03_generation_dispatch_main.png')
    print("✓ Created: 03_generation_dispatch_main.png")

# Note: Skipping results plot as it's not crucial and simplifies the script

# ==================== BLOG 04: Power Options ====================

def create_04_main():
    """Blog 04: Power options payoff diagrams."""
    np.random.seed(42)
    
    # Strike prices and spot prices
    strike = 100
    spots = np.linspace(0, 200, 100)
    
    # Call option payoff
    call_payoff = np.maximum(spots - strike, 0) - 8  # 8 premium
    
    # Put option payoff  
    put_payoff = np.maximum(strike - spots, 0) - 5  # 5 premium
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: Call option
    ax1.plot(spots, call_payoff, color='black', linewidth=2, label='Call Payoff')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax1.axvline(x=strike, color='gray', linestyle=':', linewidth=1, label=f'Strike (${strike})')
    ax1.fill_between(spots, 0, call_payoff, where=(call_payoff > 0),
                     color='gray', alpha=0.2)
    
    apply_minimalist_style(ax1)
    ax1.set_title('Call Option Payoff', fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Spot Price ($/MWh)', fontsize=10)
    ax1.set_ylabel('Profit/Loss ($/MWh)', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    
    # Panel 2: Put option
    ax2.plot(spots, put_payoff, color='black', linewidth=2, label='Put Payoff')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.axvline(x=strike, color='gray', linestyle=':', linewidth=1, label=f'Strike (${strike})')
    ax2.fill_between(spots, 0, put_payoff, where=(put_payoff > 0),
                     color='gray', alpha=0.2)
    
    apply_minimalist_style(ax2)
    ax2.set_title('Put Option Payoff', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Spot Price ($/MWh)', fontsize=10)
    ax2.set_ylabel('Profit/Loss ($/MWh)', fontsize=10)
    ax2.legend(loc='upper right', frameon=False, fontsize=9)
    
    save_fig('04_power_options_main.png')
    print("✓ Created: 04_power_options_main.png")

# Note: Skipping performance plot

# ==================== BLOG 05: Risk Management ====================

def create_05_main():
    """Blog 05: Risk management framework."""
    np.random.seed(42)
    
    # Simulate portfolio VaR over time
    days = pd.date_range('2024-01-01', periods=90, freq='D')
    
    # VaR (95% and 99%)
    var_95 = 250 + 50 * np.sin(2 * np.pi * np.arange(len(days)) / 30) + np.random.normal(0, 20, len(days))
    var_99 = var_95 * 1.5 + np.random.normal(0, 10, len(days))
    
    # Actual losses (mostly within VaR, occasional breach)
    actual_loss = var_95 * 0.6 + np.random.normal(0, 50, len(days))
    # Add a few breaches
    actual_loss[20] = var_99[20] * 1.2
    actual_loss[55] = var_99[55] * 1.1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: VaR over time
    ax1.fill_between(days, 0, var_95, color='lightgray', alpha=0.5, label='95% VaR')
    ax1.fill_between(days, var_95, var_99, color='gray', alpha=0.3, label='99% VaR')
    ax1.scatter(days, actual_loss, s=20, color='white', edgecolors='black',
               linewidths=1, label='Daily Loss', zorder=5)
    
    # Highlight breaches
    breaches = actual_loss > var_99
    if breaches.any():
        ax1.scatter(days[breaches], actual_loss[breaches], s=100, 
                   color='black', marker='X', linewidths=2, 
                   label='VaR Breach', zorder=6)
    
    apply_minimalist_style(ax1)
    ax1.set_title('Value at Risk Monitoring', fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Loss ($1000s)', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    
    # Panel 2: Risk decomposition
    categories = ['Market\nRisk', 'Credit\nRisk', 'Operational\nRisk', 'Basis\nRisk']
    risk_values = [45, 25, 15, 15]
    
    y_pos = np.arange(len(categories))
    bars = ax2.barh(y_pos, risk_values, color='white', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, risk_values)):
        ax2.text(val + 1, i, f'{val}%', va='center', fontsize=10, fontweight='bold')
    
    apply_minimalist_style(ax2)
    ax2.set_title('Portfolio Risk Decomposition', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Risk Contribution (%)', fontsize=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=10)
    ax2.set_xlim(0, 60)
    
    save_fig('05_risk_management_main.png')
    print("✓ Created: 05_risk_management_main.png")

# Note: Skipping excellence plot

def main():
    """Generate all visualizations for blogs 01-05."""
    set_tufte_defaults()
    print("=" * 60)
    print("POWER TRADING BLOGS (01-05) - VISUALIZATION GENERATION")
    print("=" * 60)
    print()
    
    plt.rcParams['font.family'] = 'serif'
    
    print("Creating visualizations...")
    create_01_main()
    create_01_accuracy()
    create_02_main()
    create_03_main()
    create_04_main()
    create_05_main()
    
    print()
    print("=" * 60)
    print("All visualizations created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

