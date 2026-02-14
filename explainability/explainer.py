"""
Explainer
Provides human-readable explanations for portfolio decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

from utils.constants import REGIME_LABELS, TARGET_VOLATILITY, MAX_DRAWDOWN_THRESHOLD
from utils.helpers import calculate_drawdown


class Explainer:
    """
    Explainability engine for portfolio decisions.
    
    Generates rule-based, deterministic explanations for:
    - Regime detection
    - Allocation changes
    - Risk management actions
    """
    
    def __init__(self):
        """Initialize Explainer."""
        self.explanation_log = []
        
    def explain_regime(
        self,
        regime_id: int,
        features: pd.Series,
        prev_regime_id: int = None
    ) -> str:
        """
        Explain why a specific regime was detected.
        
        Args:
            regime_id: Current regime ID
            features: Feature values for this date
            prev_regime_id: Previous regime ID (optional)
            
        Returns:
            Human-readable explanation
        """
        regime_name = REGIME_LABELS[regime_id]
        
        # Extract key features
        vol = features.get('portfolio_volatility', 0)
        drawdown = features.get('market_drawdown', 0)
        spy_return = features.get('SPY_rolling_return', 0)
        
        # Build explanation
        explanation_parts = [f"Regime detected: {regime_name}"]
        
        if regime_id == 3:  # Crash
            explanation_parts.append(f"Market drawdown of {abs(drawdown):.1%} indicates severe market stress.")
            explanation_parts.append("Crash regime triggered.")
            
        elif regime_id == 2:  # High Volatility
            explanation_parts.append(f"Portfolio volatility at {vol:.1%} exceeds high threshold.")
            explanation_parts.append("Elevated uncertainty in markets.")
            
        elif regime_id == 0:  # Trending Up
            explanation_parts.append(f"Positive momentum: SPY 20-day return at {spy_return:.1%}.")
            explanation_parts.append(f"Volatility moderate at {vol:.1%}.")
            explanation_parts.append("Bullish market conditions.")
            
        elif regime_id == 1:  # Trending Down
            explanation_parts.append(f"Negative momentum: SPY 20-day return at {spy_return:.1%}.")
            explanation_parts.append("Bearish market conditions.")
        
        # Regime change detection
        if prev_regime_id is not None and prev_regime_id != regime_id:
            prev_name = REGIME_LABELS[prev_regime_id]
            explanation_parts.append(f"→ Regime change from '{prev_name}' to '{regime_name}'")
        
        return " ".join(explanation_parts)
    
    def explain_allocation(
        self,
        regime_id: int,
        allocations: Dict[str, float],
        prev_allocations: Dict[str, float] = None
    ) -> str:
        """
        Explain allocation decisions.
        
        Args:
            regime_id: Current regime ID
            allocations: Current allocations
            prev_allocations: Previous allocations (optional)
            
        Returns:
            Human-readable explanation
        """
        regime_name = REGIME_LABELS[regime_id]
        
        explanation_parts = [f"Allocation strategy for {regime_name} regime:"]
        
        # Explain regime-specific allocation logic
        if regime_id == 0:  # Trending Up
            explanation_parts.append("Favoring equities (SPY, QQQ) due to positive market momentum.")
            explanation_parts.append("Reducing defensive assets (TLT, GLD).")
            
        elif regime_id == 1:  # Trending Down
            explanation_parts.append("Balanced shift toward defensive assets.")
            explanation_parts.append("Increasing bonds and gold allocation.")
            
        elif regime_id == 2:  # High Volatility
            explanation_parts.append("Reducing equity exposure due to elevated volatility.")
            explanation_parts.append("Increasing defensive assets (bonds, gold) by 40%.")
            
        elif regime_id == 3:  # Crash
            explanation_parts.append("Maximum defensive positioning activated.")
            explanation_parts.append("Reducing equities by 60%, prioritizing safe havens (TLT, GLD).")
        
        # Show current allocation
        alloc_str = ", ".join([f"{k}: {v:.1%}" for k, v in allocations.items()])
        explanation_parts.append(f"Current allocation: {alloc_str}")
        
        # Detect significant changes
        if prev_allocations is not None:
            changes = []
            for asset in allocations.keys():
                change = allocations[asset] - prev_allocations.get(asset, 0)
                if abs(change) > 0.05:  # 5% threshold
                    direction = "increased" if change > 0 else "decreased"
                    changes.append(f"{asset} {direction} by {abs(change):.1%}")
            
            if changes:
                explanation_parts.append(f"Changes: {', '.join(changes)}")
        
        return " ".join(explanation_parts)
    
    def explain_risk_action(
        self,
        risk_metrics: Dict[str, float],
        exposure_scalar: float,
        risk_signal: str
    ) -> str:
        """
        Explain risk management actions.
        
        Args:
            risk_metrics: Current risk metrics
            exposure_scalar: Risk adjustment scalar
            risk_signal: Risk signal level
            
        Returns:
            Human-readable explanation
        """
        explanation_parts = [f"Risk Management Action: {risk_signal}"]
        
        current_vol = risk_metrics.get('Current Volatility', 0)
        current_dd = risk_metrics.get('Current Drawdown', 0)
        target_vol = risk_metrics.get('Vol Target', TARGET_VOLATILITY)
        
        # Explain volatility targeting
        if current_vol > target_vol * 1.2:
            explanation_parts.append(
                f"Portfolio volatility ({current_vol:.1%}) exceeds target ({target_vol:.1%})."
            )
            explanation_parts.append("Scaling down positions to reduce volatility.")
            
        elif current_vol < target_vol * 0.8:
            explanation_parts.append(
                f"Portfolio volatility ({current_vol:.1%}) below target ({target_vol:.1%})."
            )
            explanation_parts.append("Scaling up positions to achieve target volatility.")
        
        # Explain drawdown control
        if current_dd > MAX_DRAWDOWN_THRESHOLD:
            explanation_parts.append(
                f"Current drawdown ({current_dd:.1%}) exceeds threshold ({MAX_DRAWDOWN_THRESHOLD:.1%})."
            )
            explanation_parts.append("Reducing exposure to limit further losses.")
        
        # Explain exposure adjustment
        if exposure_scalar < 0.8:
            explanation_parts.append(f"Reducing overall exposure by {(1 - exposure_scalar) * 100:.0f}%.")
        elif exposure_scalar > 1.2:
            explanation_parts.append(f"Increasing overall exposure by {(exposure_scalar - 1) * 100:.0f}%.")
        else:
            explanation_parts.append("Maintaining current exposure levels.")
        
        return " ".join(explanation_parts)
    
    def generate_daily_explanation(
        self,
        date: pd.Timestamp,
        regime_id: int,
        prev_regime_id: int,
        allocations: Dict[str, float],
        prev_allocations: Dict[str, float],
        features: pd.Series,
        risk_metrics: Dict[str, float] = None,
        exposure_scalar: float = 1.0,
        risk_signal: str = 'NORMAL'
    ) -> str:
        """
        Generate comprehensive daily explanation.
        
        Args:
            date: Current date
            regime_id: Current regime
            prev_regime_id: Previous regime
            allocations: Current allocations
            prev_allocations: Previous allocations
            features: Feature values
            risk_metrics: Risk metrics (optional)
            exposure_scalar: Risk adjustment scalar
            risk_signal: Risk signal level
            
        Returns:
            Complete daily explanation
        """
        explanation = [
            f"\n{'='*70}",
            f"PORTFOLIO DECISION EXPLANATION - {date.strftime('%Y-%m-%d')}",
            f"{'='*70}\n"
        ]
        
        # 1. Regime explanation
        regime_explanation = self.explain_regime(regime_id, features, prev_regime_id)
        explanation.append(f"[REGIME DETECTION]\n{regime_explanation}\n")
        
        # 2. Allocation explanation
        allocation_explanation = self.explain_allocation(regime_id, allocations, prev_allocations)
        explanation.append(f"[ALLOCATION DECISION]\n{allocation_explanation}\n")
        
        # 3. Risk management explanation (if applicable)
        if risk_metrics is not None:
            risk_explanation = self.explain_risk_action(risk_metrics, exposure_scalar, risk_signal)
            explanation.append(f"[RISK MANAGEMENT]\n{risk_explanation}\n")
        
        explanation.append(f"{'='*70}\n")
        
        full_explanation = "\n".join(explanation)
        
        # Log the explanation
        self.explanation_log.append({
            'date': date,
            'regime': REGIME_LABELS[regime_id],
            'explanation': full_explanation
        })
        
        return full_explanation
    
    def generate_summary_explanation(
        self,
        regimes: pd.Series,
        allocations: pd.DataFrame,
        risk_adjustments: pd.Series = None,
        n_samples: int = 5
    ) -> List[str]:
        """
        Generate sample explanations for key dates.
        
        Args:
            regimes: Series of regimes
            allocations: DataFrame of allocations
            risk_adjustments: Series of risk adjustments (optional)
            n_samples: Number of sample explanations
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Select interesting dates: regime changes
        regime_changes = regimes[regimes != regimes.shift(1)].index
        
        # Select sample dates
        if len(regime_changes) > n_samples:
            sample_dates = regime_changes[:n_samples]
        else:
            sample_dates = regimes.index[::len(regimes) // n_samples][:n_samples]
        
        print("\n" + "="*70)
        print("SAMPLE PORTFOLIO DECISION EXPLANATIONS")
        print("="*70)
        
        for i, date in enumerate(sample_dates):
            if i == 0:
                prev_regime = regimes.iloc[0]
                prev_alloc = allocations.iloc[0].to_dict()
            else:
                idx = regimes.index.get_loc(date)
                prev_regime = regimes.iloc[idx - 1]
                prev_alloc = allocations.iloc[idx - 1].to_dict()
            
            regime = regimes.loc[date]
            alloc = allocations.loc[date].to_dict()
            
            # Create dummy features for explanation
            features = pd.Series({
                'portfolio_volatility': 0.15 if regime in [2, 3] else 0.10,
                'market_drawdown': -0.12 if regime == 3 else -0.05,
                'SPY_rolling_return': 0.05 if regime == 0 else -0.03
            })
            
            # Generate explanation
            explanation = self.generate_daily_explanation(
                date=date,
                regime_id=int(regime),
                prev_regime_id=int(prev_regime),
                allocations=alloc,
                prev_allocations=prev_alloc,
                features=features
            )
            
            explanations.append(explanation)
            print(explanation)
        
        return explanations
    
    def save_explanations(self, filepath: str = 'explanations.txt'):
        """
        Save all logged explanations to file.
        
        Args:
            filepath: Path to save explanations
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in self.explanation_log:
                f.write(entry['explanation'])
                f.write("\n\n")
        
        print(f"Explanations saved to: {filepath}")
