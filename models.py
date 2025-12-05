# models.py - Financial calculation functions (extract from app.py)
"""
Separate module for all financial calculations.
Usage: from models import robust_irr, build_model_and_settle_det, run_montecarlo_vectorized
"""

import numpy as np
import streamlit as st
import logging

logger = logging.getLogger("proforma")

try:
    import numpy_financial as npf
    NPF_AVAILABLE = True
except:
    NPF_AVAILABLE = False

@st.cache_data(ttl=3600)
def robust_irr(cfs):
    """Optimized IRR calculation with caching"""
    cfs = np.array(cfs, dtype=float)
    
    if len(cfs) < 2:
        return float('nan')
    if np.all(cfs >= 0) or np.all(cfs <= 0):
        return float('nan')
    
    if NPF_AVAILABLE:
        try:
            irr = npf.irr(cfs)
            if irr is not None and not np.isnan(irr) and not np.isinf(irr) and -0.99 < irr < 10:
                return float(irr)
        except:
            pass
    
    def npv_and_derivative(r):
        npv = sum(cf / ((1 + r) ** i) for i, cf in enumerate(cfs))
        deriv = sum(-i * cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cfs))
        return npv, deriv
    
    guess = (cfs[-1] / abs(cfs[0])) ** (1 / (len(cfs) - 1)) - 1
    guess = np.clip(guess, -0.5, 2.0)
    
    for _ in range(50):
        try:
            npv_val, deriv_val = npv_and_derivative(guess)
            if abs(npv_val) < 1e-8:
                return guess
            if abs(deriv_val) < 1e-10:
                break
            guess = guess - npv_val / deriv_val
            if guess < -0.99 or guess > 10:
                break
        except:
            break
    
    return float('nan')

@st.cache_data
def annual_payment(loan, rate, amort_years):
    """Calculate annual loan payment"""
    if loan <= 0 or rate < 0 or amort_years < 0:
        return 0.0
    if amort_years == 0:
        return loan * rate
    if NPF_AVAILABLE:
        return float(-npf.pmt(rate, amort_years, loan))
    if rate == 0:
        return loan / amort_years
    x = (1 + rate) ** amort_years
    return loan * rate * x / (x - 1)

def safe_cap(rate, min_cap=0.03, max_cap=0.30):
    """Ensure cap rate within reasonable bounds"""
    return np.clip(rate, min_cap, max_cap)

@st.cache_data
def compute_amort_schedule(loan, rate, amort_years, years):
    """Compute amortization schedule"""
    if loan <= 0 or years <= 0:
        return [], [], [], []
    
    balances, interests, principals, payments = [], [], [], []
    bal = float(loan)
    
    if amort_years == 0:
        for y in range(years):
            interest = bal * rate
            interests.append(interest)
            principals.append(0.0)
            payments.append(interest)
            balances.append(bal)
        return balances, interests, principals, payments
    
    payment = annual_payment(loan, rate, amort_years)
    
    for y in range(years):
        interest = bal * rate
        principal = min(max(payment - interest, 0.0), bal)
        interests.append(interest)
        principals.append(principal)
        payments.append(interest + principal)
        balances.append(bal)
        bal = max(bal - principal, 0.0)
        if bal < 0.01:
            bal = 0.0
    
    return balances, interests, principals, payments

def validate_inputs(inputs):
    """Validate all model inputs"""
    errors = []
    
    if inputs['purchase_price'] <= 0:
        errors.append("Purchase price must be positive")
    if not 0 <= inputs['senior_ltv'] <= 1:
        errors.append("Senior LTV must be between 0-100%")
    if inputs['hold'] < 1:
        errors.append("Hold period must be at least 1 year")
    if inputs['exit_cap'] <= 0:
        errors.append("Exit cap rate must be positive")
    if inputs['senior_ltv'] + inputs['mezz_pct'] >= 1:
        errors.append("Total debt cannot exceed 100% of cost")
    
    return errors
