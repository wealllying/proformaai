# app.py — Pro Forma AI — Institutional (Fully updated)
# Run: streamlit run app.py

import os
import io
import math
import textwrap
import base64
import requests
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

# Optional PDF/image libs (reportlab) + kaleido usage
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

KALEIDO_AVAILABLE = True
try:
    import kaleido  # noqa: F401
except Exception:
    KALEIDO_AVAILABLE = False

# === Safe numpy_financial (fallback if not installed) ===
try:
    import numpy_financial as npf
except Exception:
    import numpy as _np
    # try to use numpy's financial if present (rare)
    if hasattr(_np, "financial"):
        npf = _np.financial
    else:
        # small fallback implementations
        import math as _math

        def pmt(rate, nper, pv, fv=0, when='end'):
            if rate == 0:
                return -(pv + fv) / nper
            x = 1 + rate
            pow_x = _math.pow(x, nper)
            return -(fv + pv * pow_x) * rate / (pow_x - 1) / (1 + rate * (when == 'begin'))

        def irr(values):
            def npv(r):
                return sum(v / (1 + r) ** i for i, v in enumerate(values))
            r = 0.1
            for _ in range(200):
                f = npv(r)
                if abs(f) < 1e-8:
                    return r
                # finite difference derivative
                d = 1e-6
                df = (npv(r + d) - npv(r - d)) / (2 * d)
                if df == 0:
                    break
                r -= f / df
            return r

        npf = type("npf", (), {"pmt": pmt, "irr": irr})()

# Streamlit UI config
st.set_page_config(page_title="Pro Forma AI — Institutional (Full)", layout="wide")

# ----------------------
# Environment-configurable values (safe defaults provided)
# ----------------------
STRIPE_PK = os.getenv("STRIPE_PK", "").strip()  # publishable key (client side)
APP_URL = os.getenv("APP_URL", os.getenv("DEPLOYED_URL", "http://localhost:8501")).rstrip("/") + "/"
ONE_DEAL_PRICE_ID = os.getenv("ONE_DEAL_PRICE_ID", "price_1SVfkUH2h13vRbN8zuo69kgv")
ANNUAL_PRICE_ID = os.getenv("ANNUAL_PRICE_ID", "price_1SXqY7H2h13vRbN8k0wC7IEx")
# simple token-based unlocks — can be set as env vars VALID_TOKEN_ONE/VALID_TOKEN_ANNUAL
VALID_TOKENS = {
    "one": os.getenv("VALID_TOKEN_ONE", "supersecret-onedeal-2025-x7k9p2m4v8q1r5t3"),
    "annual": os.getenv("VALID_TOKEN_ANNUAL", "supersecret-annual-2025-h4j6k8m1p3q5r7t9")
}

# ----------------------
# Session-state initialization
# ----------------------
if "pending_checkout" not in st.session_state:
    st.session_state.pending_checkout = None  # will hold dict when user clicks a payment option

# ----------------------
# Utility helper functions
# ----------------------
def robust_irr(cfs):
    try:
        irr = npf.irr(cfs)
        if irr is None or (isinstance(irr, float) and (np.isnan(irr) or np.isinf(irr))):
            raise Exception("npf failed")
        return float(irr)
    except Exception:
        def npv(r):
            return sum(cf / ((1 + r) ** i) for i, cf in enumerate(cfs))
        low, high = -0.9999, 10.0
        f_low, f_high = npv(low), npv(high)
        it = 0
        while f_low * f_high > 0 and it < 60:
            high *= 2
            f_high = npv(high)
            it += 1
        if f_low * f_high > 0:
            return float('nan')
        for _ in range(300):
            mid = (low + high) / 2
            f_mid = npv(mid)
            if abs(f_mid) < 1e-8:
                return mid
            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return (low + high) / 2

def annual_payment(loan, rate, amort_years):
    if amort_years == 0:
        return loan * rate
    return float(-npf.pmt(rate, amort_years, loan))

def safe_cap(rate):
    return min(max(rate, 0.03), 0.30)

def compute_amort_schedule(loan, rate, amort_years, years):
    balances = []
    interests = []
    principals = []
    payments = []
    bal = loan
    if amort_years == 0:
        for y in range(1, years + 1):
            interests.append(bal * rate)
            principals.append(0.0)
            payments.append(interests[-1])
            balances.append(bal)
        return balances, interests, principals, payments
    payment = annual_payment(loan, rate, amort_years)
    for y in range(1, years + 1):
        interests.append(bal * rate)
        principal = min(max(payment - interests[-1], 0.0), bal)
        principals.append(principal)
        payments.append(interests[-1] + principal)
        balances.append(bal)
        bal = max(bal - principal, 0.0)
    return balances, interests, principals, payments

# Waterfall helpers (kept close to original logic)
def settle_final_distribution(lp_cf_so_far, gp_cf_so_far, remaining_residual, equity_lp, promote_tiers):
    if remaining_residual <= 0 or promote_tiers is None or len(promote_tiers) == 0:
        lp_share = 0.8
        return remaining_residual * lp_share, remaining_residual * (1 - lp_share)
    lp_add_total = 0.0
    gp_add_total = 0.0
    residual_left = remaining_residual
    lp_so_far = [float(x) for x in lp_cf_so_far]
    for (hurdle, gp_pct) in promote_tiers:
        if residual_left <= 0:
            break
        irr_full = robust_irr(lp_so_far + [residual_left])
        if not np.isnan(irr_full) and irr_full >= hurdle:
            low, high = 0.0, residual_left
            for _ in range(80):
                mid = (low + high) / 2.0
                irr_mid = robust_irr(lp_so_far + [mid])
                if np.isnan(irr_mid):
                    low = mid
                    continue
                if irr_mid >= hurdle:
                    high = mid
                else:
                    low = mid
            X = high
            lp_add_total += X
            residual_left -= X
            gp_take = residual_left * gp_pct
            lp_take = residual_left - gp_take
            lp_add_total += lp_take
            gp_add_total += gp_take
            residual_left = 0.0
            break
        else:
            lp_add_total += residual_left
            residual_left = 0.0
            break
    if residual_left > 0:
        lp_add_total += residual_left * 0.8
        gp_add_total += residual_left * 0.2
    return lp_add_total, gp_add_total

def apply_periodic_waterfall(distributable, lp_roc_remaining, lp_pref_accrued, equity_lp, pref_annual, catchup_pct):
    lp_paid = 0.0
    gp_paid = 0.0
    rem = distributable
    if lp_roc_remaining > 0 and rem > 0:
        pay = min(lp_roc_remaining, rem)
        lp_paid += pay
        lp_roc_remaining -= pay
        rem -= pay
    if lp_pref_accrued > 0 and rem > 0:
        pay = min(lp_pref_accrued, rem)
        lp_paid += pay
        lp_pref_accrued -= pay
        rem -= pay
    if catchup_pct > 0 and rem > 0:
        gp_catch = rem * catchup_pct
        gp_paid += gp_catch
        rem -= gp_catch
    residual_left = rem
    return lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left

# Model builders & Monte Carlo (kept intact)
def build_model_and_settle_det(inputs):
    # Unpack inputs dict to avoid global reliance (safer for testing)
    purchase_price = inputs["purchase_price"]
    closing_costs_pct = inputs["closing_costs_pct"]
    senior_ltv = inputs["senior_ltv"]
    senior_rate = inputs["senior_rate"]
    senior_amort = inputs["senior_amort"]
    senior_io = inputs["senior_io"]
    use_mezz = inputs["use_mezz"]
    mezz_pct = inputs["mezz_pct"]
    gpr_y1 = inputs["gpr_y1"]
    rent_growth = inputs["rent_growth"]
    vacancy = inputs["vacancy"]
    opex_y1 = inputs["opex_y1"]
    opex_growth = inputs["opex_growth"]
    reserves = inputs["reserves"]
    hold = inputs["hold"]
    exit_cap = inputs["exit_cap"]
    selling_costs = inputs["selling_costs"]
    pref_annual = inputs["pref_annual"]
    catchup_pct = inputs["catchup_pct"]
    lp_share_default = inputs["lp_share_default"]
    promote_tiers = inputs["promote_tiers"]

    total_cost = purchase_price * (1 + closing_costs_pct)
    senior_loan = total_cost * senior_ltv
    mezz_loan = total_cost * mezz_pct if (use_mezz and mezz_pct > 0) else 0.0
    equity_total = total_cost - senior_loan - mezz_loan
    equity_lp = equity_total * lp_share_default
    equity_gp = equity_total - equity_lp

    years = []
    lp_cfs = [-equity_lp]
    gp_cfs = [-equity_gp]
    dscr_path = []
    bal = senior_loan
    lp_roc_remaining = equity_lp
    lp_pref_accrued = 0.0
    residual_accumulator = 0.0
    balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)
    noi_at = 0.0
    for y in range(1, hold + 1):
        years.append(f"Year {y}")
        gpr = gpr_y1 * ((1 + rent_growth) ** (y - 1))
        egi = gpr * (1 - vacancy)
        opex = opex_y1 * ((1 + opex_growth) ** (y - 1)) + reserves
        noi = egi - opex
        tax = 0.0
        noi_at = noi - tax
        if senior_amort == 0 or y <= senior_io:
            interest = bal * senior_rate
            principal = 0.0
            payment = interest
        else:
            if y - 1 < len(payments):
                payment = payments[y - 1]
            else:
                payment = annual_payment(senior_loan, senior_rate, senior_amort)
            interest = bal * senior_rate
            principal = min(max(payment - interest, 0.0), bal)
        bal = max(bal - principal, 0.0)
        ds = interest + principal
        dscr = noi_at / ds if ds > 0 else 99.0
        dscr_path.append(dscr)
        op_cf = noi_at - ds
        lp_pref_accrued += equity_lp * pref_annual
        lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left = apply_periodic_waterfall(
            op_cf, lp_roc_remaining, lp_pref_accrued, equity_lp, pref_annual, catchup_pct
        )
        lp_cfs.append(lp_paid)
        gp_cfs.append(gp_paid)
        residual_accumulator += residual_left

    exit_value = noi_at / safe_cap(exit_cap) if safe_cap(exit_cap) > 0 else 0.0
    exit_net = exit_value * (1 - selling_costs)
    exit_reversion = max(exit_net - bal, 0.0)
    final_residual = residual_accumulator + exit_reversion
    lp_add, gp_add = settle_final_distribution(lp_cfs, gp_cfs, final_residual, equity_lp, promote_tiers)
    lp_cfs[-1] += lp_add
    gp_cfs[-1] += gp_add

    cf_table = pd.DataFrame({
        "Period": ["Year 0"] + years,
        "LP CF": [lp_cfs[0]] + lp_cfs[1:],
        "GP CF": [gp_cfs[0]] + gp_cfs[1:],
    })

    return {
        "lp_cfs": lp_cfs,
        "gp_cfs": gp_cfs,
        "cf_table": cf_table,
        "dscr_path": dscr_path,
        "exit_value": exit_value,
        "exit_reversion": exit_reversion
    }

def run_montecarlo(inputs, n_sims):
    purchase_price = inputs["purchase_price"]
    closing_costs_pct = inputs["closing_costs_pct"]
    senior_ltv = inputs["senior_ltv"]
    senior_rate = inputs["senior_rate"]
    senior_amort = inputs["senior_amort"]
    senior_io = inputs["senior_io"]
    use_mezz = inputs["use_mezz"]
    mezz_pct = inputs["mezz_pct"]
    gpr_y1 = inputs["gpr_y1"]
    rent_growth = inputs["rent_growth"]
    vacancy = inputs["vacancy"]
    opex_y1 = inputs["opex_y1"]
    opex_growth = inputs["opex_growth"]
    reserves = inputs["reserves"]
    hold = inputs["hold"]
    exit_cap = inputs["exit_cap"]
    selling_costs = inputs["selling_costs"]
    pref_annual = inputs["pref_annual"]
    catchup_pct = inputs["catchup_pct"]
    lp_share_default = inputs["lp_share_default"]
    promote_tiers = inputs["promote_tiers"]
    sigma_rent = inputs["sigma_rent"]
    sigma_opex = inputs["sigma_opex"]
    sigma_cap = inputs["sigma_cap"]
    corr = inputs["corr"]

    total_cost = purchase_price * (1 + closing_costs_pct)
    senior_loan = total_cost * senior_ltv
    mezz_loan_amt = total_cost * mezz_pct if (use_mezz and mezz_pct > 0) else 0.0
    equity_total = total_cost - senior_loan - mezz_loan_amt
    equity_lp = equity_total * lp_share_default
    equity_gp = equity_total - equity_lp

    cov = np.diag([sigma_rent**2, sigma_opex**2, sigma_cap**2])
    cov = np.sqrt(cov) @ corr @ np.sqrt(cov)
    try:
        L = np.linalg.cholesky(cov)
    except Exception:
        L = np.diag([sigma_rent, sigma_opex, sigma_cap])

    lp_irrs = []
    dscr_breach_count = 0

    for i in range(int(n_sims)):
        z = np.random.normal(size=3)
        shocks = L @ z
        rent_shock = 1.0 + shocks[0]
        opex_shock = 1.0 + shocks[1]
        cap_shock = shocks[2]

        bal = senior_loan
        lp_cf_sim = [-equity_lp]
        gp_cf_sim = [-equity_gp]
        lp_roc_remaining = equity_lp
        lp_pref_accrued = 0.0
        residual_acc = 0.0
        dscr_vals = []
        balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)

        for y in range(1, hold + 1):
            gpr = gpr_y1 * ((1 + rent_growth) ** (y - 1)) * rent_shock
            v = float(np.clip(vacancy + np.random.normal(0, 0.01), 0.0, 0.9))
            egi = gpr * (1 - v)
            opex = opex_y1 * ((1 + opex_growth) ** (y - 1)) * opex_shock + reserves
            noi = egi - opex
            tax = 0.0
            noi_at = noi - tax
            if senior_amort == 0 or y <= senior_io:
                interest = bal * senior_rate
                principal = 0.0
                payment = interest
            else:
                if y - 1 < len(payments):
                    payment = payments[y - 1]
                else:
                    payment = annual_payment(senior_loan, senior_rate, senior_amort)
                interest = bal * senior_rate
                principal = min(max(payment - interest, 0.0), bal)
            bal = max(bal - principal, 0.0)
            ds = interest + principal
            dscr_val = noi_at / ds if ds > 0 else 99.0
            dscr_vals.append(dscr_val)
            op_cf = noi_at - ds
            lp_pref_accrued += equity_lp * pref_annual
            lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left = apply_periodic_waterfall(
                op_cf, lp_roc_remaining, lp_pref_accrued, equity_lp, pref_annual, catchup_pct
            )
            lp_cf_sim.append(lp_paid)
            gp_cf_sim.append(gp_paid)
            residual_acc += residual_left

        cap_sim = safe_cap(exit_cap + cap_shock)
        exit_value = noi_at / cap_sim if cap_sim > 0 else 0.0
        exit_net = exit_value * (1 - selling_costs)
        exit_reversion = max(exit_net - bal, 0.0)
        final_residual = residual_acc + exit_reversion
        lp_add, gp_add = settle_final_distribution(lp_cf_sim, gp_cf_sim, final_residual, equity_lp, promote_tiers)
        lp_cf_sim[-1] += lp_add
        gp_cf_sim[-1] += gp_add
        irr_sim = robust_irr(lp_cf_sim)
        if not np.isnan(irr_sim) and irr_sim > -1:
            lp_irrs.append(irr_sim)
        if any(d < 1.2 for d in dscr_vals):
            dscr_breach_count += 1

    return np.array(lp_irrs), dscr_breach_count

# ----------------------
# UI & Paywall / Stripe checkout (client-side)
# ----------------------

# Read plan/token from URL using modern API
qp = st.query_params
plan = qp.get("plan", None)
token = qp.get("token", None)

# If token present and valid => unlock (user came from a Stripe success URL)
if plan in VALID_TOKENS and token == VALID_TOKENS.get(plan):
    st.success("Access unlocked — thank you.")
    # do not hide UI; proceed to app normally

# If not unlocked, show paywall UI
if plan not in VALID_TOKENS or token != VALID_TOKENS.get(plan):
    st.title("Pro Forma AI — Institutional Access Required")
    st.markdown("### Unlock Full Model Instantly")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("One Deal — $999", type="primary", use_container_width=True, key="one"):
            st.session_state.pending_checkout = {
                "price": ONE_DEAL_PRICE_ID,
                "success_url": f"{APP_URL}?plan=one&token={VALID_TOKENS['one']}",
                "cancel_url": APP_URL
            }
            # re-run to hit the pending_checkout handler below
            st.rerun()

    with col2:
        if st.button("Unlimited — $99,000/year", type="primary", use_container_width=True, key="annual"):
            st.session_state.pending_checkout = {
                "price": ANNUAL_PRICE_ID,
                "success_url": f"{APP_URL}?plan=annual&token={VALID_TOKENS['annual']}",
                "cancel_url": APP_URL
            }
            st.rerun()

    # If pending_checkout exists, create JS that calls Stripe Checkout (client-side).
    if st.session_state.pending_checkout:
        if not STRIPE_PK:
            st.error("Stripe not configured. Set STRIPE_PK environment variable to your publishable key.")
            st.stop()

        checkout = st.session_state.pending_checkout
        js = f"""
        <script src="https://js.stripe.com/v3/"></script>
        <script>
        (function() {{
            const stripe = Stripe("{STRIPE_PK}");
            stripe.redirectToCheckout({{
                lineItems: [{{ price: "{checkout['price']}", quantity: 1 }}],
                mode: 'payment',
                successUrl: "{checkout['success_url']}",
                cancelUrl: "{checkout['cancel_url']}"
            }}).then(function(result) {{
                if (result.error) {{
                    const e = document.createElement('div');
                    e.style.padding = '12px';
                    e.style.background = '#fee';
                    e.style.border = '1px solid #f99';
                    e.innerText = result.error.message || 'Stripe checkout failed';
                    document.body.appendChild(e);
                }}
            }});
        }})();
        </script>
        """
        # height > 0 so Streamlit allows JS to run
        st.components.v1.html(js, height=240)
        # clear pending to avoid repeat
        st.session_state.pending_checkout = None
        st.stop()

# ----------------------
# Main app UI (unlocked)
# ----------------------
st.header("Pro Forma AI — Institutional Model (Full)")

# Sidebar inputs (same as your previous app but passed as inputs dict to model)
with st.sidebar:
    st.header("Acquisition & Capital Stack")
    purchase_price = st.number_input("Purchase Price ($)", value=100_000_000, step=1_000_000)
    closing_costs_pct = st.slider("Closing Costs %", 0.0, 5.0, 1.5) / 100.0
    # Debt
    st.subheader("Senior Loan")
    senior_ltv = st.slider("Senior LTV %", 0.0, 90.0, 60.0) / 100.0
    senior_rate = st.slider("Senior Interest Rate (annual %)", 0.5, 12.0, 5.5, 0.05) / 100.0
    senior_amort = st.number_input("Senior Amortization (years, 0=IO)", 0, 30, 25)
    senior_term = st.number_input("Senior Term (years)", 1, 30, 25)
    senior_io = st.number_input("Senior IO Period (years)", 0, min(10, senior_term), 0)
    st.subheader("Mezz / Subordinate (optional)")
    use_mezz = st.checkbox("Include Mezz", value=False)
    if use_mezz:
        mezz_pct = st.slider("Mezz % of Cost", 0.0, 40.0, 10.0) / 100.0
        mezz_rate = st.slider("Mezz Rate %", 0.0, 20.0, 10.0) / 100.0
        mezz_term = st.number_input("Mezz Term (years)", 1, 10, 5)
    else:
        mezz_pct = 0.0
        mezz_rate = 0.0
        mezz_term = 0
    st.subheader("Equity")
    total_equity = purchase_price * (1 + closing_costs_pct) * (1 - senior_ltv - (mezz_pct if use_mezz else 0.0))
    st.markdown(f"Estimated equity required: **${total_equity:,.0f}**")
    lp_share_default = st.slider("LP % of Equity (rest is GP)", 50, 95, 80) / 100.0

    st.header("Operating Assumptions")
    gpr_y1 = st.number_input("Year 1 GPR ($)", value=12_000_000, step=10000)
    rent_growth = st.slider("Rent Growth %", 0.0, 8.0, 3.0, 0.1) / 100.0
    vacancy = st.slider("Vacancy %", 0.0, 20.0, 5.0, 0.1) / 100.0
    opex_y1 = st.number_input("Year 1 OpEx ($)", value=3_600_000, step=10000)
    opex_growth = st.slider("OpEx Growth %", 0.0, 8.0, 2.5, 0.1) / 100.0
    reserves = st.number_input("Annual Reserves/CapEx ($)", value=400_000, step=10000)

    st.header("Exit & Waterfall")
    hold = st.slider("Hold Period (years)", 1, 10, 5)
    exit_cap = st.slider("Exit Cap %", 3.0, 12.0, 5.5, 0.05) / 100.0
    selling_costs = st.slider("Selling Costs %", 0.0, 8.0, 5.0) / 100.0
    pref_annual = st.slider("Preferred Return (LP annual %)", 0.0, 15.0, 8.0, 0.1) / 100.0
    catchup_pct = st.slider("Catch-up % to GP after pref", 0.0, 100.0, 0.0, 1.0) / 100.0

    st.markdown("### Promote tiers (IRR-hurdle driven)")
    use_promote = st.checkbox("Enable promote tiers", True)
    if use_promote:
        tier1_hurdle = st.number_input("Tier1 Hurdle IRR (%)", value=12.0, step=0.5) / 100.0
        tier1_gp = st.number_input("Tier1 GP % of residual", value=30.0, step=1.0) / 100.0
        tier2_hurdle = st.number_input("Tier2 Hurdle IRR (%)", value=20.0, step=0.5) / 100.0
        tier2_gp = st.number_input("Tier2 GP % of residual", value=50.0, step=1.0) / 100.0
        promote_tiers = [(tier1_hurdle, tier1_gp), (tier2_hurdle, tier2_gp)]
    else:
        promote_tiers = None

    st.header("Monte Carlo & Performance")
    n_sims = st.number_input("Monte Carlo sims", min_value=500, max_value=20000, value=5000, step=500)
    sigma_rent = st.slider("Rent vol (σ)", 0.0, 0.25, 0.02, 0.005)
    sigma_opex = st.slider("OpEx vol (σ)", 0.0, 0.25, 0.015, 0.005)
    sigma_cap = st.slider("Cap vol (σ)", 0.0, 0.10, 0.004, 0.001)
    corr = np.array([[1.0, 0.2, -0.4],
                     [0.2, 1.0, -0.2],
                     [-0.4, -0.2, 1.0]])

    st.header("Report branding")
    logo_mode = st.selectbox("Logo input", options=["None", "Upload file (PNG/JPG)", "Provide image URL"])
    logo_file = None
    logo_url = None
    if logo_mode == "Upload file (PNG/JPG)":
        logo_file = st.file_uploader("Upload logo", type=["png", "jpg", "jpeg"])
    elif logo_mode == "Provide image URL":
        logo_url = st.text_input("Logo image URL (https://...)")

# Pack inputs into dict for functions
inputs = {
    "purchase_price": purchase_price,
    "closing_costs_pct": closing_costs_pct,
    "senior_ltv": senior_ltv,
    "senior_rate": senior_rate,
    "senior_amort": senior_amort,
    "senior_io": senior_io,
    "use_mezz": use_mezz,
    "mezz_pct": mezz_pct,
    "gpr_y1": gpr_y1,
    "rent_growth": rent_growth,
    "vacancy": vacancy,
    "opex_y1": opex_y1,
    "opex_growth": opex_growth,
    "reserves": reserves,
    "hold": hold,
    "exit_cap": exit_cap,
    "selling_costs": selling_costs,
    "pref_annual": pref_annual,
    "catchup_pct": catchup_pct,
    "lp_share_default": lp_share_default,
    "promote_tiers": promote_tiers,
    "n_sims": n_sims,
    "sigma_rent": sigma_rent,
    "sigma_opex": sigma_opex,
    "sigma_cap": sigma_cap,
    "corr": corr
}

# ----------------------
# Run model & Monte Carlo when button pressed
# ----------------------
if st.button("Run Full Institutional Model (Deterministic + Monte Carlo)"):
    with st.spinner("Running deterministic build..."):
        det = build_model_and_settle_det(inputs)
        lp_cfs = det["lp_cfs"]
        gp_cfs = det["gp_cfs"]
        cf_table = det["cf_table"]
        exit_value = det["exit_value"]
        exit_reversion = det["exit_reversion"]
        dscr_path = det["dscr_path"]
        lp_irr = robust_irr(lp_cfs)
        st.success("Deterministic build complete")

    st.subheader("Deterministic Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("LP IRR (det)", f"{lp_irr:.2%}" if not math.isnan(lp_irr) else "N/A")
    c2.metric("Min DSCR", f"{min(dscr_path):.2f}x" if dscr_path else "N/A")
    c3.metric("Exit Value (net)", f"${exit_value*(1-selling_costs):,.0f}")

    st.markdown("### Deterministic Cashflow Table (LP)")
    st.dataframe(cf_table)
    csv_bytes = cf_table.to_csv(index=False).encode()
    st.download_button("Download deterministic CF CSV", csv_bytes, "deterministic_cf.csv", "text/csv")

    st.info(f"Running Monte Carlo with {n_sims} sims — this may take a bit.")
    with st.spinner("Running Monte Carlo..."):
        irrs, breaches = run_montecarlo(inputs, int(n_sims))

    if irrs.size == 0:
        st.error("Monte Carlo produced no valid IRRs. Check inputs.")
        p5 = p50 = p95 = None
        fig_monte = None
    else:
        p5, p50, p95 = np.percentile(irrs, [5, 50, 95])
        st.subheader("Monte Carlo Results (LP IRR)")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("P5", f"{p5:.2%}")
        cc2.metric("P50", f"{p50:.2%}")
        cc3.metric("P95", f"{p95:.2%}")
        st.metric("Probability DSCR < 1.2", f"{breaches / max(1, int(n_sims)):.1%}")

        # Monte Carlo histogram (kept in same view)
        fig_monte = px.histogram(irrs * 100, nbins=80, title="LP IRR Distribution (Monte Carlo)")
        fig_monte.add_vline(x=p50 * 100, line_color="white", line_width=3)
        st.plotly_chart(fig_monte, use_container_width=True)

    # Deterministic waterfall chart
    try:
        op_sum = sum([x for x in det['lp_cfs'][1:-1]]) if len(det['lp_cfs']) > 2 else 0
        wf = go.Figure(go.Waterfall(
            x=["Equity In", "Operating CF", "Exit/Residual"],
            y=[-abs(det['lp_cfs'][0]), op_sum, det['lp_cfs'][-1]],
            connector={"line": {"color": "white"}}
        ))
        wf.update_layout(title="Deterministic LP Waterfall", template="plotly_white")
        fig_waterfall = wf
        st.plotly_chart(fig_waterfall, use_container_width=True)
    except Exception:
        fig_waterfall = None

    # Save Monte Carlo IRRs CSV
    if irrs.size > 0:
        buf = io.StringIO()
        pd.DataFrame({"LP_IRR": irrs}).to_csv(buf, index=False)
        st.download_button("Download Monte Carlo IRRs (CSV)", buf.getvalue().encode(), "mc_irrs.csv", "text/csv")

    # Generate PDF memo (includes charts if available)
    def figure_to_png_bytes(fig):
        try:
            # prefer kaleido if fig has to_image
            return fig.to_image(format="png")
        except Exception:
            return None

    def fetch_logo_blob(logo_file, logo_url):
        if logo_file:
            try:
                return logo_file.read(), "uploaded"
            except Exception:
                return None
        if logo_url:
            try:
                r = requests.get(logo_url, timeout=6)
                if r.status_code == 200:
                    return r.content, "url"
            except Exception:
                return None
        return None

    logo_blob = fetch_logo_blob(logo_file if 'logo_file' in locals() else None,
                                logo_url if 'logo_url' in locals() else None)

    # Try to build PDF (best effort)
    try:
        pdf_bytes = None
        filename = None
        if REPORTLAB_AVAILABLE:
            # reuse the deterministic memo builder from earlier logic (concise here)
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=letter,
                                    rightMargin=36, leftMargin=36,
                                    topMargin=72, bottomMargin=36)
            story = []
            styles = getSampleStyleSheet()
            normal = styles["Normal"]
            heading = styles["Heading1"]
            story.append(Paragraph("<b>Pro Forma AI — Institutional Memorandum</b>", heading))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Date: {datetime.today().strftime('%B %d, %Y')}", normal))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            exec_text = textwrap.fill(
                f"This memorandum summarizes deterministic and Monte Carlo analysis for a proposed acquisition at a purchase price of ${purchase_price:,.0f}.",
                200)
            story.append(Paragraph(exec_text, normal))
            story.append(Spacer(1, 12))
            # Add key bullets
            story.append(PageBreak())
            story.append(Paragraph("Deterministic Cashflows (LP & GP)", styles['Heading2']))
            df = det['cf_table'].copy()
            df_formatted = df.copy()
            for col in ["LP CF", "GP CF"]:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:,.0f}")
            parts = [df_formatted]  # simple single-table approach
            for part in parts:
                data = [list(part.columns)] + part.values.tolist()
                t = Table(data, repeatRows=1, hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                    ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
                    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
            story.append(PageBreak())
            story.append(Paragraph("Monte Carlo Results (LP IRR)", styles['Heading2']))
            try:
                png_hist = figure_to_png_bytes(fig_monte) if 'fig_monte' in locals() and fig_monte is not None else None
                png_wf = figure_to_png_bytes(fig_waterfall) if 'fig_waterfall' in locals() and fig_waterfall is not None else None
                if png_hist:
                    story.append(Paragraph("LP IRR Distribution", styles['Heading3']))
                    story.append(Spacer(1, 6))
                    story.append(ImageReader(BytesIO(png_hist)))
                    story.append(Spacer(1, 6))
                if png_wf:
                    story.append(Paragraph("Deterministic LP Waterfall", styles['Heading3']))
                    story.append(Spacer(1, 6))
                    story.append(ImageReader(BytesIO(png_wf)))
                    story.append(Spacer(1, 6))
            except Exception:
                pass
            story.append(PageBreak())
            story.append(Paragraph("Appendix: Inputs", styles['Heading2']))
            doc.build(story)
            buf.seek(0)
            pdf_bytes = buf.getvalue()
            filename = f"Pro_Forma_AI_Memo_{datetime.today().strftime('%Y%m%d')}.pdf"
        else:
            # fallback: simple text summary
            summary_text = (f"Pro Forma AI Summary\nDate: {datetime.today().strftime('%B %d, %Y')}\n"
                            f"LP IRR (det): {robust_irr(det['lp_cfs']):.2%}\n"
                            f"P50 (MC): {p50:.2% if 'p50' in locals() else 'N/A'}\n"
                            f"P95 (MC): {p95:.2% if 'p95' in locals() else 'N/A'}\n")
            pdf_bytes = summary_text.encode()
            filename = f"Pro_Forma_AI_Summary_{datetime.today().strftime('%Y%m%d')}.txt"

        # Provide download
        st.success("PDF memo ready")
        st.download_button("Download Full PDF Memo", pdf_bytes, filename, "application/octet-stream")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        # provide summary fallback
        summary_text = (f"Deterministic LP IRR: {robust_irr(det['lp_cfs']):.2%}\n"
                        f"P50 (MC): {p50 if 'p50' in locals() else 'N/A'}\n"
                        f"P95 (MC): {p95 if 'p95' in locals() else 'N/A'}\n")
        st.download_button("Download Summary (TXT)", summary_text.encode(), "proforma_summary.txt", "text/plain")

st.markdown("---")
st.info("This is an institutional-grade model with multi-tier promote settlement, Monte Carlo analysis, and automated PDF reporting. Adjust inputs and rerun.")

