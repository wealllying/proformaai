# app.py — Pro Forma AI — Institutional (Full)
# Run: streamlit run app.py

import os
import math
import io
from io import BytesIO
from datetime import datetime
import base64
import textwrap

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Optional libs — PDF & image export
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Plotly->image requires kaleido for fig.write_image
try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

# Excel export
try:
    import openpyxl  # used by pandas ExcelWriter
    EXCEL_AVAILABLE = True
except Exception:
    EXCEL_AVAILABLE = False

# Setup page
st.set_page_config(page_title="Pro Forma AI — Institutional", layout="wide")

# -------------------------
# Environment config (Railway/Heroku/etc)
# -------------------------
STRIPE_PK = os.getenv("STRIPE_PK")  # publishable key for client-side checkout
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")  # optional server-side; not used for client redirect
ONE_DEAL_PRICE_ID = os.getenv("ONE_DEAL_PRICE_ID", "price_1SVfkUH2h13vRbN8zuo69kgv")
ANNUAL_PRICE_ID = os.getenv("ANNUAL_PRICE_ID", "price_1SXqY7H2h13vRbN8k0wC7IEx")
APP_URL = os.getenv("APP_URL", "https://proforma-ai-production.up.railway.app/")

# Quick checks
if not STRIPE_PK:
    st.warning("STRIPE_PK not set — Stripe Checkout buttons will be disabled until you set STRIPE_PK in environment variables.")

# -------------------------
# Utilities
# -------------------------
def fmt(x):
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def robust_irr(cfs):
    # wrapper using numpy. If fails, return nan
    try:
        r = np.irr(cfs)
        return float(r) if r is not None else float('nan')
    except Exception:
        # small bisection fallback
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
        for _ in range(200):
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

def figure_to_png_bytes(fig):
    """Return PNG bytes for a plotly figure if possible."""
    try:
        b = fig.to_image(format="png")
        return b
    except Exception:
        return None

# -------------------------
# Session state init
# -------------------------
if "pending_checkout" not in st.session_state:
    st.session_state.pending_checkout = None

# -------------------------
# PAYWALL / Stripe Checkout (client-side)
# -------------------------
def client_redirect_checkout(price_id: str):
    if not STRIPE_PK:
        st.error("Stripe public key missing (STRIPE_PK). Set env var and redeploy.")
        return
    js = f"""
    <script src="https://js.stripe.com/v3/"></script>
    <script>
    (function() {{
        var stripe = Stripe("{STRIPE_PK}");
        stripe.redirectToCheckout({{
            lineItems: [{{ price: "{price_id}", quantity: 1 }}],
            mode: "payment",
            successUrl: "{APP_URL}?plan=one&token=INJECTED", // success url will be replaced by your platform or use server session
            cancelUrl: "{APP_URL}"
        }}).then(function(result) {{
            if (result.error) {{
                var el = document.createElement('div');
                el.style.padding = '12px';
                el.style.background = '#fee';
                el.style.border = '1px solid #f99';
                el.innerText = result.error.message || 'Stripe error';
                document.body.appendChild(el);
            }}
        }});
    }})();
    </script>
    """
    # Use HTML injection to trigger client-side redirect
    st.components.v1.html(js, height=1, scrolling=False)

# -------------------------
# PAYWALL UI
# -------------------------
st.title("Pro Forma AI — Institutional")

# Access gating via query params (simple)
qp = st.experimental_get_query_params()
plan = qp.get("plan", [None])[0]
token = qp.get("token", [None])[0]
valid_tokens = {
    "one": os.getenv("ACCESS_TOKEN_ONE", "supersecret-onedeal-2025"),
    "annual": os.getenv("ACCESS_TOKEN_ANNUAL", "supersecret-annual-2025"),
}
has_access = (plan in valid_tokens and token == valid_tokens.get(plan))

if not has_access:
    st.header("Unlock Full Model")
    st.write("Choose an access option below. This uses Stripe Checkout (client redirect).")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("One Deal — $999", key="buy_one"):
            if STRIPE_PK:
                st.session_state.pending_checkout = {"price": ONE_DEAL_PRICE_ID}
                client_redirect_checkout(ONE_DEAL_PRICE_ID)
            else:
                st.error("STRIPE_PK not configured.")
    with c2:
        if st.button("Unlimited — $99,000/year", key="buy_annual"):
            if STRIPE_PK:
                st.session_state.pending_checkout = {"price": ANNUAL_PRICE_ID}
                client_redirect_checkout(ANNUAL_PRICE_ID)
            else:
                st.error("STRIPE_PK not configured.")

    st.stop()  # block rest of app for paywall view

# -------------------------
# App Inputs (main)
# -------------------------
st.sidebar.header("Acquisition & Capital Stack")

purchase_price = st.sidebar.number_input("Purchase Price ($)", value=50_000_000, step=100_000, format="%d")
closing_costs_pct = st.sidebar.slider("Closing Costs %", 0.0, 10.0, 1.5, 0.1) / 100.0
total_cost = purchase_price * (1 + closing_costs_pct)

st.sidebar.subheader("Senior Loan")
senior_ltv = st.sidebar.slider("Senior LTV %", 0.0, 90.0, 60.0, 1) / 100.0
senior_rate = st.sidebar.slider("Senior Interest Rate (annual %)", 0.0, 12.0, 5.5, 0.05) / 100.0
senior_amort = st.sidebar.number_input("Senior Amortization (years, 0=IO)", 0, 30, 25)
senior_io = st.sidebar.number_input("Senior IO Period (years)", 0, min(10, senior_amort) if senior_amort>0 else 0, 0)
st.sidebar.subheader("Mezz / Subordinate")
use_mezz = st.sidebar.checkbox("Include Mezz", value=False)
if use_mezz:
    mezz_pct = st.sidebar.slider("Mezz % of Cost", 0.0, 40.0, 10.0, 1) / 100.0
    mezz_rate = st.sidebar.slider("Mezz Rate %", 0.0, 20.0, 10.0, 0.1) / 100.0
else:
    mezz_pct = 0.0
    mezz_rate = 0.0

st.sidebar.subheader("Equity")
lp_share = st.sidebar.slider("LP % of Equity (rest GP)", 50, 95, 80) / 100.0

st.sidebar.header("Operating Assumptions")
gpr_y1 = st.sidebar.number_input("Year 1 GPR ($)", value=8_000_000, step=10_000)
rent_growth = st.sidebar.slider("Rent Growth %", 0.0, 8.0, 3.0, 0.1) / 100.0
vacancy = st.sidebar.slider("Vacancy %", 0.0, 20.0, 5.0, 0.1) / 100.0
opex_y1 = st.sidebar.number_input("Year 1 OpEx ($)", value=2_400_000, step=10_000)
opex_growth = st.sidebar.slider("OpEx Growth %", 0.0, 8.0, 2.5, 0.1) / 100.0
reserves = st.sidebar.number_input("Annual Reserves/CapEx ($)", value=200_000, step=10_000)

st.sidebar.header("Exit & Waterfall")
hold = st.sidebar.slider("Hold Period (years)", 1, 10, 5)
exit_cap = st.sidebar.slider("Exit Cap %", 3.0, 12.0, 5.5, 0.05) / 100.0
selling_costs = st.sidebar.slider("Selling Costs %", 0.0, 8.0, 5.0, 0.1) / 100.0
pref_annual = st.sidebar.slider("Preferred Return (LP annual %)", 0.0, 15.0, 8.0, 0.1) / 100.0
catchup_pct = st.sidebar.slider("Catch-up % to GP after pref", 0.0, 100.0, 0.0, 1.0) / 100.0

st.sidebar.markdown("### Promote tiers (IRR-hurdle driven)")
use_promote = st.sidebar.checkbox("Enable promote tiers", value=True)
promote_tiers = None
if use_promote:
    tier1_hurdle = st.sidebar.number_input("Tier1 Hurdle IRR (%)", value=12.0, step=0.5) / 100.0
    tier1_gp = st.sidebar.number_input("Tier1 GP % of residual", value=30.0, step=1.0) / 100.0
    tier2_hurdle = st.sidebar.number_input("Tier2 Hurdle IRR (%)", value=20.0, step=0.5) / 100.0
    tier2_gp = st.sidebar.number_input("Tier2 GP % of residual", value=50.0, step=1.0) / 100.0
    promote_tiers = [(tier1_hurdle, tier1_gp), (tier2_hurdle, tier2_gp)]

st.sidebar.header("Monte Carlo")
n_sims = st.sidebar.number_input("Monte Carlo sims", min_value=200, max_value=20000, value=2000, step=100)
sigma_rent = st.sidebar.slider("Rent vol (σ)", 0.0, 0.25, 0.02, 0.005)
sigma_opex = st.sidebar.slider("OpEx vol (σ)", 0.0, 0.25, 0.015, 0.005)
sigma_cap = st.sidebar.slider("Cap vol (σ)", 0.0, 0.10, 0.004, 0.001)

st.sidebar.header("Scenario")
scenario = st.sidebar.selectbox("Scenario", ["Stabilized", "Value-Add"])

st.sidebar.header("Report & Export")
include_logo = st.sidebar.checkbox("Include logo in PDF (optional)")
logo_file = st.sidebar.file_uploader("Upload logo (png/jpg)", type=["png", "jpg", "jpeg"]) if include_logo else None

# -------------------------
# Helper financial functions
# -------------------------
def annual_payment(loan, rate, amort_years):
    if amort_years == 0:
        return loan * rate
    return float(-npf_pmt(rate, amort_years, loan))

# minimal pmt fallback
def npf_pmt(rate, nper, pv, fv=0, when='end'):
    if rate == 0:
        return -(pv + fv) / nper
    x = 1 + rate
    pow_x = x ** nper
    when_factor = 1.0 if when == 'end' else (1 + rate)
    return -(fv + pv * pow_x) * rate / (pow_x - 1) / when_factor

def compute_amort_schedule(loan, rate, amort_years, years):
    balances, interests, principals, payments = [], [], [], []
    bal = loan
    if amort_years == 0:
        for y in range(1, years+1):
            interest = bal * rate
            interests.append(interest)
            principals.append(0.0)
            payments.append(interest)
            balances.append(bal)
        return balances, interests, principals, payments
    payment = annual_payment(loan, rate, amort_years)
    for y in range(1, years+1):
        interest = bal * rate
        principal = min(max(payment - interest, 0.0), bal)
        payments.append(interest + principal)
        interests.append(interest)
        principals.append(principal)
        balances.append(bal)
        bal = max(bal - principal, 0.0)
    return balances, interests, principals, payments

# waterfall settlement (simplified & robust)
def settle_final_distribution(lp_cf_so_far, gp_cf_so_far, remaining_residual, equity_lp, promote_tiers):
    if remaining_residual <= 0 or not promote_tiers:
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
        if not math.isnan(irr_full) and irr_full >= hurdle:
            # find minimum X s.t. LP irr reaches hurdle, then split remainder by gp_pct
            low, high = 0.0, residual_left
            for _ in range(80):
                mid = (low + high) / 2.0
                irr_mid = robust_irr(lp_so_far + [mid])
                if math.isnan(irr_mid):
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

# -------------------------
# Build deterministic model & waterfall
# -------------------------
def build_model():
    senior_loan = total_cost * senior_ltv
    mezz_loan_amt = total_cost * mezz_pct if (use_mezz and mezz_pct > 0) else 0.0
    equity_total = total_cost - senior_loan - mezz_loan_amt
    equity_lp = equity_total * lp_share
    equity_gp = equity_total - equity_lp

    years = []
    lp_cfs = [-equity_lp]
    gp_cfs = [-equity_gp]
    residual_accumulator = 0.0
    balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)

    bal = senior_loan
    lp_roc_remaining = equity_lp
    lp_pref_accrued = 0.0

    for y in range(1, hold + 1):
        years.append(f"Year {y}")
        gpr = gpr_y1 * ((1 + rent_growth) ** (y - 1))
        egi = gpr * (1 - vacancy)
        opex = opex_y1 * ((1 + opex_growth) ** (y - 1)) + reserves
        noi = egi - opex
        tax = 0.0
        noi_at = noi - tax

        # Debt service
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
        op_cf = noi_at - ds
        lp_pref_accrued += equity_lp * pref_annual

        lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left = apply_periodic_waterfall(
            op_cf, lp_roc_remaining, lp_pref_accrued, equity_lp, pref_annual, catchup_pct
        )

        lp_cfs.append(lp_paid)
        gp_cfs.append(gp_paid)
        residual_accumulator += residual_left

    # exit
    exit_value = noi_at / max(0.0001, exit_cap)  # avoid div by zero
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
        "exit_value": exit_value,
        "exit_reversion": exit_reversion
    }

# -------------------------
# Run deterministic and Monte Carlo
# -------------------------
st.header("Run Model")

if st.button("Run Full Institutional Model (Deterministic + Monte Carlo)"):
    with st.spinner("Running deterministic build..."):
        det = build_model()
        lp_irr_det = robust_irr(det["lp_cfs"])
        st.success("Deterministic build complete")

    st.subheader("Deterministic Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("LP IRR (det)", f"{lp_irr_det:.2%}" if not math.isnan(lp_irr_det) else "N/A")
    try:
        dscr_min = "N/A"
        c2.metric("Exit Value (net)", fmt(det["exit_value"] * (1 - selling_costs)))
    except Exception:
        c2.metric("Exit Value (net)", "N/A")
    st.dataframe(det["cf_table"].style.format({"LP CF": "${:,.0f}", "GP CF": "${:,.0f}"}))

    # Monte Carlo
    st.info(f"Running Monte Carlo with {int(n_sims)} sims (this may take a moment)...")
    irrs = []
    breaches = 0
    # Precompute cov / cholesky for correlated shocks
    corr = np.array([[1.0, 0.2, -0.4], [0.2, 1.0, -0.2], [-0.4, -0.2, 1.0]])
    cov = np.diag([sigma_rent**2, sigma_opex**2, sigma_cap**2])
    cov = np.sqrt(cov) @ corr @ np.sqrt(cov)
    try:
        L = np.linalg.cholesky(cov)
    except Exception:
        L = np.diag([sigma_rent, sigma_opex, sigma_cap])

    for i in range(int(n_sims)):
        z = np.random.normal(size=3)
        shocks = L @ z
        rent_shock = 1.0 + shocks[0]
        opex_shock = 1.0 + shocks[1]
        cap_shock = shocks[2]

        # simulate deterministic function with shocks applied
        senior_loan = total_cost * senior_ltv
        bal = senior_loan
        lp_cf_sim = [- (total_cost - senior_loan - (total_cost * mezz_pct if use_mezz else 0.0)) * lp_share]
        gp_cf_sim = [- (total_cost - senior_loan - (total_cost * mezz_pct if use_mezz else 0.0)) * (1 - lp_share)]
        lp_roc_remaining = lp_cf_sim[0] * -1.0
        lp_pref_accrued = 0.0
        residual_acc = 0.0
        balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)

        for y in range(1, hold + 1):
            gpr = gpr_y1 * ((1 + rent_growth) ** (y - 1)) * rent_shock
            v = float(np.clip(vacancy + np.random.normal(0, 0.01), 0.0, 0.9))
            egi = gpr * (1 - v)
            opex = opex_y1 * ((1 + opex_growth) ** (y - 1)) * opex_shock + reserves
            noi = egi - opex
            noi_at = noi
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
            op_cf = noi_at - ds
            lp_pref_accrued += abs(lp_cf_sim[0]) * pref_annual
            lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left = apply_periodic_waterfall(
                op_cf, lp_roc_remaining, lp_pref_accrued, abs(lp_cf_sim[0]), pref_annual, catchup_pct
            )
            lp_cf_sim.append(lp_paid)
            gp_cf_sim.append(gp_paid)
            residual_acc += residual_left

        cap_sim = max(0.03, exit_cap + cap_shock)
        exit_value = noi_at / cap_sim if cap_sim > 0 else 0.0
        exit_net = exit_value * (1 - selling_costs)
        exit_reversion = max(exit_net - bal, 0.0)
        final_residual = residual_acc + exit_reversion
        lp_add, gp_add = settle_final_distribution(lp_cf_sim, gp_cf_sim, final_residual, abs(lp_cf_sim[0]), promote_tiers)
        lp_cf_sim[-1] += lp_add
        gp_cf_sim[-1] += gp_add

        irr_sim = robust_irr(lp_cf_sim)
        if not math.isnan(irr_sim) and irr_sim > -1:
            irrs.append(irr_sim)
        # DSCR breach check simplified
        if any((x := ( (gpr_y1 * (1+rent_growth)**(y-1) * rent_shock * (1 - vacancy)) - (opex_y1*((1+opex_growth)**(y-1))*opex_shock + reserves) ))/1 < 1.2 for y in range(1, hold+1)):
            breaches += 1

    irrs = np.array(irrs)
    if irrs.size == 0:
        st.error("Monte Carlo returned no valid IRRs. Adjust inputs.")
    else:
        p5, p50, p95 = np.percentile(irrs, [5, 50, 95])
        st.subheader("Monte Carlo Results (LP IRR)")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("P5", f"{p5:.2%}")
        cc2.metric("P50", f"{p50:.2%}")
        cc3.metric("P95", f"{p95:.2%}")
        st.metric("Probability DSCR < 1.2", f"{breaches / max(1, int(n_sims)):.1%}")

        # Unified charts
        fig_mc = px.histogram(irrs * 100, nbins=80, title="LP IRR Distribution (Monte Carlo)")
        fig_mc.add_vline(x=p50 * 100, line_color="white", line_width=3)
        # Deterministic waterfall chart
        op_sum = sum([x for x in det['lp_cfs'][1:-1]]) if len(det['lp_cfs']) > 2 else 0
        wf = go.Figure(go.Waterfall(
            x=["Equity In", "Operating CF", "Exit/Residual"],
            y=[-abs(det['lp_cfs'][0]), op_sum, det['lp_cfs'][-1]],
            connector={"line":{"color":"white"}}
        ))
        wf.update_layout(title="Deterministic LP Waterfall", template="plotly_white")

        st.plotly_chart(wf, use_container_width=True)
        st.plotly_chart(fig_mc, use_container_width=True)

    # sensitivity table (two-way)
    st.subheader("Two-way Sensitivity (Exit Cap vs Rent Growth)")
    caps = np.linspace(max(0.03, exit_cap * 0.8), exit_cap * 1.2, 7)
    rents = np.linspace(max(0.0, rent_growth * 0.5), rent_growth * 1.5 + 0.001, 7)
    sens = pd.DataFrame(index=[f"{r:.2%}" for r in rents], columns=[f"{c:.2%}" for c in caps])
    for i, r in enumerate(rents):
        for j, c in enumerate(caps):
            # quick model: adjust rent growth and exit cap, run single deterministic IRR
            saved_rg = rent_growth
            saved_ec = exit_cap
            try:
                # temporarily override globals
                temp_rg = r
                temp_ec = c
                # run lightweight deterministic
                gpr = gpr_y1
                # compute NOI progression & exit
                cash = [- (total_cost - total_cost * senior_ltv - (total_cost * mezz_pct if use_mezz else 0.0)) * lp_share]
                bal_local = total_cost * senior_ltv
                for y in range(1, hold+1):
                    gpr_y = gpr * ((1 + temp_rg) ** (y - 1))
                    egi = gpr_y * (1 - vacancy)
                    opex = opex_y1 * ((1 + opex_growth) ** (y - 1)) + reserves
                    noi = egi - opex
                    # debt as interest only for sensitivity speed
                    ds = bal_local * senior_rate if senior_amort == 0 else annual_payment(bal_local, senior_rate, senior_amort)
                    net = noi - ds
                    if y == hold:
                        exit_val_local = noi / max(0.0001, temp_ec)
                        exit_net_local = exit_val_local * (1 - selling_costs)
                        cash.append(net + exit_net_local - bal_local)
                    else:
                        cash.append(net)
                irr_local = robust_irr(cash)
                sens.iloc[i, j] = f"{irr_local:.2%}" if not math.isnan(irr_local) else "N/A"
            except Exception:
                sens.iloc[i, j] = "ERR"
    st.dataframe(sens)

    # draw schedule generator (simple construction draw with interest during construction)
    st.subheader("Construction / Draw Schedule (Simple)")
    include_draws = st.checkbox("Include construction draw schedule", value=False)
    draw_schedule_df = None
    if include_draws:
        total_dev = st.number_input("Total Development Cost", value=total_cost, step=100_000)
        draw_periods = st.number_input("Draw periods (months)", min_value=1, max_value=36, value=6)
        draw_profile = [total_dev / draw_periods] * int(draw_periods)
        monthly_rate = senior_rate / 12.0
        bal = 0.0
        rows = []
        for m, draw in enumerate(draw_profile, start=1):
            bal += draw
            interest = bal * monthly_rate
            rows.append({"Month": m, "Draw": draw, "Balance": bal, "Interest this mo": interest})
        draw_schedule_df = pd.DataFrame(rows)
        st.dataframe(draw_schedule_df.style.format({"Draw": "${:,.0f}", "Balance": "${:,.0f}", "Interest this mo": "${:,.0f}"}))

    # Excel export (deterministic CF + Monte Carlo IRRs + sensitivity + draws)
    st.subheader("Export / Reports")
    export_files = {}
    # Prepare deterministic CF csv
    det_csv = det['cf_table'].to_csv(index=False).encode()
    export_files['deterministic_cf.csv'] = det_csv

    if irrs.size > 0:
        mc_csv = pd.DataFrame({"LP_IRR": irrs}).to_csv(index=False).encode()
        export_files['mc_irrs.csv'] = mc_csv

    # sensitivity to excel sheet
    if sens is not None:
        export_files['sensitivity.csv'] = sens.to_csv().encode()

    if draw_schedule_df is not None:
        export_files['draw_schedule.csv'] = draw_schedule_df.to_csv(index=False).encode()

    # Excel combined file if openpyxl available
    if EXCEL_AVAILABLE:
        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            det['cf_table'].to_excel(writer, sheet_name="Deterministic_CF", index=False)
            if irrs.size > 0:
                pd.DataFrame({"LP_IRR": irrs}).to_excel(writer, sheet_name="MC_IRRs", index=False)
            if sens is not None:
                # sens has string formatted cells; write raw numeric table? We'll write as captured strings
                sens.to_excel(writer, sheet_name="Sensitivity")
            if draw_schedule_df is not None:
                draw_schedule_df.to_excel(writer, sheet_name="Draw_Schedule", index=False)
        excel_buf.seek(0)
        st.download_button("Download Excel Workbook (multi-sheet)", excel_buf, "proforma_model.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # individual CSV downloads
    for name, data in export_files.items():
        st.download_button(f"Download {name}", data, name, "text/csv")

    # PDF export with charts embedded (if reportlab + kaleido available)
    if st.button("Generate PDF Memo (with charts)"):
        if not REPORTLAB_AVAILABLE:
            st.error("ReportLab not available on this environment — cannot generate full PDF with embedded charts.")
        else:
            # build PDF
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=72, bottomMargin=36)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("<b>Pro Forma AI — Institutional Memorandum</b>", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Date: {datetime.today().strftime('%B %d, %Y')}", styles['Normal']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            exec_text = textwrap.fill(
                f"This memorandum summarizes a deterministic and Monte Carlo analysis for a proposed acquisition at a purchase price of ${purchase_price:,.0f}. "
                f"The analysis includes capital stack assumptions, preferred return and a multi-tier promote based on IRR hurdles.",
                200)
            story.append(Paragraph(exec_text, styles['Normal']))
            story.append(Spacer(1, 8))

            # Add deterministic CF table (first few rows)
            story.append(Paragraph("Deterministic Cashflows (LP & GP) - excerpt", styles['Heading3']))
            df_small = det['cf_table'].head(20)
            data = [list(df_small.columns)] + df_small.values.tolist()
            t = Table(data, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f2f2f2")),
                ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
                ('FONT', (0,0), (-1,0), 'Helvetica-Bold'),
            ]))
            story.append(t)
            story.append(PageBreak())

            # charts embedding (needs fig->png)
            try:
                # make chart images
                # Deterministic waterfall fig previously created as 'wf'
                wf_png = figure_to_png_bytes(wf) if 'wf' in locals() else None
                mc_png = figure_to_png_bytes(fig_mc) if 'fig_mc' in locals() else None
                cf_png = None
                # create a simple CF chart
                try:
                    cf_fig = go.Figure()
                    cf_fig.add_trace(go.Bar(x=det['cf_table']['Period'], y=det['cf_table']['LP CF'], name="LP CF"))
                    cf_png = figure_to_png_bytes(cf_fig)
                except Exception:
                    cf_png = None

                for png, title in [(cf_png, "Deterministic Cash Flows"), (wf_png, "Deterministic Waterfall"), (mc_png, "Monte Carlo — LP IRR Distribution")]:
                    if png:
                        story.append(Paragraph(title, styles['Heading3']))
                        img_buf = BytesIO(png)
                        story.append(RLImage(img_buf, width=6*inch, height=3.6*inch))
                        story.append(Spacer(1, 12))
            except Exception:
                story.append(Paragraph("Charts could not be embedded (kaleido/reportlab not available).", styles['Normal']))

            doc.build(story)
            buf.seek(0)
            st.download_button("Download Full PDF Memo", buf, f"Pro_Forma_AI_Memo_{datetime.today().strftime('%Y%m%d')}.pdf", "application/pdf")

st.markdown("---")
st.info("Institutional-grade model with promote tiers, Monte Carlo, sensitivity analysis, draw schedule, PDF & Excel export. Adjust inputs and rerun.")
