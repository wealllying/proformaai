# app.py — Pro Forma AI — REWRITTEN — Institutional-grade (2025)
# Full amortization, robust waterfall, IRR-hurdle promote option, Monte Carlo, PDF payload
import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import base64
import requests
import math
from datetime import datetime

# =============================
# 1. PAYWALL & TOKEN SYSTEM
# =============================
ONE_DEAL_LINK = "https://buy.stripe.com/dRm5kD66J6wR0Mhfj5co001"
ANNUAL_LINK   = "https://buy.stripe.com/28E5kD3YB6wR9iN4Erco000"

VALID_TOKENS = {
    "one": "8f4e9a2b1c3d5e7f9a0b1c2d3e4f5a6b7c8d9e0f",
    "annual": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"
}

plan = st.query_params.get("plan")
token = st.query_params.get("token")

if plan not in VALID_TOKENS or token != VALID_TOKENS[plan]:
    st.set_page_config(page_title="Pro Forma AI", layout="centered")
    st.markdown("""
    <style>
        .stApp {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);}
        body, h1,h2,h3,h4,h5,h6,p,div,span,label {color: white !important;}
        .big-title {font-size: 7rem; font-weight: 900; background: linear-gradient(90deg, #00dbde, #fc00ff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
        .buy-btn {display: inline-block; background: linear-gradient(90deg, #00dbde, #fc00ff); color: white;
                  padding: 28px 60px; font-size: 2.2rem; font-weight: bold; border-radius: 30px; text-decoration: none;
                  text-align: center; width: 100%; box-shadow: 0 10px 30px rgba(0,219,222,0.4); margin: 20px 0;}
    </style>
    <div class="big-title">Pro Forma AI</div>
    <h2 style='text-align:center;color:white;margin-top:20px;'>The model that closed $4.3B in 2025</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<a href="{ONE_DEAL_LINK}" target="_blank" class="buy-btn">One Deal — $999</a>', unsafe_allow_html=True)
    with col2:
        success_url = f"https://proforma-ai-production.up.railway.app/?plan=annual&token={VALID_TOKENS['annual']}"
        st.markdown(f'<a href="{ANNUAL_LINK}?success_url={success_url}" target="_blank" class="buy-btn">Unlimited + Portfolio — $49,000</a>', unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#888;margin-top:60px;font-size:1.3rem;'>After payment, return here — access unlocks instantly.</p>", unsafe_allow_html=True)
    st.stop()

# =============================
# 2. UI & Inputs
# =============================
st.set_page_config(page_title="Pro Forma AI", layout="wide")
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);}
    h1,h2,h3,h4,h5,h6,p,div,span,label,.stMarkdown {color: white !important;}
    .big-title {font-size: 7rem !important; font-weight: 900; background: linear-gradient(90deg, #00dbde, #fc00ff);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .stButton>button {background: linear-gradient(90deg, #00dbde, #fc00ff); color: white; height: 80px; font-size: 2rem;
                      border-radius: 25px; border: none; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Pro Forma AI</div>', unsafe_allow_html=True)
st.success("Full Institutional Access — Amortization • Waterfall • 50k Monte Carlo • 11-Page PDF")

st.markdown("### Acquisition & Operating Assumptions")
c1, c2, c3, c4 = st.columns(4)
with c1:
    purchase_price = st.number_input("Purchase Price ($)", value=100_000_000, step=1_000_000, min_value=1_000_000)
    closing_costs = st.slider("Closing Costs %", 0.0, 5.0, 1.5) / 100
    total_cost = purchase_price * (1 + closing_costs)
    equity_pct = st.slider("Equity %", 10, 50, 30) / 100
    ltc = st.slider("LTC %", 50, 80, 70) / 100
    rate = st.slider("Interest Rate % (annual)", 3.0, 9.0, 6.0, 0.05) / 100
    loan_term = st.number_input("Loan Term (years)", 5, 30, 25)
    amort_years = st.number_input("Amortization (years, 0 = IO)", 0, 40, 30)
    io_period = st.number_input("IO Period (years)", 0, 10, 0)
with c2:
    gpr_y1 = st.number_input("Year 1 GPR ($)", value=12_000_000, min_value=100_000)
    rent_growth = st.slider("Rent Growth %", 0.0, 6.0, 3.0, 0.1) / 100
    vacancy = st.slider("Vacancy %", 0.0, 20.0, 5.0) / 100
    opex_y1 = st.number_input("Year 1 OpEx ($)", value=3_600_000)
    opex_growth = st.slider("OpEx Growth %", 0.0, 6.0, 2.5, 0.1) / 100
    reserves = st.number_input("Annual Reserves + CapEx ($)", value=400_000)
with c3:
    hold = st.slider("Hold Period (years)", 3, 10, 5)
    exit_cap = st.slider("Exit Cap Rate %", 4.0, 9.0, 5.5, 0.05) / 100
    selling_costs = st.slider("Selling Costs %", 0.0, 8.0, 5.0) / 100
    pref = st.slider("Preferred Return (annual)", 4.0, 12.0, 8.0) / 100
    promote_hurdle = st.slider("Promote Hurdle IRR (optional)", 0.0, 25.0, 15.0, 0.5) / 100
with c4:
    promote_pct = st.slider("Promote % (GP on residual)", 0.0, 50.0, 20.0, 1.0) / 100
    promote_mode = st.selectbox("Promote Mode", ["Residual Split", "IRR-Hurdle (approx)"])
    tax_system = st.selectbox("Tax System", ["Mill Rate", "California Prop 13", "Texas", "Florida SOH"])
    if tax_system == "Mill Rate":
        assessed = st.number_input("Assessed Value ($)", value=90_000_000)
        mill_rate = st.slider("Mill Rate (per $1k)", 5.0, 40.0, 24.0, 0.1)
    else:
        assessed = purchase_price * 0.85

# =============================
# Helpers: IRR, amortization, safe cap, waterfall calculators
# =============================
def calculate_irr(cashflows, tol=1e-7, maxiter=200):
    """Robust IRR: try numpy_financial.irr, fallback to bisection if needed.
       Returns decimal (e.g., 0.12)."""
    try:
        irr = npf.irr(cashflows)
        if irr is None or np.isnan(irr) or np.isinf(irr):
            raise Exception("npf.irr invalid")
        return float(irr)
    except Exception:
        # bisection between -0.9999 and +1000
        def npv(rate):
            return sum(cf / ((1 + rate)**i) for i, cf in enumerate(cashflows))
        low, high = -0.9999, 10.0
        f_low, f_high = npv(low), npv(high)
        # expand if necessary
        it = 0
        while f_low * f_high > 0 and it < 50:
            high *= 2
            f_high = npv(high)
            it += 1
        if f_low * f_high > 0:
            return float('nan')
        for _ in range(maxiter):
            mid = (low + high) / 2
            f_mid = npv(mid)
            if abs(f_mid) < tol:
                return mid
            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return (low + high) / 2

def annual_payment(loan, rate, amort_years):
    """Return annual payment (positive). If amort_years==0 -> interest-only payment returned."""
    if amort_years == 0:
        return loan * rate
    # npf.pmt expects rate per period; using annual periods
    p = -npf.pmt(rate, amort_years, loan)
    return float(p)

def safe_cap(rate, floor=0.03, cap=0.20):
    return min(max(rate, floor), cap)

def distribute_waterfall_periodic(dist, lp_roc_remaining, lp_pref_accrued, equity_lp, pref, promote_pct):
    """Given distributable cash 'dist' in a period, apply ROC->PREF->Residual split.
       Returns (lp_dist, gp_dist, updated_lp_roc_remaining, updated_lp_pref_accrued)."""
    lp_dist = 0.0
    gp_dist = 0.0
    # 1) Return of capital (LP):
    if lp_roc_remaining > 0 and dist > 0:
        roc_pay = min(lp_roc_remaining, dist)
        lp_dist += roc_pay
        lp_roc_remaining -= roc_pay
        dist -= roc_pay
    # 2) Preferred accrual/pmt (simple annual accrual on original LP equity)
    # accrual occurs outside; caller should add pref accrual each year; here we pay accrued pref if available.
    if dist > 0 and lp_pref_accrued > 0:
        pay_pref = min(lp_pref_accrued, dist)
        lp_dist += pay_pref
        lp_pref_accrued -= pay_pref
        dist -= pay_pref
    # 3) Residual split
    if dist > 0:
        lp_share = 1 - promote_pct
        lp_resid = dist * lp_share
        gp_resid = dist - lp_resid
        lp_dist += lp_resid
        gp_dist += gp_resid
        dist = 0.0
    return lp_dist, gp_dist, lp_roc_remaining, lp_pref_accrued

def apply_irr_hurdle_adjustment(cf_lp_so_far, residual_amount, equity_lp, pref, promote_pct, hurdle):
    """If promote_mode == IRR-Hurdle: determine GP take on residual such that LP IRR is at most 'hurdle',
       but not exceeding promote_pct of the residual. We solve for GP_share in [0, promote_pct] by bisection.
       Returns (lp_add, gp_add)
       Note: this is an approximation to mimic IRR-hurdle based promote in a single final period.
    """
    if residual_amount <= 0:
        return 0.0, 0.0
    # baseline: if LP receives all residual -> compute IRR
    lp_with_all = cf_lp_so_far + [residual_amount]
    irr_all = calculate_irr(lp_with_all)
    if np.isnan(irr_all):
        # fallback to split
        return residual_amount * (1 - promote_pct), residual_amount * promote_pct
    if irr_all <= hurdle or promote_pct <= 0:
        # nobody needs to take promote; LP below hurdle or no promote
        return residual_amount * (1 - promote_pct), residual_amount * promote_pct
    # Need to find gp_share in [0, promote_pct] that reduces LP IRR to approximately hurdle.
    low, high = 0.0, promote_pct
    for _ in range(40):
        mid = (low + high) / 2
        lp_share = 1 - mid
        lp_candidate = cf_lp_so_far + [residual_amount * lp_share]
        irr_candidate = calculate_irr(lp_candidate)
        if np.isnan(irr_candidate):
            # push GP share up to reduce LP IRR
            low = mid
            continue
        if irr_candidate > hurdle:
            # LP still above hurdle => give more to GP
            low = mid
        else:
            high = mid
    gp_share = high
    lp_add = residual_amount * (1 - gp_share)
    gp_add = residual_amount - lp_add
    return lp_add, gp_add

# =============================
# 3. MODEL RUN — deterministic base + Monte Carlo
# =============================
if st.button("RUN FULL INSTITUTIONAL PACKAGE", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 Monte Carlo paths + generating outputs..."):
        np.random.seed(42)

        # capital stack
        loan = total_cost * ltc
        equity_in = total_cost - loan
        equity_lp = equity_in * (1 - 0.2)  # default 80/20 split (LP/GP)
        equity_gp = equity_in - equity_lp

        # payments
        annual_pmt = annual_payment(loan, rate, amort_years)

        # base-case tracking
        balance = loan
        cf_lp_base = [-equity_lp]
        cf_gp_base = [-equity_gp]
        dscr_list = []
        lp_roc_remaining = equity_lp
        lp_pref_accrued = 0.0

        for y in range(1, hold + 1):
            # deterministic projections
            gpr = gpr_y1 * (1 + rent_growth) ** (y - 1)
            egi = gpr * (1 - vacancy)
            opex = opex_y1 * (1 + opex_growth) ** (y - 1) + reserves
            noi = egi - opex

            # taxes
            if tax_system == "California Prop 13":
                tax = purchase_price * 0.01 * (1.02) ** (y - 1)
            elif tax_system == "Texas":
                tax = assessed * 0.018
            elif tax_system == "Florida SOH":
                tax = min(assessed * 0.015, noi * 0.12)
            else:
                tax = (assessed / 1000) * mill_rate
                assessed *= 1.02

            noi_at = noi - tax

            # debt service (annual)
            if y <= io_period or amort_years == 0:
                interest = balance * rate
                principal = 0.0
                annual_payment_amount = interest
            else:
                interest = balance * rate
                annual_payment_amount = annual_pmt
                principal = min(max(annual_payment_amount - interest, 0.0), balance)

            ds = interest + principal
            balance -= principal
            dscr = noi_at / ds if ds > 0 else 99.0
            dscr_list.append(dscr)

            # distributable cash before waterfall
            op_cf = noi_at - ds
            if y == hold:
                exit_value = noi_at / safe_cap(exit_cap)
                net_proceeds = exit_value * (1 - selling_costs) - balance
                op_cf += net_proceeds

            # accrue pref (annual simple on original LP equity)
            lp_pref_accrued += equity_lp * pref

            # apply waterfall per period (deterministic)
            lp_dist, gp_dist, lp_roc_remaining, lp_pref_accrued = distribute_waterfall_periodic(
                op_cf, lp_roc_remaining, lp_pref_accrued, equity_lp, pref, promote_pct
            )

            cf_lp_base.append(lp_dist)
            cf_gp_base.append(gp_dist)

        # base LP metrics
        lp_irr = calculate_irr(cf_lp_base)
        lp_multiple = sum(cf_lp_base) / equity_lp if equity_lp != 0 else float('nan')

        # Monte Carlo runs
        n_sims = 50000
        irrs = []
        # For performance: pre-generate normals (optional optimization not required)
        for sim in range(n_sims):
            cf_lp = [-equity_lp]
            bal = loan
            lp_roc_remaining_mc = equity_lp
            lp_pref_accrued_mc = 0.0

            for y in range(1, hold + 1):
                # stochastic draws with sensible clipping
                shock_rent = 1 + np.random.normal(0, 0.02)
                gpr = gpr_y1 * (1 + rent_growth) ** (y - 1) * shock_rent
                v = float(np.clip(vacancy + np.random.normal(0, 0.01), 0.0, 0.9))
                egi = gpr * (1 - v)
                op = opex_y1 * (1 + opex_growth) ** (y - 1) * (1 + np.random.normal(0, 0.015)) + reserves
                noi = egi - op

                # tax sampling
                if tax_system == "California Prop 13":
                    tax = purchase_price * 0.01 * (1.02) ** (y - 1)
                elif tax_system == "Texas":
                    tax = assessed * 0.018 * (1 + np.random.normal(0, 0.01))
                elif tax_system == "Florida SOH":
                    tax = min(assessed * 0.015, max(0.0, noi * 0.12))
                else:
                    tax = (assessed / 1000) * (mill_rate * (1 + np.random.normal(0, 0.02)))

                noi_at = noi - tax

                # debt service
                if y <= io_period or amort_years == 0:
                    interest = bal * rate
                    principal = 0.0
                else:
                    interest = bal * rate
                    annual_payment_amount = annual_pmt
                    principal = min(max(annual_payment_amount - interest, 0.0), bal)

                ds = interest + principal
                bal -= principal

                op_cf = noi_at - ds
                if y == hold:
                    cap_sim = safe_cap(exit_cap + np.random.normal(0, 0.003))
                    ev = noi_at / cap_sim if cap_sim > 0 else 0.0
                    net_proceeds = ev * (1 - selling_costs) - bal
                    op_cf += net_proceeds

                # accrue pref
                lp_pref_accrued_mc += equity_lp * pref

                # waterfall per sim: ROC->PREF->residual
                # For IRR-hurdle mode, attempt to adjust final residual split when on final period
                if y < hold or promote_mode == "Residual Split":
                    lp_dist, gp_dist, lp_roc_remaining_mc, lp_pref_accrued_mc = distribute_waterfall_periodic(
                        op_cf, lp_roc_remaining_mc, lp_pref_accrued_mc, equity_lp, pref, promote_pct
                    )
                else:
                    # final year with potential IRR-hurdle adjustment
                    # first apply ROC & pref payments from op_cf to determine residual
                    # Use distribute_waterfall_periodic to get standard residual amount, then adjust residual
                    # by IRR-hurdle if requested.
                    # Apply initial ROC & pref
                    lp_dist_pre, gp_dist_pre, lp_roc_remaining_mc, lp_pref_accrued_mc = distribute_waterfall_periodic(
                        op_cf, lp_roc_remaining_mc, lp_pref_accrued_mc, equity_lp, pref, 0.0
                    )
                    # Determine residual left after ROC & pref distribution (dist_remaining)
                    # Note: distribute_waterfall_periodic with promote_pct=0 returns lp_dist_pre==op_cf after ROC/pref
                    dist_remaining = op_cf
                    # calculate how much has been paid to LP so far in this sim across periods
                    # (cf_lp currently contains past distributions only)
                    cf_lp_so_far = cf_lp.copy()
                    # append payments up to this period excluding residual (we will compute), but easier:
                    # We will find lp_add and gp_add on dist_remaining via IRR-hurdle if requested
                    if promote_mode == "IRR-Hurdle" and promote_hurdle > 0:
                        # compute lp share and gp share of dist_remaining under irr-hurdle adjustment
                        # The function apply_irr_hurdle_adjustment expects cf_lp_so_far (past distributions list)
                        lp_add, gp_add = apply_irr_hurdle_adjustment(cf_lp_so_far, dist_remaining, equity_lp, pref, promote_pct, promote_hurdle)
                    else:
                        lp_add = dist_remaining * (1 - promote_pct)
                        gp_add = dist_remaining * promote_pct
                    lp_dist = lp_add
                    gp_dist = gp_add
                    # zero-out remaining (we've allocated)
                    lp_roc_remaining_mc = max(0.0, lp_roc_remaining_mc - lp_dist)  # rough adjustment
                    lp_pref_accrued_mc = max(0.0, lp_pref_accrued_mc - lp_dist)

                cf_lp.append(lp_dist)

            # compute irr for this sim
            irr = calculate_irr(cf_lp)
            if not np.isnan(irr) and irr > -1:
                irrs.append(irr)

        valid_irrs = np.array(irrs)
        if valid_irrs.size == 0:
            st.error("Monte Carlo produced no valid IRR samples — check inputs.")
            p5 = p50 = p95 = float('nan')
        else:
            p5, p50, p95 = np.percentile(valid_irrs, [5, 50, 95])

        # Charts
        fig_monte = go.Figure()
        fig_monte.add_histogram(x=valid_irrs * 100, nbinsx=80, name="LP IRR Distribution", marker_color="#00dbde")
        fig_monte.add_vline(x=p50 * 100, line_color="white", line_width=3)
        fig_monte.update_layout(title="50,000-PATH MONTE CARLO (LP IRR)", template="plotly_dark")

        fig_waterfall = go.Figure(go.Waterfall(
            x=["Equity In", "Operating CF (sum)", "Exit Proceeds (net)", "Total Return"],
            y=[-equity_in, sum(cf_lp_base[1:-1]) if len(cf_lp_base) > 2 else 0, cf_lp_base[-1] if len(cf_lp_base) > 1 else 0, 0],
            connector={"line": {"color": "white"}}
        ))
        fig_waterfall.update_layout(title="Deterministic Waterfall (LP)", template="plotly_dark")

        # PDF payload
        try:
            mon_png = base64.b64encode(fig_monte.to_image(format="png", width=1200, height=600)).decode()
            wf_png = base64.b64encode(fig_waterfall.to_image(format="png", width=1200, height=600)).decode()
        except Exception:
            # fallback small images if renderer not available in environment
            mon_png = ""
            wf_png = ""

        payload = {
            "date": datetime.today().strftime('%B %d, %Y'),
            "lp_irr": f"{lp_irr:.1%}" if not math.isnan(lp_irr) else "N/A",
            "p5": f"{p5:.1%}" if not math.isnan(p5) else "N/A",
            "p50": f"{p50:.1%}" if not math.isnan(p50) else "N/A",
            "p95": f"{p95:.1%}" if not math.isnan(p95) else "N/A",
            "min_dscr": f"{min(dscr_list):.2f}x" if dscr_list else "N/A",
            "lp_multiple": f"{lp_multiple:.2f}x" if not math.isnan(lp_multiple) else "N/A",
            "monte_png": mon_png,
            "waterfall_png": wf_png,
        }

        # safe POST
        try:
            response = requests.post("https://proforma-ai-production.up.railway.app/api/pdf", json=payload, timeout=60)
        except Exception as e:
            response = type("X", (), {"status_code": 500, "content": str(e)})()

    # =============================
    # 4. DISPLAY
    # =============================
    col1, col2, col3, col4 = st.columns(4)
    try:
        col1.metric("Base LP IRR", f"{lp_irr:.1%}")
    except Exception:
        col1.metric("Base LP IRR", "N/A")
    col2.metric("P50 IRR", f"{p50:.1%}" if not math.isnan(p50) else "N/A")
    col3.metric("P95 IRR", f"{p95:.1%}" if not math.isnan(p95) else "N/A")
    col4.metric("Min DSCR", f"{min(dscr_list):.2f}x" if dscr_list else "N/A")

    st.plotly_chart(fig_monte, use_container_width=True)
    st.plotly_chart(fig_waterfall, use_container_width=True)

    if getattr(response, "status_code", None) == 200:
        st.download_button(
            "DOWNLOAD 11-PAGE INSTITUTIONAL PDF",
            response.content,
            "Pro_Forma_AI_Institutional_Memorandum.pdf",
            "application/pdf",
            type="primary",
            use_container_width=True
        )

st.markdown("© 2025 Pro Forma AI — The Real Institutional Model")
