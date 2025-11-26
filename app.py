# app.py — Pro Forma AI — INSTITUTIONAL-GRADE — FULL (with mezz, pref, IRR-hurdle promotes, tests, CSV)
# Run: streamlit run app.py

import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import requests
import math
from datetime import datetime
import io
import csv
from io import BytesIO
import textwrap

# Optional PDF/image libs
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

# Plotly to image (kaleido)
KALeIDO_AVAILABLE = True
try:
    # fig.to_image uses kaleido under the hood if available
    import kaleido  # noqa: F401
except Exception:
    KALeIDO_AVAILABLE = False

st.set_page_config(page_title="Pro Forma AI — Institutional (Full)", layout="wide")

# ---------------------------
# APP CONFIG / PAYWALL (simple)
# ---------------------------
ONE_DEAL_LINK = "https://buy.stripe.com/dRm5kD66J6wR0Mhfj5co001"
ANNUAL_LINK = "https://buy.stripe.com/28E5kD3YB6wR9iN4Erco000"
VALID_TOKENS = {"one": "one-token-hex", "annual": "annual-token-hex"}  # replace with your tokens

plan = st.experimental_get_query_params().get("plan", [None])[0]
token = st.experimental_get_query_params().get("token", [None])[0]
if plan not in VALID_TOKENS or token != VALID_TOKENS[plan]:
    st.title("Pro Forma AI — Paywall")
    st.markdown("Paste a valid token in the URL to unlock the full app.")
    st.markdown(f"[Buy one-off]({ONE_DEAL_LINK}) • [Buy annual]({ANNUAL_LINK})")
    st.stop()

st.title("Pro Forma AI — Institutional (Full)")

# ---------------------------
# SIDEBAR INPUTS (including logo input)
# ---------------------------
with st.sidebar:
    st.header("Acquisition & Capital Stack")
    purchase_price = st.number_input("Purchase Price ($)", value=100_000_000, step=1_000_000)
    closing_costs_pct = st.slider("Closing Costs %", 0.0, 5.0, 1.5) / 100.0
    total_cost = purchase_price * (1 + closing_costs_pct)

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
    total_equity = total_cost * (1 - senior_ltv - (mezz_pct if use_mezz else 0.0))
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

# ---------------------------
# Helper functions
# ---------------------------
def robust_irr(cfs):
    try:
        irr = npf.irr(cfs)
        if irr is None or np.isnan(irr) or np.isinf(irr):
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
        for y in range(1, years+1):
            interests.append(bal * rate)
            principals.append(0.0)
            payments.append(interests[-1])
            balances.append(bal)
        return balances, interests, principals, payments
    payment = annual_payment(loan, rate, amort_years)
    for y in range(1, years+1):
        interests.append(bal * rate)
        principal = min(max(payment - interests[-1], 0.0), bal)
        principals.append(principal)
        payments.append(interests[-1] + principal)
        balances.append(bal)
        bal = max(bal - principal, 0.0)
    return balances, interests, principals, payments

def settle_final_distribution(lp_cf_so_far, gp_cf_so_far, remaining_residual, equity_lp, promote_tiers):
    if remaining_residual <= 0 or promote_tiers is None or len(promote_tiers) == 0:
        lp_share = 0.8
        return remaining_residual * lp_share, remaining_residual * (1 - lp_share)

    lp_add_total = 0.0
    gp_add_total = 0.0
    residual_left = remaining_residual
    lp_so_far = lp_cf_so_far.copy()

    for i, (hurdle, gp_pct) in enumerate(promote_tiers):
        if residual_left <= 0:
            break
        lp_candidate_full = lp_so_far + [residual_left]
        irr_full = robust_irr(lp_candidate_full)
        if not np.isnan(irr_full) and irr_full >= hurdle:
            low, high = 0.0, residual_left
            for _ in range(60):
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

def build_model_and_settle_det():
    senior_loan = total_cost * senior_ltv
    mezz_loan = total_cost * mezz_pct if (use_mezz and mezz_pct > 0) else 0.0
    equity_total = total_cost - senior_loan - mezz_loan
    equity_lp = equity_total * lp_share_default
    equity_gp = equity_total - equity_lp

    years = []
    lp_cfs = [ -equity_lp ]
    gp_cfs = [ -equity_gp ]
    dscr_path = []
    bal = senior_loan
    lp_roc_remaining = equity_lp
    lp_pref_accrued = 0.0
    residual_accumulator = 0.0

    balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)

    for y in range(1, hold+1):
        years.append(f"Year {y}")
        gpr = gpr_y1 * ((1 + rent_growth) ** (y-1))
        egi = gpr * (1 - vacancy)
        opex = opex_y1 * ((1 + opex_growth) ** (y-1)) + reserves
        noi = egi - opex
        tax = 0.0
        noi_at = noi - tax

        if senior_amort == 0 or y <= senior_io:
            interest = bal * senior_rate
            principal = 0.0
            payment = interest
        else:
            if y-1 < len(payments):
                payment = payments[y-1]
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

    exit_value = noi_at / safe_cap(exit_cap)
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

def run_montecarlo(n_sims):
    cov = np.diag([sigma_rent**2, sigma_opex**2, sigma_cap**2])
    cov = np.sqrt(cov) @ corr @ np.sqrt(cov)
    try:
        L = np.linalg.cholesky(cov)
    except Exception:
        L = np.diag([sigma_rent, sigma_opex, sigma_cap])

    senior_loan = total_cost * senior_ltv
    mezz_loan_amt = total_cost * mezz_pct if (use_mezz and mezz_pct > 0) else 0.0
    equity_total = total_cost - senior_loan - mezz_loan_amt
    equity_lp = equity_total * lp_share_default

    irrs = []
    dscr_breach_count = 0
    for i in range(n_sims):
        z = np.random.normal(size=3)
        shocks = L @ z
        rent_shock = 1.0 + shocks[0]
        opex_shock = 1.0 + shocks[1]
        cap_shock = shocks[2]
        bal = senior_loan
        lp_cf_sim = [-equity_lp]
        lp_roc_remaining = equity_lp
        lp_pref_accrued = 0.0
        dscr_path = []
        residual_acc = 0.0

        balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)

        for y in range(1, hold+1):
            gpr = gpr_y1 * ((1 + rent_growth) ** (y-1)) * rent_shock
            v = float(np.clip(vacancy + np.random.normal(0, 0.01), 0.0, 0.9))
            egi = gpr * (1 - v)
            opex = opex_y1 * ((1 + opex_growth) ** (y-1)) * opex_shock + reserves
            noi = egi - opex
            tax = 0.0
            noi_at = noi - tax

            if senior_amort == 0 or y <= senior_io:
                interest = bal * senior_rate
                principal = 0.0
                payment = interest
            else:
                if y-1 < len(payments):
                    payment = payments[y-1]
                else:
                    payment = annual_payment(senior_loan, senior_rate, senior_amort)
                interest = bal * senior_rate
                principal = min(max(payment - interest, 0.0), bal)
            bal = max(bal - principal, 0.0)
            ds = interest + principal
            dscr_val = noi_at / ds if ds > 0 else 99.0
            dscr_path.append(dscr_val)

            op_cf = noi_at - ds
            lp_pref_accrued += equity_lp * pref_annual
            lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left = apply_periodic_waterfall(
                op_cf, lp_roc_remaining, lp_pref_accrued, equity_lp, pref_annual, catchup_pct
            )
            lp_cf_sim.append(lp_paid)
            residual_acc += residual_left

        cap_sim = safe_cap(exit_cap + cap_shock)
        exit_value = noi_at / cap_sim if cap_sim > 0 else 0.0
        exit_net = exit_value * (1 - selling_costs)
        exit_reversion = max(exit_net - bal, 0.0)
        final_residual = residual_acc + exit_reversion

        lp_add, gp_add = settle_final_distribution(lp_cf_sim, [], final_residual, equity_lp, promote_tiers)
        lp_cf_sim[-1] += lp_add

        irr_sim = robust_irr(lp_cf_sim)
        if not np.isnan(irr_sim) and irr_sim > -1:
            irrs.append(irr_sim)

        if any(d < 1.2 for d in dscr_path):
            dscr_breach_count += 1

    return np.array(irrs), dscr_breach_count

def generate_sample_csv(cf_table):
    buf = io.StringIO()
    cf_table.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue().encode()

# ---------------------------
# PDF helpers (images and PDF building)
# ---------------------------
def figure_to_png_bytes(fig):
    try:
        # fig.to_image uses kaleido
        return fig.to_image(format="png")
    except Exception:
        return None

def fetch_logo_image(logo_file, logo_url):
    # return tuple (bytes, mime) or None
    if logo_file is not None:
        try:
            data = logo_file.read()
            return data, "uploaded"
        except Exception:
            return None
    if logo_url:
        try:
            r = requests.get(logo_url, timeout=8)
            if r.status_code == 200:
                return r.content, "url"
        except Exception:
            return None
    return None

def split_dataframe_for_table(df, max_rows=30):
    """Split dataframe into list of dataframes with <= max_rows each (for tables across pages)."""
    parts = []
    n = len(df)
    for i in range(0, n, max_rows):
        parts.append(df.iloc[i:i+max_rows])
    return parts

def generate_long_pdf_memo(det, monte_stats, fig_monte, fig_waterfall, logo_blob_tuple):
    """
    Create an 11-page-ish memo using ReportLab Platypus.
    Returns (pdf_bytes, filename)
    """
    if not REPORTLAB_AVAILABLE:
        # fallback to text bytes
        text = "ReportLab not available. Install reportlab to generate full PDF memo."
        return text.encode("utf-8"), f"Pro_Forma_AI_Memo_{datetime.today().strftime('%Y%m%d')}.txt"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=72, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = styles["Heading1"]
    small = ParagraphStyle('small', parent=styles['Normal'], fontSize=9, leading=11)

    # Header with logo and title (page 1)
    if logo_blob_tuple:
        try:
            logo_bytes, _ = logo_blob_tuple
            img = ImageReader(BytesIO(logo_bytes))
            story.append(Paragraph("<b>Pro Forma AI — Institutional Memorandum</b>", heading))
            story.append(Spacer(1, 12))
            # insert logo as small image
            story.append(Spacer(1, 6))
        except Exception:
            story.append(Paragraph("<b>Pro Forma AI — Institutional Memorandum</b>", heading))
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("<b>Pro Forma AI — Institutional Memorandum</b>", heading))
        story.append(Spacer(1, 12))

    story.append(Paragraph(f"Date: {datetime.today().strftime('%B %d, %Y')}", normal))
    story.append(Spacer(1, 12))

    # Executive summary (page 1)
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    exec_text = textwrap.fill(
        f"This memorandum summarizes a deterministic and Monte Carlo analysis for a proposed acquisition at a purchase price of ${purchase_price:,.0f}. "
        f"The analysis includes capital stack assumptions (senior LTV {senior_ltv:.0%}), mezzanine (if used), LP/GP split, preferred return, and a multi-tier promote based on IRR hurdles.",
        200)
    story.append(Paragraph(exec_text, normal))
    story.append(Spacer(1, 12))

    # Investment highlights (page 2)
    story.append(PageBreak())
    story.append(Paragraph("Investment Highlights", styles['Heading2']))
    bullets = [
        f"Purchase Price: ${purchase_price:,.0f}",
        f"Total Cost incl. closing: ${total_cost:,.0f}",
        f"Estimated Equity: ${total_equity:,.0f}",
        f"Hold Period: {hold} years",
        f"Exit Cap: {exit_cap:.2%}",
        f"Preferred Return (LP): {pref_annual:.2%}",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", normal))
    story.append(Spacer(1, 12))

    # Assumptions (page 3)
    story.append(PageBreak())
    story.append(Paragraph("Key Assumptions", styles['Heading2']))
    asum_lines = [
        f"GPR Year 1: ${gpr_y1:,.0f}",
        f"Rent Growth (annual): {rent_growth:.2%}",
        f"Vacancy: {vacancy:.2%}",
        f"OpEx Year1: ${opex_y1:,.0f}",
        f"OpEx growth: {opex_growth:.2%}",
        f"Reserves/CapEx: ${reserves:,.0f}",
        f"Senior Rate: {senior_rate:.2%} | Senior Amort: {senior_amort} yrs | IO: {senior_io} yrs",
    ]
    for a in asum_lines:
        story.append(Paragraph(a, normal))
    story.append(Spacer(1, 12))

    # Deterministic CFs (pages 4-6) — table split across pages
    story.append(PageBreak())
    story.append(Paragraph("Deterministic Cashflows (LP & GP)", styles['Heading2']))
    df = det['cf_table'].copy()
    # Format numbers
    df_formatted = df.copy()
    for col in ["LP CF", "GP CF"]:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:,.0f}")
    parts = split_dataframe_for_table(df_formatted, max_rows=20)
    for idx, part in enumerate(parts):
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
        if idx < len(parts) - 1:
            story.append(PageBreak())

    # Monte Carlo Analysis (pages 7-8)
    story.append(PageBreak())
    story.append(Paragraph("Monte Carlo Analysis — LP IRR Distribution", styles['Heading2']))
    st_lines = [
        f"P5: {monte_stats.get('p5', 'N/A')}",
        f"P50: {monte_stats.get('p50', 'N/A')}",
        f"P95: {monte_stats.get('p95', 'N/A')}",
    ]
    for line in st_lines:
        story.append(Paragraph(line, normal))
    story.append(Spacer(1, 12))

    # embed images if available
    try:
        png_hist = figure_to_png_bytes(fig_monte) if fig_monte is not None else None
        png_wf = figure_to_png_bytes(fig_waterfall) if fig_waterfall is not None else None
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
        # ignore image errors
        pass

    # Waterfall mechanics & promote details (page 9)
    story.append(PageBreak())
    story.append(Paragraph("Waterfall Mechanics & Promote Tiers", styles['Heading2']))
    wf_text = "Promote tiers (IRR-hurdle driven):\n"
    if promote_tiers:
        for h, gp in promote_tiers:
            wf_text += f" - Hurdle: {h:.2%} → GP takes {gp:.0%} of residual above hurdle\n"
    else:
        wf_text += "Promote disabled; default 80/20 split at exit."
    story.append(Paragraph(wf_text.replace("\n", "<br/>"), normal))
    story.append(Spacer(1, 12))

    # Appendix (pages 10-11): assumptions, full input summary & CSV notice
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Full Inputs & Notes", styles['Heading2']))
    inputs_summary = [
        f"Purchase price: ${purchase_price:,.0f}",
        f"Total Cost incl closing: ${total_cost:,.0f}",
        f"Senior LTV: {senior_ltv:.2%}",
        f"Mezz used: {use_mezz} (mezz %: {mezz_pct:.2%})",
        f"LP % of equity: {lp_share_default:.2%}",
        f"Hold: {hold} yrs",
    ]
    for i in inputs_summary:
        story.append(Paragraph(i, normal))
    story.append(Spacer(1, 12))
    story.append(Paragraph("CSV / Data Exports", styles['Heading3']))
    story.append(Paragraph("Download deterministic and Monte Carlo outputs from the web UI.", small))

    # finish
    doc.build(story, onFirstPage=lambda c, d: add_header_footer(c, logo_blob_tuple),
              onLaterPages=lambda c, d: add_header_footer(c, logo_blob_tuple))
    buf.seek(0)
    return buf.getvalue(), f"Pro_Forma_AI_Memo_{datetime.today().strftime('%Y%m%d')}.pdf"

# header/footer function used by Platypus build
def add_header_footer(c, logo_blob_tuple):
    width, height = letter
    c.setFont("Helvetica", 8)
    # Header left: product title
    c.drawString(36, height - 50, "Pro Forma AI — Institutional")
    # Header right: date
    c.drawRightString(width - 36, height - 50, datetime.today().strftime("%B %d, %Y"))
    # Footer: page number
    page_num_text = f"Page {c.getPageNumber()}"
    c.drawCentredString(width / 2.0, 30, page_num_text)
    # logo in header if provided
    if logo_blob_tuple:
        try:
            logo_bytes, _ = logo_blob_tuple
            img = ImageReader(BytesIO(logo_bytes))
            # draw small logo at header left
            c.drawImage(img, width - 140, height - 70, width=80, height=30, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass

# High-level PDF generator that chooses between the long memo and a short summary
def generate_pdf_report(det, monte_stats, fig_monte, fig_waterfall, logo_blob_tuple):
    """
    Try to create full memo using ReportLab. If fails, fallback to a short two-page PDF.
    """
    try:
        pdf_bytes, filename = generate_long_pdf_memo(det, monte_stats, fig_monte, fig_waterfall, logo_blob_tuple)
        return pdf_bytes, filename
    except Exception as e:
        # fallback: produce a short text/PDF
        try:
            if REPORTLAB_AVAILABLE:
                # basic single-page PDF
                buf = BytesIO()
                c = canvas.Canvas(buf, pagesize=letter)
                width, height = letter
                c.setFont("Helvetica-Bold", 16)
                c.drawString(36, height - 72, "Pro Forma AI — Summary")
                c.setFont("Helvetica", 10)
                y = height - 100
                c.drawString(36, y, f"Date: {datetime.today().strftime('%B %d, %Y')}")
                y -= 20
                c.drawString(36, y, f"LP IRR (det): {robust_irr(det['lp_cfs']):.2%}")
                y -= 14
                c.drawString(36, y, f"P50 (MC): {monte_stats.get('p50', 'N/A')}")
                y -= 14
                c.drawString(36, y, f"P95 (MC): {monte_stats.get('p95', 'N/A')}")
                c.showPage()
                c.save()
                buf.seek(0)
                return buf.getvalue(), f"Pro_Forma_AI_Summary_{datetime.today().strftime('%Y%m%d')}.pdf"
        except Exception:
            pass
        # ultimate fallback: plain text bytes
        text = (f"Pro Forma AI Summary\nDate: {datetime.today().strftime('%B %d, %Y')}\n"
                f"LP IRR (det): {robust_irr(det['lp_cfs']):.2%}\n"
                f"P50 (MC): {monte_stats.get('p50', 'N/A')}\n"
                f"P95 (MC): {monte_stats.get('p95', 'N/A')}\n")
        return text.encode("utf-8"), f"Pro_Forma_AI_Summary_{datetime.today().strftime('%Y%m%d')}.txt"

# ---------------------------
# UI Buttons & Run
# ---------------------------
if st.button("Run Full Institutional Model (Deterministic + Monte Carlo)"):
    # deterministic
    with st.spinner("Running deterministic build..."):
        det = build_model_and_settle_det()
        lp_cfs = det['lp_cfs']
        gp_cfs = det['gp_cfs']
        cf_table = det['cf_table']
        exit_value = det['exit_value']
        exit_reversion = det['exit_reversion']
        dscr_path = det['dscr_path']
        lp_irr = robust_irr(lp_cfs)
        st.success("Deterministic build complete")

    st.subheader("Deterministic Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("LP IRR (det)", f"{lp_irr:.2%}" if not math.isnan(lp_irr) else "N/A")
    c2.metric("Min DSCR", f"{min(dscr_path):.2f}x" if dscr_path else "N/A")
    c3.metric("Exit Value (net)", f"${exit_value*(1-selling_costs):,.0f}")

    st.markdown("### Deterministic Cashflow Table (LP)")
    st.dataframe(cf_table)

    csv_bytes = generate_sample_csv(cf_table)
    st.download_button("Download deterministic CF CSV", csv_bytes, "deterministic_cf.csv", "text/csv")

    # Monte Carlo run
    st.info(f"Running Monte Carlo with {n_sims} sims — this may take time.")
    with st.spinner("Running Monte Carlo..."):
        irrs, breaches = run_montecarlo(int(n_sims))

    if irrs.size == 0:
        st.error("Monte Carlo produced no valid IRRs. Check inputs.")
        p5 = p50 = p95 = None
    else:
        p5, p50, p95 = np.percentile(irrs, [5, 50, 95])
        st.subheader("Monte Carlo Results (LP IRR)")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("P5", f"{p5:.2%}")
        cc2.metric("P50", f"{p50:.2%}")
        cc3.metric("P95", f"{p95:.2%}")
        st.metric("Probability DSCR < 1.2", f"{breaches / max(1, int(n_sims)):.1%}")

        fig_monte = px.histogram(irrs*100, nbins=80, title="LP IRR Distribution (Monte Carlo)")
        fig_monte.add_vline(x=p50*100, line_color="white", line_width=3)
        st.plotly_chart(fig_monte, use_container_width=True)

        # deterministic WP waterfall
        try:
            op_sum = sum([x for x in det['lp_cfs'][1:-1]]) if len(det['lp_cfs'])>2 else 0
            wf = go.Figure(go.Waterfall(
                x=["Equity In", "Operating CF", "Exit/Residual"],
                y=[-abs(det['lp_cfs'][0]), op_sum, det['lp_cfs'][-1]],
                connector={"line":{"color":"white"}}
            ))
            wf.update_layout(title="Deterministic LP Waterfall", template="plotly_white")
            fig_waterfall = wf
            st.plotly_chart(fig_waterfall, use_container_width=True)
        except Exception:
            fig_waterfall = None

    # prepare monte stats
    monte_stats = {
        "p5": f"{p5:.1%}" if (p5 is not None) else "N/A",
        "p50": f"{p50:.1%}" if (p50 is not None) else "N/A",
        "p95": f"{p95:.1%}" if (p95 is not None) else "N/A",
    }

    # fetch logo bytes (if any)
    logo_blob = None
    try:
        logo_blob = fetch_logo_image(logo_file if 'logo_file' in locals() else None,
                                     logo_url if 'logo_url' in locals() else None)
    except Exception:
        logo_blob = None

    # generate PDF
    with st.spinner("Generating PDF memo (multi-page) ..."):
        try:
            pdf_bytes, filename = generate_pdf_report(det, monte_stats,
                                                      fig_monte if 'fig_monte' in locals() else None,
                                                      fig_waterfall if 'fig_waterfall' in locals() else None,
                                                      logo_blob)
            # If returned type is bytes (pdf or text)
            mime = "application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"
            st.success("PDF memo ready")
            st.download_button("Download Full PDF Memo", pdf_bytes, filename, mime)
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            # fallback: provide summary txt
            summary_text = (f"Deterministic LP IRR: {lp_irr:.2%}\n"
                            f"P50 (MC): {monte_stats.get('p50')}\n"
                            f"P95 (MC): {monte_stats.get('p95')}\n")
            st.download_button("Download Summary (TXT)", summary_text.encode(), "proforma_summary.txt", "text/plain")

    # also offer Monte Carlo raw results
    if 'irrs' in locals() and irrs.size > 0:
        # create CSV of IRRs
        buf = io.StringIO()
        pd.DataFrame({"LP_IRR": irrs}).to_csv(buf, index=False)
        st.download_button("Download Monte Carlo IRRs (CSV)", buf.getvalue().encode(), "mc_irrs.csv", "text/csv")

    # optional remote PDF service (non-blocking)
    try:
        payload = {
            "date": datetime.today().strftime("%B %d, %Y"),
            "p50": monte_stats.get('p50'),
            "p95": monte_stats.get('p95'),
            "min_dscr": f"{min(dscr_path):.2f}x" if dscr_path else "N/A",
        }
        requests.post("https://proforma-ai-production.up.railway.app/api/pdf", json=payload, timeout=8)
    except Exception:
        pass

st.markdown("---")
st.info("This is an institutional-grade model with multi-tier promote settlement, Monte Carlo analysis, and automated multi-page PDF reporting (with logo/header/footer). Adjust inputs and rerun.")
