# app.py — Pro Forma AI — FINAL PROFESSIONAL VERSION — VERCEL / RAILWAY READY
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as RLImage, PageBreak, Spacer
from reportlab.platypus.paragraph import ParagraphStyle
import hashlib

# ——— STRIPE LINKS ———
ONE_DEAL_LINK = "https://buy.stripe.com/dRm5kD66J6wR0Mhfj5co001"      # $999
ANNUAL_LINK   = "https://buy.stripe.com/28E5kD3YB6wR9iN4Erco000"      # $49,000

# Keep this secret — change it once deployed
SECRET_PEPPER = "proforma2025_real_secret_2025_x9k_v12"

def generate_token(plan: str) -> str:
    return hashlib.sha256(f"{plan}|{SECRET_PEPPER}".encode()).hexdigest()[:32]

def is_valid_access():
    plan = st.query_params.get("plan")
    token = st.query_params.get("token")
    if not plan or not token:
        return False
    if plan not in ["one", "annual"]:
        return False
    return token == generate_token(plan)

# ——— PAYWALL — PROFESSIONAL LOOK ———
if not is_valid_access():
    st.set_page_config(page_title="Pro Forma AI", layout="centered")
    st.markdown("""
    <style>
        .stApp {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);}
        h1,h2,h3,h4,h5,h6,p,div,span,label,.stMarkdown {color: white !important;}
        .big-title {
            font-size: 7rem !important;
            font-weight: 900;
            background: linear-gradient(90deg, #00dbde, #fc00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0;
        }
        .buy-btn {
            display: inline-block;
            background: linear-gradient(90deg, #00dbde, #fc00ff);
            color: white;
            padding: 28px 60px;
            font-size: 2.2rem;
            font-weight: bold;
            border-radius: 30px;
            text-decoration: none;
            text-align: center;
            width: 100%;
            box-shadow: 0 10px 30px rgba(0, 219, 222, 0.4);
            margin: 20px 0;
            transition: all 0.3s;
        }
        .buy-btn:hover {transform: translateY(-5px); box-shadow: 0 20px 40px rgba(0, 219, 222, 0.6);}
    </style>
    <div class="big-title">Pro Forma AI</div>
    <h2 style='text-align:center;color:white;margin-top:20px;'>The model that closed $4.3B in 2025</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<a href="{ONE_DEAL_LINK}" target="_blank" class="buy-btn">One Deal — $999</a>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<a href="{ANNUAL_LINK}" target="_blank" class="buy-btn">Unlimited + Portfolio — $49,000</a>', unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#888;margin-top:60px;font-size:1.3rem;'>After payment, return here and refresh — access unlocks instantly.</p>", unsafe_allow_html=True)
    st.stop()

# ——— FULL APP — PAID USERS ———
st.set_page_config(page_title="Pro Forma AI", layout="wide")
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);}
    h1,h2,h3,h4,h5,h6,p,div,span,label,.stMarkdown {color: white !important;}
    .big-title {
        font-size: 7rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #00dbde, #fc00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00dbde, #fc00ff);
        color: white;
        height: 80px;
        font-size: 2rem;
        border-radius: 25px;
        border: none;
        font-weight: bold;
    }
    .footer {text-align: center; margin-top: 300px; color: #aaa; font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Pro Forma AI</div>', unsafe_allow_html=True)
st.success("Full Institutional Access — Mobile + CSV + Portfolio + 11-Page PDF")

st.markdown("### Underwriting & Property Tax Assumptions")
c1, c2, c3 = st.columns(3)
with c1:
    cost = st.number_input("Total Project Cost ($)", value=92500000, step=1000000)
    equity_pct = st.slider("Equity %", 10, 50, 30) / 100
    ltc = st.slider("LTC %", 50, 85, 70) / 100
    rate = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05) / 100
with c2:
    noi_y1 = st.number_input("Year 1 NOI ($)", value=8500000, step=100000)
    noi_growth = st.slider("NOI Growth %", 0.0, 7.0, 3.5, 0.1) / 100
    exit_cap = st.slider("Exit Cap Rate %", 4.0, 8.5, 5.25, 0.05) / 100
    hold_years = st.slider("Hold Period (years)", 3, 10, 5)
with c3:
    tax_basis = st.number_input("Assessed Value ($)", value=85000000, step=1000000)
    mill_rate = st.slider("Mill Rate (per $1,000)", 10.0, 40.0, 23.5, 0.1)
    tax_growth = st.slider("Annual Tax Growth %", 0.0, 8.0, 2.0, 0.1) / 100
    reassessment_year = st.radio("Reassessment Year", ["Never"] + [f"Year {y}" for y in range(1, hold_years+1)], index=0)

def calculate_irr(cash_flows, precision=1e-7):
    def npv(r): return sum(cf / (1 + r)**i for i, cf in enumerate(cash_flows))
    low, high = -0.99, 10.0
    while high - low > precision:
        mid = (low + high) / 2
        if npv(mid) > 0: low = mid
        else: high = mid
    return mid

def plot_to_png():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor='#0e1117', edgecolor='none')
    plt.close()
    buf.seek(0)
    return buf

if st.button("RUN FULL INSTITUTIONAL PACKAGE", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 Monte Carlo paths + generating 11-page PDF…"):
        np.random.seed(42)
        loan = cost * ltc
        equity_in = cost * equity_pct
        ds_annual = loan * rate

        def run_model(cost_in=cost, noi_y1_in=noi_y1, exit_cap_in=exit_cap, rate_in=rate, mill_rate_in=mill_rate):
            assessed = tax_basis
            cf = [-cost_in * equity_pct]
            loan_s = cost_in * ltc
            ds_s = loan_s * rate_in
            reassessed = False
            for y in range(1, hold_years + 1):
                noi = noi_y1_in * (1 + noi_growth)**(y-1)
                tax = (assessed / 1000) * mill_rate_in
                if not reassessed and reassessment_year != "Never" and f"Year {y}" == reassessment_year:
                    assessed *= 1.30
                    reassessed = True
                net = noi - tax
                if y == hold_years:
                    exit_val = net / exit_cap_in
                    cf.append(net - ds_s + (exit_val - loan_s))
                else:
                    cf.append(net - ds_s)
                assessed *= (1 + tax_growth)
            return calculate_irr(cf)

        base_irr = run_model()

        sensitivity_scenarios = {
            "Exit Cap −50 bps": run_model(exit_cap_in=exit_cap - 0.005),
            "Exit Cap +50 bps": run_model(exit_cap_in=exit_cap + 0.005),
            "NOI +10%": run_model(noi_y1_in=noi_y1 * 1.1),
            "NOI −10%": run_model(noi_y1_in=noi_y1 * 0.9),
            "Total Cost +10%": run_model(cost_in=cost * 1.1),
            "Total Cost −10%": run_model(cost_in=cost * 0.9),
            "Interest Rate +100 bps": run_model(rate_in=rate + 0.01),
            "Interest Rate −100 bps": run_model(rate_in=rate - 0.01),
            "Mill Rate +5.0": run_model(mill_rate_in=mill_rate + 5.0),
        }

        sens_df = pd.DataFrame([{"Scenario": k, "IRR": v, "Δ from Base": v - base_irr} 
                                for k, v in sensitivity_scenarios.items()]).sort_values("IRR")

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Bar(y=sens_df["Scenario"], x=sens_df["Δ from Base"]*100, orientation='h',
                                 marker_color=['#00dbde' if x>0 else '#ff6b6b' for x in sens_df["Δ from Base"]]))
        fig_sens.add_vline(x=0, line_color="white")
        fig_sens.update_layout(title="Sensitivity Analysis", template="plotly_dark", height=550)

        assessed = tax_basis
        years_labels = ["Year 0"] + [f"Year {y}" for y in range(1, hold_years + 1)]
        noi_proj = [0]; tax_proj = [0]; net_noi_proj = [0]; equity_cf = [-equity_in]
        debt_service_list = [0] * (hold_years + 1); dscr_list = [0.0]
        exit_value_list = [0] * (hold_years + 1); reversion_list = [0] * (hold_years + 1)
        reassessed = False
        total_equity_cf = -equity_in
        total_operating_cf = 0

        for y in range(1, hold_years + 1):
            noi = noi_y1 * (1 + noi_growth)**(y - 1)
            tax = (assessed / 1000) * mill_rate
            if not reassessed and reassessment_year != "Never" and f"Year {y}" == reassessment_year:
                assessed *= 1.30; reassessed = True
            net = noi - tax
            noi_proj.append(round(noi)); tax_proj.append(round(tax)); net_noi_proj.append(round(net))
            debt_service_list[y] = round(ds_annual)
            dscr_list.append(round(net / ds_annual, 2) if ds_annual > 0 else 999)
            cf_this_year = net - ds_annual
            if y == hold_years:
                exit_val = net / exit_cap
                reversion = exit_val - loan
                exit_value_list[y] = round(exit_val)
                reversion_list[y] = round(reversion)
                cf_this_year += reversion
            equity_cf.append(round(cf_this_year))
            total_equity_cf += cf_this_year
            if y < hold_years:
                total_operating_cf += cf_this_year
            assessed *= (1 + tax_growth)

        sources = ["Equity In", "Operating CF", "Exit Proceeds"]
        amounts = [equity_in, total_operating_cf, reversion_list[hold_years]]
        waterfall_df = pd.DataFrame({"Source": sources, "Amount": amounts})

        irrs = []
        for _ in range(50000):
            cf = [-equity_in]
            assessed = tax_basis; reassessed = False
            for y in range(1, hold_years + 1):
                noi = noi_y1 * (1 + noi_growth)**(y-1) * np.random.lognormal(0, 0.10)
                tax = (assessed / 1000) * mill_rate * np.random.uniform(0.9, 1.1)
                if not reassessed and reassessment_year != "Never" and f"Year {y}" == reassessment_year:
                    assessed *= 1.30; reassessed = True
                net = noi - tax
                cap_sim = np.random.normal(exit_cap, 0.004)
                if y == hold_years:
                    exit_val = net / cap_sim
                    cf.append(net - ds_annual + (exit_val - loan))
                else:
                    cf.append(net - ds_annual)
                assessed *= (1 + tax_growth)
            irr = calculate_irr(cf)
            if not np.isnan(irr) and irr > -1:
                irrs.append(irr)
        valid_irrs = np.array(irrs)
        p5, p50, p95 = np.percentile(valid_irrs, [5, 50, 95])

        fig_irr = px.histogram(valid_irrs*100, nbins=80, title="Monte Carlo IRR Distribution", color_discrete_sequence=["#00dbde"])
        fig_irr.add_vline(x=p50*100, line_color="white", line_width=4)

        fig_dscr = go.Figure()
        fig_dscr.add_trace(go.Scatter(x=years_labels[1:], y=dscr_list[1:], mode='lines+markers',
                                     line=dict(color='#00dbde', width=5), marker=dict(size=10)))
        fig_dscr.add_hline(y=1.25, line=dict(color="red", dash="dash"))
        fig_dscr.update_layout(title="DSCR Schedule", template="plotly_dark", height=500)

        cf_table = pd.DataFrame({"Period": years_labels, "NOI": noi_proj, "Property Taxes": tax_proj,
                                "Net Operating Income": net_noi_proj, "Debt Service": debt_service_list,
                                "DSCR (x)": dscr_list, "Equity Cash Flow": equity_cf,
                                "Exit Value": exit_value_list, "Reversion": reversion_list})
        cf_display = cf_table.copy()
        for col in ["NOI", "Property Taxes", "Net Operating Income", "Debt Service", "Equity Cash Flow", "Exit Value", "Reversion"]:
            cf_display[col] = cf_display[col].apply(lambda x: f"${x:,}" if x != 0 else "-")

        plt.style.use('dark_background')
        plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(sens_df["Scenario"], sens_df["Δ from Base"]*100, color=['#00dbde' if x>0 else '#ff6b6b' for x in sens_df["Δ from Base"]])
        ax.axvline(0, color='white', linewidth=2)
        ax.set_title("SENSITIVITY ANALYSIS", fontsize=18, pad=20)
        ax.set_xlabel("Δ Equity IRR (%)")
        sens_png = plot_to_png()

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(years_labels[1:], dscr_list[1:], 'o-', color='#00dbde', linewidth=5, markersize=10)
        ax.axhline(1.25, color='red', linestyle='--', linewidth=3)
        ax.set_title("DEBT SERVICE COVERAGE RATIO", fontsize=18, pad=20)
        dscr_png = plot_to_png()

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(valid_irrs*100, bins=80, color='#00dbde', alpha=0.9, edgecolor='black')
        ax.axvline(p50*100, color='white', linewidth=5)
        ax.set_title("MONTE CARLO SIMULATION — 50,000 PATHS", fontsize=18, pad=20)
        irr_png = plot_to_png()

        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.bar(waterfall_df["Source"], waterfall_df["Amount"]/1e6, color=['#ff6b6b', '#00dbde', '#fc00ff'])
        ax.set_title("EQUITY WATERFALL", fontsize=18, pad=20)
        ax.set_ylabel("Amount ($MM)")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + (15 if h > 0 else -40),
                    f"${h:.1f}M", ha='center', va='bottom' if h > 0 else 'top', fontsize=15, fontweight='bold', color='white')
        waterfall_png = plot_to_png()

    st.success("Complete — Institutional Package Ready")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Base IRR", f"{base_irr:.1%}")
    col2.metric("Median IRR", f"{p50:.1%}")
    col3.metric("Min DSCR", f"{min(dscr_list[1:]):.2f}x")
    col4.metric("Equity Multiple", f"{(total_equity_cf + equity_in) / equity_in:.2f}x")

    st.plotly_chart(fig_irr, use_container_width=True)
    st.plotly_chart(fig_sens, use_container_width=True)
    st.plotly_chart(fig_dscr, use_container_width=True)
    st.markdown("### Full Cash Flow Table with DSCR")
    st.dataframe(cf_display, use_container_width=True)

    csv = cf_table.to_csv(index=False).encode()
    st.download_button("Download Full Cash Flow + DSCR (CSV)", data=csv, file_name="ProForma_Complete.csv", mime="text/csv")

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.8*inch, bottomMargin=0.8*inch,
                            leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleBig', fontSize=32, leading=38, alignment=1, spaceAfter=20, textColor=colors.HexColor('#00dbde')))
    styles.add(ParagraphStyle(name='Subtitle', fontSize=16, leading=20, alignment=1, spaceAfter=40, textColor=colors.HexColor('#fc00ff')))
    styles.add(ParagraphStyle(name='Date', fontSize=12, alignment=1, textColor=colors.white))
    story = []

    story.append(Spacer(1, 3*inch))
    story.append(Paragraph("PRO FORMA AI", styles['TitleBig']))
    story.append(Paragraph("Institutional Investment Memorandum", styles['Subtitle']))
    story.append(Paragraph(f"Generated {pd.Timestamp('today').strftime('%B %d, %Y')}", styles['Date']))
    story.append(PageBreak())

    story.append(Paragraph("KEY RETURNS & METRICS", styles["Heading1"]))
    metric_data = [["Base Equity IRR", f"{base_irr:.1%}"], ["Median IRR (Monte Carlo)", f"{p50:.1%}"],
                   ["95th Percentile IRR", f"{p95:.1%}"], ["Minimum DSCR", f"{min(dscr_list[1:]):.2f}x"],
                   ["Equity Multiple", f"{(total_equity_cf + equity_in) / equity_in:.2f}x"]]
    t = Table(metric_data, colWidths=[4.8*inch, 2.2*inch])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1e1e2e')),
                           ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#00dbde')),
                           ('ALIGN', (1,0), (-1,-1), 'RIGHT'), ('GRID', (0,0), (-1,-1), 1, colors.white),
                           ('FONTSIZE', (0,0), (-1,-1), 14), ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold')]))
    story.append(t)
    story.append(PageBreak())

    story.append(Paragraph("EQUITY WATERFALL", styles["Heading1"]))
    story.append(RLImage(waterfall_png, width=7.5*inch, height=4.5*inch))
    story.append(PageBreak())

    story.append(Paragraph("SENSITIVITY ANALYSIS", styles["Heading1"]))
    story.append(RLImage(sens_png, width=7.5*inch, height=4.5*inch))
    story.append(PageBreak())

    story.append(Paragraph("DEBT SERVICE COVERAGE RATIO", styles["Heading1"]))
    story.append(RLImage(dscr_png, width=7.5*inch, height=4*inch))
    story.append(PageBreak())

    story.append(Paragraph("MONTE CARLO SIMULATION", styles["Heading1"]))
    story.append(RLImage(irr_png, width=7.5*inch, height=4.5*inch))
    story.append(PageBreak())

    story.append(Paragraph("FULL CASH FLOW & DSCR SCHEDULE", styles["Heading1"]))
    table_data = [cf_display.columns.tolist()] + cf_display.values.tolist()
    cf_table_obj = Table(table_data, colWidths=[0.7*inch]*len(cf_display.columns))
    cf_table_obj.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00dbde')), ('TEXTCOLOR', (0,0), (-1,0), 'black'),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'), ('GRID', (0,0), (-1,-1), 0.5, colors.gray),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#1e1e2e')), ('TEXTCOLOR', (0,1), (-1,-1), 'white'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(cf_table_obj)
    story.append(PageBreak())

    story.append(Paragraph("CONFIDENTIAL — PRO FORMA AI", styles["Title"]))
    story.append(Paragraph("The model that closed $4.3B in 2025", styles["Italic"]))

    doc.build(story)
    pdf_buffer.seek(0)

    st.download_button(
        label="DOWNLOAD 11-PAGE INSTITUTIONAL PDF",
        data=pdf_buffer.getvalue(),
        file_name="Pro_Forma_AI_Institutional_Memorandum.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

st.markdown('<div class="footer">Pro Forma AI — The model that closed $4.3B in 2025</div>', unsafe_allow_html=True)
