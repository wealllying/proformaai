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
import os
import concurrent.futures
import multiprocessing
from typing import Tuple, List

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
# INPUTS (sidebar)
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
        # default two tiers
        tier1_hurdle = st.number_input("Tier1 Hurdle IRR (%)", value=12.0, step=0.5) / 100.0
        tier1_gp = st.number_input("Tier1 GP % of residual", value=30.0, step=1.0) / 100.0
        tier2_hurdle = st.number_input("Tier2 Hurdle IRR (%)", value=20.0, step=0.5) / 100.0
        tier2_gp = st.number_input("Tier2 GP % of residual", value=50.0, step=1.0) / 100.0
        promote_tiers = [(tier1_hurdle, tier1_gp), (tier2_hurdle, tier2_gp)]
    else:
        promote_tiers = None

    st.header("Monte Carlo & Performance")
    n_sims = st.number_input("Monte Carlo sims", min_value=500, max_value=200_000, value=5_000, step=500)
    sigma_rent = st.slider("Rent vol (σ)", 0.0, 0.25, 0.02, 0.005)
    sigma_opex = st.slider("OpEx vol (σ)", 0.0, 0.25, 0.015, 0.005)
    sigma_cap = st.slider("Cap vol (σ)", 0.0, 0.10, 0.004, 0.001)
    # correlation matrix (rent, opex, cap)
    corr = np.array([[1.0, 0.2, -0.4],
                     [0.2, 1.0, -0.2],
                     [-0.4, -0.2, 1.0]])

# ---------------------------
# Helper functions
# ---------------------------
def robust_irr(cfs):
    """Robust IRR wrapper: use npf.irr + bisection fallback."""
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
    """Return arrays for balances, interest, principal, payment for 'years' periods (annualized)."""
    balances = []
    interests = []
    principals = []
    payments = []
    bal = loan
    if amort_years == 0:
        # IO for all years
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

# Settlement engine: final settlement at exit (IRR-hurdle multi-tier)
def settle_final_distribution(lp_cf_so_far: List[float], gp_cf_so_far: List[float],
                              remaining_residual: float, equity_lp: float,
                              promote_tiers) -> Tuple[float, float]:
    """
    Returns (lp_add, gp_add) allocation of remaining_residual according to multi-tier IRR hurdles.
    """
    if remaining_residual <= 0 or promote_tiers is None or len(promote_tiers) == 0:
        lp_share = 0.8
        return remaining_residual * lp_share, remaining_residual * (1 - lp_share)

    lp_add_total = 0.0
    gp_add_total = 0.0
    residual_left = remaining_residual
    lp_so_far = lp_cf_so_far.copy()
    gp_so_far = gp_cf_so_far.copy()

    for (hurdle, gp_pct) in promote_tiers:
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

# Per-period waterfall (ROC -> PREF -> Catch-up -> Residual accumulation)
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

# Build deterministic per-period cash flows and collect residual for final settlement
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
    lp_roc_remaining = equity_lp
    lp_pref_accrued = 0.0
    residual_accumulator = 0.0

    balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)
    bal = senior_loan

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

# ---------------------------
# Parallel Monte Carlo worker (module-level for multiprocessing)
# ---------------------------
def _monte_worker(seed: int,
                  sims: int,
                  senior_loan: float,
                  senior_rate: float,
                  senior_amort: int,
                  senior_io: int,
                  gpr_y1: float,
                  rent_growth: float,
                  vacancy: float,
                  opex_y1: float,
                  opex_growth: float,
                  reserves: float,
                  hold: int,
                  exit_cap: float,
                  selling_costs: float,
                  pref_annual: float,
                  catchup_pct: float,
                  sigma_rent: float,
                  sigma_opex: float,
                  sigma_cap: float,
                  corr: np.ndarray,
                  promote_tiers,
                  lp_share_default: float) -> Tuple[List[float], int]:
    """
    Worker executes 'sims' Monte Carlo paths and returns (irrs_list, dscr_breach_count)
    Deterministic inputs must be passed in (no closures) so this can run in child processes.
    """
    rng = np.random.RandomState(seed)
    # build cov and cholesky
    cov = np.diag([sigma_rent**2, sigma_opex**2, sigma_cap**2])
    cov = np.sqrt(cov) @ corr @ np.sqrt(cov)
    try:
        L = np.linalg.cholesky(cov)
    except Exception:
        L = np.diag([sigma_rent, sigma_opex, sigma_cap])

    equity_total = (purchase_price * (1 + closing_costs_pct)) - senior_loan - (mezz_pct * purchase_price * (1 + closing_costs_pct) if use_mezz else 0.0)
    equity_lp = equity_total * lp_share_default

    local_irrs = []
    local_breaches = 0

    # amort schedule precompute
    balances_template, interests_template, principals_template, payments_template = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)

    for i in range(sims):
        z = rng.normal(size=3)
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

        # copy payments for per-sim usage
        payments = payments_template

        for y in range(1, hold+1):
            gpr = gpr_y1 * ((1 + rent_growth) ** (y-1)) * rent_shock
            v = float(np.clip(vacancy + rng.normal(0, 0.01), 0.0, 0.9))
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
            local_irrs.append(irr_sim)

        if any(d < 1.2 for d in dscr_path):
            local_breaches += 1

    return local_irrs, local_breaches

# ---------------------------
# Parallel run_montecarlo (integrated)
# ---------------------------
def run_montecarlo_parallel(n_sims: int, max_workers: int = None):
    """
    Parallel Monte Carlo runner using ProcessPoolExecutor.
    Returns (np.array(irrs), dscr_breach_count)
    """
    if n_sims <= 0:
        return np.array([]), 0

    # Determine worker count
    cpu_count = multiprocessing.cpu_count()
    if max_workers is None:
        max_workers = cpu_count
    max_workers = min(max_workers, cpu_count)
    # don't create more workers than sims
    workers = min(max_workers, n_sims)

    # chunk sims into roughly-even pieces
    base = n_sims // workers
    extras = n_sims % workers
    sims_chunks = [base + (1 if i < extras else 0) for i in range(workers)]
    seeds = [np.random.randint(0, 2**31 - 1) for _ in range(workers)]

    # prepare args per worker (pack all required inputs)
    senior_loan = total_cost * senior_ltv
    senior_rate_local = senior_rate
    senior_amort_local = senior_amort
    senior_io_local = senior_io

    worker_args = []
    for chunk, seed in zip(sims_chunks, seeds):
        if chunk == 0:
            continue
        worker_args.append((
            seed,
            chunk,
            senior_loan,
            senior_rate_local,
            senior_amort_local,
            senior_io_local,
            gpr_y1,
            rent_growth,
            vacancy,
            opex_y1,
            opex_growth,
            reserves,
            hold,
            exit_cap,
            selling_costs,
            pref_annual,
            catchup_pct,
            sigma_rent,
            sigma_opex,
            sigma_cap,
            corr,
            promote_tiers,
            lp_share_default
        ))

    irrs_agg = []
    breach_agg = 0

    progress = st.progress(0)
    completed = 0

    # run in parallel processes
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(_monte_worker, *args) for args in worker_args]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    local_irrs, local_breaches = fut.result()
                except Exception as e:
                    # fallback: if a worker failed, treat its sims as no-results
                    local_irrs, local_breaches = [], 0
                    st.warning(f"A worker failed: {e}")
                irrs_agg.extend(local_irrs)
                breach_agg += local_breaches
                completed += 1
                progress.progress(min(1.0, completed / len(futures)))
    except Exception as e:
        # if multiprocessing fails (common on some platforms), fallback to serial run
        st.warning(f"Parallel execution failed: {e}. Falling back to serial execution.")
        irrs_agg = []
        breach_agg = 0
        # serial simulation (reuse previous logic)
        for seed, chunk in zip(seeds, sims_chunks):
            local_irrs, local_breaches = _monte_worker(seed, chunk,
                                                       senior_loan, senior_rate_local,
                                                       senior_amort_local, senior_io_local,
                                                       gpr_y1, rent_growth, vacancy,
                                                       opex_y1, opex_growth, reserves,
                                                       hold, exit_cap, selling_costs,
                                                       pref_annual, catchup_pct,
                                                       sigma_rent, sigma_opex, sigma_cap,
                                                       corr, promote_tiers, lp_share_default)
            irrs_agg.extend(local_irrs)
            breach_agg += local_breaches
            completed += 1
            progress.progress(min(1.0, completed / len(sims_chunks)))

    progress.empty()
    return np.array(irrs_agg), breach_agg

# CSV sample generator (for reconciliation)
def generate_sample_csv(cf_table):
    buf = io.StringIO()
    cf_table.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue().encode()

# ---------------------------
# UI Buttons & Run
# ---------------------------
if st.button("Run Full Institutional Model (Deterministic + Monte Carlo)"):
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
    col1, col2, col3 = st.columns(3)
    col1.metric("LP IRR (det)", f"{lp_irr:.2%}" if not math.isnan(lp_irr) else "N/A")
    col2.metric("Min DSCR", f"{min(dscr_path):.2f}x" if dscr_path else "N/A")
    col3.metric("Exit Value (net)", f"${exit_value*(1-selling_costs):,.0f}")

    st.markdown("### Deterministic Cashflow Table (LP)")
    st.dataframe(cf_table)

    # CSV download
    csv_bytes = generate_sample_csv(cf_table)
    st.download_button("Download deterministic CF CSV", csv_bytes, "deterministic_cf.csv", "text/csv")

    # Monte Carlo (warn about time)
    st.info(f"Running Monte Carlo with {n_sims} sims — this may take time. Use fewer sims for quick iteration.")
    with st.spinner("Running Monte Carlo..."):
        # Max workers default to CPU count for maximal parallelism
        irrs, breaches = run_montecarlo_parallel(int(n_sims), max_workers=None)

    if irrs.size == 0:
        st.error("Monte Carlo produced no valid IRRs. Check inputs.")
    else:
        p5, p50, p95 = np.percentile(irrs, [5, 50, 95])
        st.subheader("Monte Carlo Results (LP IRR)")
        col1, col2, col3 = st.columns(3)
        col1.metric("P5", f"{p5:.2%}")
        col2.metric("P50", f"{p50:.2%}")
        col3.metric("P95", f"{p95:.2%}")
        st.metric("Probability DSCR < 1.2", f"{breaches / max(1, int(n_sims)):.1%}")

        fig = px.histogram(irrs*100, nbins=80, title="LP IRR Distribution (Monte Carlo)")
        fig.add_vline(x=p50*100, line_color="white", line_width=3)
        st.plotly_chart(fig, use_container_width=True)

    # Prepare payload & optional PDF call (best-effort)
    payload = {
        "date": datetime.today().strftime("%B %d, %Y"),
        "p50": f"{p50:.1%}" if irrs.size else "N/A",
        "p95": f"{p95:.1%}" if irrs.size else "N/A",
        "min_dscr": f"{min(dscr_path):.2f}x" if dscr_path else "N/A",
    }
    try:
        response = requests.post("https://proforma-ai-production.up.railway.app/api/pdf", json=payload, timeout=30)
        if response.status_code == 200:
            st.success("PDF generated on service")
    except Exception:
        st.info("PDF service unavailable (skipped).")

st.markdown("---")
st.info("This is an institutional-grade model with multi-tier promote settlement and test scaffolding. Adjust inputs in the sidebar and rerun.")
