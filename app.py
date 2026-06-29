"""
SignalHealth — Industrial Sensor Anomaly Detection Dashboard
Manufacturing / Power Plant IoT Monitoring Platform
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignalHealth — Industrial Anomaly Detection",
    layout="wide",
    page_icon="📡",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_CONFIG = {
    "temperature": {"unit": "°C",    "mean": 85.0,  "std": 5.0,  "min": 70.0,  "max": 105.0},
    "pressure":    {"unit": "bar",   "mean": 12.0,  "std": 0.8,  "min":  9.5,  "max":  14.5},
    "vibration":   {"unit": "mm/s",  "mean":  2.5,  "std": 0.4,  "min":  0.5,  "max":   6.0},
    "current":     {"unit": "A",     "mean": 48.0,  "std": 3.0,  "min": 40.0,  "max":  60.0},
    "voltage":     {"unit": "V",     "mean": 415.0, "std": 10.0, "min": 390.0, "max": 440.0},
    "flow_rate":   {"unit": "L/min", "mean": 120.0, "std": 8.0,  "min":  90.0, "max": 150.0},
}
SENSOR_NAMES = list(SENSOR_CONFIG.keys())

ANOMALY_EVENTS = [
    {"t": 2000, "name": "Thermal Runaway",  "sensors": ["temperature", "current"],   "magnitude":  4.5},
    {"t": 5500, "name": "Bearing Fault",    "sensors": ["vibration",   "current"],   "magnitude":  5.0},
    {"t": 8000, "name": "Seal Failure",     "sensors": ["pressure",    "flow_rate"], "magnitude": -4.0},
]

N               = 10_000
ANOMALY_HALF_W  = 50
COLORS          = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_sensor_data():
    rng   = np.random.default_rng(42)
    start = datetime(2024, 1, 1)
    ts    = np.arange(N, dtype=float)

    data = {}
    for sensor, cfg in SENSOR_CONFIG.items():
        mu, sigma = cfg["mean"], cfg["std"]
        daily   = 0.30 * sigma * np.sin(2 * np.pi * ts / 24)
        weekly  = 0.15 * sigma * np.sin(2 * np.pi * ts / 168)
        noise   = rng.normal(0, sigma, N)
        data[sensor] = mu + daily + weekly + noise

    labels = np.zeros(N, dtype=int)
    for ev in ANOMALY_EVENTS:
        tc, mag = ev["t"], ev["magnitude"]
        gauss = mag * np.exp(-((ts - tc) ** 2) / (2 * 15 ** 2))
        for sensor in ev["sensors"]:
            data[sensor] += gauss * SENSOR_CONFIG[sensor]["std"]
        lo, hi = max(0, tc - ANOMALY_HALF_W), min(N, tc + ANOMALY_HALF_W + 1)
        labels[lo:hi] = 1

    timestamps = [start + timedelta(hours=int(i)) for i in range(N)]
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = "timestamp"
    return df, labels

df, labels = generate_sensor_data()
X_all = df[SENSOR_NAMES].values.astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION  (all three methods, cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_all_scores(ewma_alpha: float):
    X = X_all
    # Fit stats on first 1 000 normal points
    X_fit = X[:1000][labels[:1000] == 0]
    mu_fit = X_fit.mean(0)
    sd_fit = X_fit.std(0) + 1e-8

    # 1 — Z-Score
    Z         = np.abs((X - mu_fit) / sd_fit)
    z_raw     = Z.max(1)
    z_score   = np.clip(z_raw / 10.0, 0, 1)

    # 2 — EWMA
    ewma_mu = np.empty_like(X)
    ewma_mu[0] = mu_fit
    alpha = ewma_alpha
    for t in range(1, N):
        ewma_mu[t] = alpha * X[t] + (1 - alpha) * ewma_mu[t - 1]
    dev      = np.sqrt(((X - ewma_mu) ** 2).mean(1))
    dev_thr  = dev[:500].mean() + 3 * dev[:500].std()
    ew_score = np.clip(dev / (dev_thr * 2 + 1e-8), 0, 1)

    # 3 — PCA reconstruction error
    X_c   = X_fit - mu_fit
    cov   = np.cov(X_c.T)
    evals, evecs = np.linalg.eigh(cov)
    idx_s = np.argsort(evals)[::-1]
    V     = evecs[:, idx_s[:3]]
    Xc    = X - mu_fit
    Xrecon = Xc @ V @ V.T + mu_fit
    re_raw = np.sum((X - Xrecon) ** 2, 1)
    re_ref = re_raw[:500].mean() + 5 * re_raw[:500].std() + 1e-8
    pca_score = np.clip(re_raw / re_ref, 0, 1)

    ensemble = (z_score + ew_score + pca_score) / 3.0

    return z_score, ew_score, pca_score, ensemble, mu_fit, sd_fit, re_raw, V, mu_fit

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Control Panel")
selected_sensors   = st.sidebar.multiselect("Sensors", SENSOR_NAMES,
                                             default=["temperature", "vibration", "pressure"])
anomaly_threshold  = st.sidebar.slider("Anomaly Score Threshold", 0.10, 1.00, 0.50, 0.05)
window_size        = st.sidebar.slider("Rolling Window Size (hrs)", 12, 168, 48, 12)
ewma_alpha_val     = st.sidebar.slider("EWMA Alpha (α)", 0.01, 0.50, 0.10, 0.01)
cusum_k_mult       = st.sidebar.slider("CUSUM Slack K (× σ)", 0.10, 2.00, 0.50, 0.10)
cusum_h_mult       = st.sidebar.slider("CUSUM Threshold H (× σ)", 1.00, 10.0, 4.00, 0.50)
alert_min_gap      = st.sidebar.slider("Alert Suppression Gap (hrs)", 1, 48, 6, 1)

sensors_to_plot = selected_sensors if selected_sensors else SENSOR_NAMES

# Compute scores (re-runs only when ewma_alpha changes)
z_score, ew_score, pca_score, ensemble_score, mu_fit, sd_fit, pca_re, V_pca, mu_pca = \
    compute_all_scores(ewma_alpha_val)

# Build suppressed alert list
alert_indices_all = np.where(ensemble_score >= anomaly_threshold)[0]
suppressed_alerts = []
last_a = -9999
for ia in alert_indices_all:
    if ia - last_a >= alert_min_gap:
        suppressed_alerts.append(ia)
        last_a = ia

# Health Index
beta         = 0.00005
cum_re       = np.cumsum(pca_re)
hi_raw       = np.exp(-beta * cum_re)
hi_arr       = (hi_raw - hi_raw.min()) / (hi_raw.max() - hi_raw.min() + 1e-10) * 100
current_hi   = float(hi_arr[-1])

# ─────────────────────────────────────────────────────────────────────────────
# HEADER KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.title("📡 SignalHealth — Industrial Sensor Anomaly Detection")
st.caption("Manufacturing / Power Plant IoT Monitoring Platform  |  10 000-hr multi-sensor dataset")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Timestamps",   f"{N:,}")
k2.metric("Anomaly Rate",       f"{labels.mean()*100:.2f}%")
k3.metric("True Anomaly Events", len(ANOMALY_EVENTS))
k4.metric("Active Alerts",      len(suppressed_alerts))
k5.metric("Current HI",         f"{current_hi:.1f} / 100")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Sensor Data Explorer",
    "🔬 Statistical Process Control",
    "🤖 Anomaly Detection Models",
    "⚠️ Alert Management",
    "🔧 Predictive Maintenance",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SENSOR DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Sensor Data Explorer")

    t_lo, t_hi = st.slider("View Window (index)", 0, N - 1, (0, 2500), 50, key="t1_win")
    df_v      = df.iloc[t_lo:t_hi]
    lab_v     = labels[t_lo:t_hi]
    n_v       = len(df_v)

    # Multi-line time-series
    st.subheader("Multi-Sensor Time Series")
    n_plots = len(sensors_to_plot)
    fig1, axes1 = plt.subplots(n_plots, 1, figsize=(14, 2.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes1 = [axes1]

    for ax, sensor, color in zip(axes1, sensors_to_plot, COLORS):
        cfg = SENSOR_CONFIG[sensor]
        idx_x = np.arange(n_v)
        ax.plot(idx_x, df_v[sensor].values, color=color, lw=0.8, label=sensor)
        ax.axhline(cfg["mean"], color="gray", ls="--", lw=0.8, alpha=0.7, label="μ")
        ax.fill_between(idx_x, cfg["min"], cfg["max"],
                        alpha=0.07, color=color, label="Normal range")
        for ev in ANOMALY_EVENTS:
            ev_local = ev["t"] - t_lo
            if 0 < ev_local < n_v:
                ax.axvline(ev_local, color="red", lw=1.4, ls=":", alpha=0.85)
                ylim_top = ax.get_ylim()[1] if ax.get_ylim()[1] != ax.get_ylim()[0] else cfg["mean"] * 1.1
                ax.text(ev_local + max(3, n_v // 200), ylim_top * 0.98,
                        ev["name"], fontsize=6, color="red", rotation=90, va="top")
        ax.set_ylabel(f"{sensor}\n({cfg['unit']})", fontsize=8)
        ax.legend(fontsize=7, loc="upper right", ncol=3)
        ax.grid(True, alpha=0.3)

    axes1[-1].set_xlabel("Hours from window start")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # Correlation heatmap + distribution overlay
    col_h, col_d = st.columns(2)

    with col_h:
        st.subheader("Sensor Correlation Heatmap")
        corr = df[SENSOR_NAMES].corr().values
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(SENSOR_NAMES)))
        ax2.set_yticks(range(len(SENSOR_NAMES)))
        ax2.set_xticklabels(SENSOR_NAMES, rotation=45, ha="right", fontsize=8)
        ax2.set_yticklabels(SENSOR_NAMES, fontsize=8)
        for i in range(len(SENSOR_NAMES)):
            for j in range(len(SENSOR_NAMES)):
                ax2.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="black" if abs(corr[i, j]) < 0.65 else "white")
        plt.colorbar(im, ax=ax2, label="Pearson r")
        ax2.set_title("Cross-Sensor Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_d:
        st.subheader("Normal vs Anomaly Distribution")
        dist_sensor = st.selectbox("Sensor", sensors_to_plot, key="dist_sel")
        cfg_d  = SENSOR_CONFIG[dist_sensor]
        norm_v = df[dist_sensor].values[labels == 0]
        anom_v = df[dist_sensor].values[labels == 1]
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.hist(norm_v, bins=80, density=True, alpha=0.6, color="steelblue", label="Normal")
        ax3.hist(anom_v, bins=40, density=True, alpha=0.6, color="tomato",    label="Anomaly")
        ax3.axvline(cfg_d["mean"], color="navy", ls="--", lw=1.5, label=f"μ={cfg_d['mean']}")
        ax3.axvline(cfg_d["min"],  color="orange", ls=":",  lw=1.2)
        ax3.axvline(cfg_d["max"],  color="orange", ls=":",  lw=1.2, label="Normal range")
        ax3.set_xlabel(f"{dist_sensor} ({cfg_d['unit']})")
        ax3.set_ylabel("Density")
        ax3.set_title(f"{dist_sensor}: Normal vs Anomaly")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # Summary table
    st.subheader("Sensor Summary Statistics")
    rows_sum = []
    for s in SENSOR_NAMES:
        cfg = SENSOR_CONFIG[s]
        v   = df[s].values
        pct = ((v < cfg["min"]) | (v > cfg["max"])).mean() * 100
        rows_sum.append({
            "Sensor": s, "Unit": cfg["unit"],
            "Mean":   f"{v.mean():.2f}", "Std":    f"{v.std():.2f}",
            "Min":    f"{v.min():.2f}",  "Max":    f"{v.max():.2f}",
            "% Time in Alert": f"{pct:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows_sum), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATISTICAL PROCESS CONTROL
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Statistical Process Control (SPC)")

    spc_sensor = st.selectbox("Sensor for SPC", SENSOR_NAMES, key="spc_sel")
    cfg_spc    = SENSOR_CONFIG[spc_sensor]
    x_full     = df[spc_sensor].values
    mu_spc     = x_full[:500].mean()
    sd_spc     = x_full[:500].std() + 1e-8
    UCL = mu_spc + 3 * sd_spc
    LCL = mu_spc - 3 * sd_spc
    CL  = mu_spc

    # Equations
    st.subheader("Shewhart Control Chart")
    eq1, eq2, eq3 = st.columns(3)
    eq1.latex(r"UCL = \bar{x} + 3\sigma")
    eq2.latex(r"CL  = \bar{x}")
    eq3.latex(r"LCL = \bar{x} - 3\sigma")

    spc_start = st.slider("View start (index)", 0, N - 500, 1800, 50, key="spc_start")
    x_win     = x_full[spc_start : spc_start + 500]
    idx_w     = np.arange(len(x_win))
    ooc       = (x_win > UCL) | (x_win < LCL)

    fig_spc, ax_spc = plt.subplots(figsize=(14, 4))
    ax_spc.plot(idx_w, x_win, color="steelblue", lw=0.8, label=spc_sensor)
    ax_spc.axhline(UCL, color="red",   ls="--", lw=1.4, label=f"UCL = {UCL:.2f}")
    ax_spc.axhline(CL,  color="green", ls="-",  lw=1.2, label=f"CL  = {CL:.2f}")
    ax_spc.axhline(LCL, color="red",   ls="--", lw=1.4, label=f"LCL = {LCL:.2f}")
    ax_spc.scatter(idx_w[ooc], x_win[ooc], color="red", zorder=5, s=30, label="Out-of-control")
    ax_spc.fill_between(idx_w, LCL, UCL, alpha=0.05, color="green")
    ax_spc.set_xlabel("Sample index")
    ax_spc.set_ylabel(f"{spc_sensor} ({cfg_spc['unit']})")
    ax_spc.set_title(f"Shewhart X-bar Control Chart — {spc_sensor}")
    ax_spc.legend(fontsize=8); ax_spc.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_spc); plt.close(fig_spc)

    # Western Electric Rules
    st.subheader("Western Electric Rules Violation Detection")
    we_defs = [
        (r"\text{Rule 1:}\;|x_t - CL| > 3\sigma",                         "1 point beyond 3σ"),
        (r"\text{Rule 2: 9 consecutive points same side of } CL",           "Run of 9"),
        (r"\text{Rule 3: 6 consecutive points monotonically } \uparrow / \downarrow", "Trend of 6"),
        (r"\text{Rule 4: 14 alternating up/down points}",                   "Alternating 14"),
    ]
    for eq, desc in we_defs:
        c1, c2 = st.columns([3, 4])
        c1.latex(eq); c2.caption(desc)

    violations = []
    xc = x_win - CL

    # R1
    for i in np.where(np.abs(xc) > 3 * sd_spc)[0]:
        violations.append({"Idx": spc_start + i, "Rule": "R1 (>3σ)",
                            "Value": f"{x_win[i]:.3f}", "Description": "Beyond 3σ"})
    # R2
    sign_v = np.sign(xc)
    for i in range(len(sign_v) - 8):
        if sign_v[i] != 0 and np.all(sign_v[i:i+9] == sign_v[i]):
            violations.append({"Idx": spc_start + i, "Rule": "R2 (run-9)",
                                "Value": f"{x_win[i]:.3f}", "Description": "9 same side"})
    # R3
    for i in range(len(x_win) - 5):
        d = np.diff(x_win[i:i+6])
        if np.all(d > 0) or np.all(d < 0):
            violations.append({"Idx": spc_start + i, "Rule": "R3 (trend-6)",
                                "Value": f"{x_win[i]:.3f}", "Description": "6 monotone"})
    # R4
    for i in range(len(x_win) - 13):
        d = np.diff(x_win[i:i+14])
        if np.all(np.abs(np.diff(np.sign(d))) == 2):
            violations.append({"Idx": spc_start + i, "Rule": "R4 (alt-14)",
                                "Value": f"{x_win[i]:.3f}", "Description": "14 alternating"})

    df_viol = (pd.DataFrame(violations)
               .drop_duplicates(subset=["Idx", "Rule"])
               .head(60))
    st.metric("WE Violations in window", len(df_viol))
    if not df_viol.empty:
        st.dataframe(df_viol, use_container_width=True)
    else:
        st.info("No WE violations detected in current window.")

    # CUSUM
    st.subheader("CUSUM Chart")
    st.latex(r"C^+_t = \max\!\left(0,\;C^+_{t-1} + (x_t - \mu - K)\right)")
    st.latex(r"C^-_t = \max\!\left(0,\;C^-_{t-1} - (x_t - \mu + K)\right)")
    st.latex(r"\text{Alarm when }C^+_t > H \text{ or } C^-_t > H")

    K_v = cusum_k_mult * sd_spc
    H_v = cusum_h_mult * sd_spc
    Cp  = np.zeros(len(x_win))
    Cm  = np.zeros(len(x_win))
    for i in range(1, len(x_win)):
        Cp[i] = max(0.0, Cp[i-1] + (x_win[i] - mu_spc - K_v))
        Cm[i] = max(0.0, Cm[i-1] - (x_win[i] - mu_spc + K_v))
    cusum_alarm = (Cp > H_v) | (Cm > H_v)

    fig_cu, (ax_cu1, ax_cu2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    ax_cu1.plot(idx_w, Cp, color="tomato",    lw=0.9, label="C⁺")
    ax_cu1.plot(idx_w, Cm, color="steelblue", lw=0.9, label="C⁻")
    ax_cu1.axhline(H_v, color="black", ls="--", lw=1.2, label=f"H = {H_v:.2f}")
    ax_cu1.scatter(idx_w[cusum_alarm], Cp[cusum_alarm],
                   color="red", s=18, zorder=5, label="Alarm")
    ax_cu1.set_ylabel("CUSUM Statistic")
    ax_cu1.set_title(f"CUSUM Chart — {spc_sensor}")
    ax_cu1.legend(fontsize=8); ax_cu1.grid(True, alpha=0.3)

    ax_cu2.plot(idx_w, x_win, color="steelblue", lw=0.7)
    ax_cu2.axhline(UCL, color="red", ls="--", lw=1.0)
    ax_cu2.axhline(LCL, color="red", ls="--", lw=1.0)
    ax_cu2.set_xlabel("Sample index")
    ax_cu2.set_ylabel(f"{spc_sensor} ({cfg_spc['unit']})")
    ax_cu2.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_cu); plt.close(fig_cu)

    st.metric("CUSUM alarms in window", int(cusum_alarm.sum()))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION MODELS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Anomaly Detection Models")

    # Method equations
    col_z, col_e, col_p = st.columns(3)
    with col_z:
        st.subheader("1 · Z-Score")
        st.latex(r"z_t = \frac{|x_t - \mu|}{\sigma}")
        st.caption("Alert if z > 3  (worst sensor per timestamp)")
    with col_e:
        st.subheader("2 · EWMA")
        st.latex(r"\mu_t = \alpha x_t + (1-\alpha)\mu_{t-1}")
        st.caption("Alert on large deviation from adaptive mean")
    with col_p:
        st.subheader("3 · PCA Reconstruction Error")
        st.latex(r"RE_t = \|x_t - VV^\top x_t\|^2")
        st.caption("Top-3 principal components, high RE = anomaly")

    method_scores = {
        "Z-Score":  z_score,
        "EWMA":     ew_score,
        "PCA-RE":   pca_score,
        "Ensemble": ensemble_score,
    }
    m_colors = {"Z-Score": "tomato", "EWMA": "steelblue",
                "PCA-RE": "seagreen", "Ensemble": "purple"}

    # ROC & PR helper functions (pure NumPy)
    def compute_roc(y_true, scores, n_thr=200):
        thrs = np.linspace(0, 1, n_thr)
        fprs, tprs = [], []
        for thr in thrs:
            p  = (scores >= thr).astype(int)
            tp = ((p==1)&(y_true==1)).sum(); fp = ((p==1)&(y_true==0)).sum()
            fn = ((p==0)&(y_true==1)).sum(); tn = ((p==0)&(y_true==0)).sum()
            tprs.append(tp/(tp+fn+1e-9)); fprs.append(fp/(fp+tn+1e-9))
        return np.array(fprs), np.array(tprs)

    def compute_pr(y_true, scores, n_thr=200):
        thrs = np.linspace(0, 1, n_thr)
        precs, recs = [], []
        for thr in thrs:
            p  = (scores >= thr).astype(int)
            tp = ((p==1)&(y_true==1)).sum(); fp = ((p==1)&(y_true==0)).sum()
            fn = ((p==0)&(y_true==1)).sum()
            precs.append(tp/(tp+fp+1e-9)); recs.append(tp/(tp+fn+1e-9))
        return np.array(recs), np.array(precs)

    def auc_trap(x, y):
        o = np.argsort(x)
        return float(np.trapz(y[o], x[o]))

    col_roc, col_pr = st.columns(2)
    with col_roc:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        for name, sc in method_scores.items():
            fpr_r, tpr_r = compute_roc(labels, sc)
            ax_roc.plot(fpr_r, tpr_r, color=m_colors[name], lw=1.6,
                        label=f"{name} (AUC={auc_trap(fpr_r, tpr_r):.3f})")
        ax_roc.plot([0,1],[0,1],"k--", lw=0.7)
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves"); ax_roc.legend(fontsize=8); ax_roc.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_roc); plt.close(fig_roc)

    with col_pr:
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        for name, sc in method_scores.items():
            rec_r, prec_r = compute_pr(labels, sc)
            ax_pr.plot(rec_r, prec_r, color=m_colors[name], lw=1.6,
                       label=f"{name} (PR-AUC={auc_trap(rec_r, prec_r):.3f})")
        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curves")
        ax_pr.legend(fontsize=8); ax_pr.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_pr); plt.close(fig_pr)

    # Anomaly score timeline
    st.subheader("Anomaly Score Timeline vs True Labels")
    t3_lo = st.slider("Timeline start", 0, N-1200, 1500, 100, key="t3_lo")
    t3_hi = t3_lo + 1200
    idx_tl = np.arange(t3_lo, t3_hi)

    fig_tl, (ax_tl1, ax_tl2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    ax_tl1.plot(idx_tl, ensemble_score[t3_lo:t3_hi], color="purple",    lw=1.0, label="Ensemble")
    ax_tl1.plot(idx_tl, z_score[t3_lo:t3_hi],        color="tomato",    lw=0.6, alpha=0.6, label="Z-Score")
    ax_tl1.plot(idx_tl, ew_score[t3_lo:t3_hi],       color="steelblue", lw=0.6, alpha=0.6, label="EWMA")
    ax_tl1.plot(idx_tl, pca_score[t3_lo:t3_hi],      color="seagreen",  lw=0.6, alpha=0.6, label="PCA-RE")
    ax_tl1.axhline(anomaly_threshold, color="orange", ls="--", lw=1.2,
                   label=f"Threshold={anomaly_threshold}")
    for ev in ANOMALY_EVENTS:
        if t3_lo <= ev["t"] <= t3_hi:
            ax_tl1.axvline(ev["t"], color="red", ls=":", lw=1.6)
            ax_tl1.text(ev["t"]+10, 0.96, ev["name"], color="red",
                        fontsize=7, rotation=90, va="top")
    ax_tl1.set_ylabel("Anomaly Score [0–1]"); ax_tl1.set_ylim(0, 1.05)
    ax_tl1.legend(fontsize=7, ncol=3); ax_tl1.grid(True, alpha=0.3)

    ax_tl2.fill_between(idx_tl, labels[t3_lo:t3_hi], color="tomato", alpha=0.5, label="True Anomaly")
    ax_tl2.set_ylabel("True Label"); ax_tl2.set_xlabel("Timestamp Index")
    ax_tl2.set_ylim(-0.05, 1.3); ax_tl2.legend(fontsize=8); ax_tl2.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_tl); plt.close(fig_tl)

    # Precision / Recall / F1 table
    st.subheader("Precision / Recall / F1 at Various Thresholds")
    thr_list  = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    prf_rows  = []
    for name, sc in method_scores.items():
        for thr_v in thr_list:
            p  = (sc >= thr_v).astype(int)
            tp = ((p==1)&(labels==1)).sum(); fp = ((p==1)&(labels==0)).sum()
            fn = ((p==0)&(labels==1)).sum()
            pr = tp/(tp+fp+1e-9); rc = tp/(tp+fn+1e-9)
            f1 = 2*pr*rc/(pr+rc+1e-9)
            prf_rows.append({"Method": name, "Threshold": thr_v,
                             "Precision": f"{pr:.3f}", "Recall": f"{rc:.3f}", "F1": f"{f1:.3f}"})
    st.dataframe(pd.DataFrame(prf_rows), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALERT MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Alert Management")

    def severity(sc):
        if sc >= 0.9: return "Critical"
        if sc >= 0.7: return "High"
        if sc >= 0.5: return "Medium"
        return "Low"

    # Dominant sensor per timestamp
    Z_all = np.abs((X_all - mu_fit) / sd_fit)
    dom_sensor_idx = Z_all.argmax(1)

    # Build alert table
    alert_rows = []
    for ia in suppressed_alerts[:300]:
        sc_v  = float(ensemble_score[ia])
        s_nm  = SENSOR_NAMES[dom_sensor_idx[ia]]
        ts_v  = str(df.index[ia])
        alert_rows.append({
            "Timestamp":     ts_v,
            "Sensor":        s_nm,
            "Anomaly Score": round(sc_v, 4),
            "Severity":      severity(sc_v),
            "True Anomaly":  "Yes" if labels[ia] == 1 else "No",
        })
    df_alerts = pd.DataFrame(alert_rows)

    sev_cts = df_alerts["Severity"].value_counts().to_dict() if not df_alerts.empty else {}
    tp_cnt  = (df_alerts["True Anomaly"] == "Yes").sum() if not df_alerts.empty else 0

    ka1, ka2, ka3, ka4 = st.columns(4)
    ka1.metric("Total Alerts",    len(df_alerts))
    ka2.metric("Critical",        sev_cts.get("Critical", 0))
    ka3.metric("High",            sev_cts.get("High",     0))
    ka4.metric("True Positives",  tp_cnt)

    if not df_alerts.empty:
        st.dataframe(df_alerts, use_container_width=True)
    else:
        st.info("No alerts above current threshold.")

    # MTBF & MTTD
    st.subheader("MTBF & MTTD")
    col_m1, col_m2 = st.columns(2)

    tp_idx_list = [ia for ia in suppressed_alerts if labels[ia] == 1]
    if len(tp_idx_list) > 1:
        mtbf = float(np.diff(tp_idx_list).mean())
        col_m1.metric("MTBF (hrs between true-positive alerts)", f"{mtbf:.1f}")
    else:
        col_m1.metric("MTBF", "Insufficient data")

    mttd_vals = []
    for ev in ANOMALY_EVENTS:
        ev_start = ev["t"] - ANOMALY_HALF_W
        first_hit = next((ia for ia in suppressed_alerts
                          if ia >= ev_start and labels[ia] == 1), None)
        if first_hit is not None:
            mttd_vals.append(first_hit - ev_start)
    mttd = float(np.mean(mttd_vals)) if mttd_vals else float("nan")
    col_m2.metric("MTTD — Mean Time To Detect (hrs)",
                  f"{mttd:.1f}" if not np.isnan(mttd) else "N/A")

    # Rolling FPR
    st.subheader("Rolling 7-Day False Positive Rate (168-hr window)")
    roll_w   = 168
    fpr_roll = np.zeros(N)
    for i in range(roll_w, N):
        wl = labels[i-roll_w:i]
        wp = (ensemble_score[i-roll_w:i] >= anomaly_threshold).astype(int)
        fp = ((wp==1)&(wl==0)).sum(); tn = ((wp==0)&(wl==0)).sum()
        fpr_roll[i] = fp / (fp + tn + 1e-9)

    fig_fpr, ax_fpr = plt.subplots(figsize=(14, 3))
    ax_fpr.plot(fpr_roll[roll_w:], color="tomato", lw=0.7, label="Rolling FPR (7-day)")
    ax_fpr.axhline(fpr_roll[roll_w:].mean(), color="navy", ls="--", lw=1.2,
                   label=f"Mean FPR = {fpr_roll[roll_w:].mean():.3f}")
    for ev in ANOMALY_EVENTS:
        ax_fpr.axvline(ev["t"] - roll_w, color="red", ls=":", lw=1.2, alpha=0.7)
    ax_fpr.set_xlabel("Timestamp Index"); ax_fpr.set_ylabel("False Positive Rate")
    ax_fpr.set_title("Rolling 7-Day FPR")
    ax_fpr.legend(fontsize=8); ax_fpr.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_fpr); plt.close(fig_fpr)

    # Shift aggregation
    st.subheader("Alert Count by 8-Hour Shift")

    def shift_label(hr):
        if  6 <= hr < 14: return "Day (06–14)"
        if 14 <= hr < 22: return "Evening (14–22)"
        return "Night (22–06)"

    shift_cts = {"Day (06–14)": 0, "Evening (14–22)": 0, "Night (22–06)": 0}
    for ia in suppressed_alerts:
        shift_cts[shift_label(df.index[ia].hour)] += 1

    fig_sh, ax_sh = plt.subplots(figsize=(7, 4))
    sh_names = list(shift_cts.keys())
    sh_vals  = [shift_cts[s] for s in sh_names]
    bars_sh  = ax_sh.bar(sh_names, sh_vals, color=["#f39c12","#3498db","#2c3e50"], width=0.5)
    for bar, v in zip(bars_sh, sh_vals):
        ax_sh.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   str(v), ha="center", fontsize=11, fontweight="bold")
    ax_sh.set_ylabel("Alert Count"); ax_sh.set_title("Alerts per 8-Hour Production Shift")
    ax_sh.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); st.pyplot(fig_sh); plt.close(fig_sh)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICTIVE MAINTENANCE
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Predictive Maintenance")

    # ── RUL ──────────────────────────────────────────────────────────────────
    st.subheader("Remaining Useful Life (RUL) Estimation")
    st.latex(r"RUL = \frac{x_{\text{fail}} - x_{\text{current}}}{\dot{x}_{\text{degrad}}}")

    vib_vals = df["vibration"].values
    bear_t   = ANOMALY_EVENTS[1]["t"]                        # t=5500
    w_lo, w_hi = bear_t - ANOMALY_HALF_W, bear_t + ANOMALY_HALF_W
    t_win    = np.arange(w_lo, w_hi, dtype=float)
    v_win    = vib_vals[w_lo:w_hi]
    coefs    = np.polyfit(t_win, v_win, 1)
    degrad   = coefs[0]
    fail_thr = SENSOR_CONFIG["vibration"]["max"]
    cur_vib  = float(vib_vals[-100])
    rul_hrs  = max(0.0, (fail_thr - cur_vib) / degrad) if degrad > 0 else float("inf")
    rul_days = rul_hrs / 24.0 if rul_hrs != float("inf") else float("inf")

    col_rul1, col_rul2 = st.columns([1, 2])
    with col_rul1:
        st.metric("Degradation Rate (mm/s / hr)", f"{degrad:.5f}")
        st.metric("Current Vibration (mm/s)",      f"{cur_vib:.3f}")
        st.metric("Failure Threshold (mm/s)",       f"{fail_thr}")
        if rul_hrs == float("inf"):
            st.metric("Estimated RUL", "> 365 days (stable)")
        else:
            st.metric("Estimated RUL (hours)", f"{rul_hrs:.0f}")
            st.metric("Estimated RUL (days)",  f"{rul_days:.1f}")

    with col_rul2:
        rul_pct  = min(rul_days / 90.0, 1.0) if rul_days != float("inf") else 1.0
        rul_col  = "#2ecc71" if rul_pct > 0.6 else "#f39c12" if rul_pct > 0.3 else "#e74c3c"
        theta_bg = np.linspace(np.pi, 0, 300)
        theta_fg = np.linspace(np.pi, np.pi - rul_pct * np.pi, 300)
        fig_g, ax_g = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        ax_g.plot(np.cos(theta_bg), np.sin(theta_bg), color="lightgray", lw=14, solid_capstyle="round")
        ax_g.plot(np.cos(theta_fg), np.sin(theta_fg), color=rul_col,    lw=14, solid_capstyle="round")
        lbl = f"{rul_days:.0f}d" if rul_days != float("inf") else "Stable"
        ax_g.text(0, -0.05, lbl,            ha="center", va="center", fontsize=30,
                  fontweight="bold", color=rul_col)
        ax_g.text(0, -0.42, "Remaining Useful Life", ha="center", va="center",
                  fontsize=9, color="gray")
        ax_g.set_xlim(-1.2, 1.2); ax_g.set_ylim(-0.65, 1.2); ax_g.axis("off")
        ax_g.set_title("RUL Gauge (relative to 90-day horizon)", fontsize=9)
        plt.tight_layout(); st.pyplot(fig_g); plt.close(fig_g)

    # ── Health Index ──────────────────────────────────────────────────────────
    st.subheader("Equipment Health Index (HI)")
    st.latex(r"HI_t = 100 \cdot \frac{\exp(-\beta \sum_{i=0}^{t} RE_i) - \min}{\max - \min}")

    fig_hi, ax_hi = plt.subplots(figsize=(14, 3))
    ax_hi.plot(hi_arr, color="seagreen", lw=0.8, label="Health Index")
    ax_hi.fill_between(range(N), hi_arr, alpha=0.12, color="seagreen")
    for lvl, col, lbl in [(80,"green","Watch (80)"),(60,"orange","Warning (60)"),
                          (40,"orangered","High Risk (40)"),(20,"red","Critical (20)")]:
        ax_hi.axhline(lvl, color=col, ls="--", lw=1.0, label=lbl)
    for ev in ANOMALY_EVENTS:
        ax_hi.axvline(ev["t"], color="red", ls=":", lw=1.4, alpha=0.7)
        ax_hi.text(ev["t"]+60, 3, ev["name"], color="red", fontsize=7, rotation=90, va="bottom")
    ax_hi.set_xlabel("Timestamp Index"); ax_hi.set_ylabel("HI [0–100]")
    ax_hi.set_title("Equipment Health Index Over Time")
    ax_hi.set_ylim(0, 108); ax_hi.legend(fontsize=7, ncol=4); ax_hi.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_hi); plt.close(fig_hi)

    # ── Maintenance Decision ──────────────────────────────────────────────────
    st.subheader("Maintenance Decision Matrix")
    st.metric("Current Health Index", f"{current_hi:.1f} / 100")

    def maint_action(hi):
        if hi > 80:  return "Normal",    "Next scheduled maintenance in 30 days",  "#2ecc71"
        if hi > 60:  return "Watch",     "Inspect within 14 days",                 "#3498db"
        if hi > 40:  return "Warning",   "Inspect within 7 days",                  "#f39c12"
        if hi > 20:  return "High Risk", "Inspect within 72 hours",                "#e67e22"
        return "Critical", "Immediate shutdown required",                           "#e74c3c"

    status_lbl, action_str, _ = maint_action(current_hi)

    maint_table = [
        {"HI Range": "HI > 80",  "Status": "Normal",    "Recommended Action": "Next maintenance in 30 days"},
        {"HI Range": "60 – 80",  "Status": "Watch",     "Recommended Action": "Inspect in 14 days"},
        {"HI Range": "40 – 60",  "Status": "Warning",   "Recommended Action": "Inspect in 7 days"},
        {"HI Range": "20 – 40",  "Status": "High Risk", "Recommended Action": "Inspect within 72 hours"},
        {"HI Range": "HI < 20",  "Status": "Critical",  "Recommended Action": "Immediate shutdown"},
    ]
    st.dataframe(pd.DataFrame(maint_table), use_container_width=True)
    st.info(f"**Current Status: {status_lbl}** — {action_str}  (HI = {current_hi:.1f})")

    # ── Cost-Benefit Analysis ─────────────────────────────────────────────────
    st.subheader("Cost-Benefit Analysis")
    cb1, cb2 = st.columns(2)
    cost_fail  = cb1.number_input("Cost of Unplanned Failure ($)", value=250_000, step=10_000)
    cost_prev  = cb1.number_input("Cost of Preventive Maintenance ($)", value=15_000, step=1_000)
    p_no_maint = cb2.slider("P(failure | no maintenance)",   0.0, 1.0, 0.35, 0.01)
    p_maint    = cb2.slider("P(failure | maintenance done)", 0.0, 1.0, 0.03, 0.01)

    exp_no  = p_no_maint * cost_fail
    exp_yes = cost_prev + p_maint * cost_fail
    savings = exp_no - exp_yes
    roi     = savings / cost_prev * 100 if cost_prev > 0 else 0

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("E[Cost] — No Maint.",   f"${exp_no:,.0f}")
    cc2.metric("E[Cost] — With Maint.", f"${exp_yes:,.0f}")
    cc3.metric("Expected Savings",      f"${savings:,.0f}")
    cc4.metric("Maint. ROI",            f"{roi:.0f}%")

    fig_cb, ax_cb = plt.subplots(figsize=(7, 4))
    cb_cats = ["No Maintenance\n(Expected Cost)", "Preventive\nMaintenance"]
    cb_vals = [exp_no, exp_yes]
    cb_col  = ["#e74c3c", "#2ecc71"] if exp_no >= exp_yes else ["#2ecc71", "#e74c3c"]
    bars_cb = ax_cb.bar(cb_cats, cb_vals, color=cb_col, width=0.5)
    for bar, v in zip(bars_cb, cb_vals):
        ax_cb.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                   f"${v:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_cb.set_ylabel("Expected Cost ($)"); ax_cb.set_title("Cost-Benefit Analysis")
    ax_cb.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); st.pyplot(fig_cb); plt.close(fig_cb)

    # ── Gantt Maintenance Schedule ────────────────────────────────────────────
    st.subheader("Maintenance Schedule — Gantt Chart")
    gantt = [
        {"Task": "Bearing Inspection",     "Start":  2, "Duration":  4, "Type": "Planned"},
        {"Task": "Seal Replacement",       "Start": 10, "Duration":  8, "Type": "Planned"},
        {"Task": "Thermal Check (Unplan)", "Start":  0, "Duration":  2, "Type": "Actual"},
        {"Task": "Vibration Re-balance",   "Start": 22, "Duration":  6, "Type": "Planned"},
        {"Task": "Pressure System Audit",  "Start": 36, "Duration": 12, "Type": "Planned"},
        {"Task": "Full Overhaul",          "Start": 60, "Duration": 24, "Type": "Planned"},
    ]
    g_colors = {"Planned": "#3498db", "Actual": "#e67e22"}

    fig_gnt, ax_gnt = plt.subplots(figsize=(12, 5))
    for i, task in enumerate(gantt):
        col_g = g_colors[task["Type"]]
        ax_gnt.barh(i, task["Duration"], left=task["Start"],
                    color=col_g, alpha=0.78, edgecolor="white", height=0.6)
        ax_gnt.text(task["Start"] + task["Duration"]/2, i,
                    task["Task"], ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")
    ax_gnt.set_yticks(range(len(gantt)))
    ax_gnt.set_yticklabels([t["Task"] for t in gantt], fontsize=8)
    ax_gnt.set_xlabel("Hours from Now")
    ax_gnt.set_title("Maintenance Schedule — Gantt View")
    ax_gnt.set_xlim(0, 95)
    planned_p = mpatches.Patch(color="#3498db", alpha=0.78, label="Planned")
    actual_p  = mpatches.Patch(color="#e67e22", alpha=0.78, label="Actual / Emergency")
    ax_gnt.legend(handles=[planned_p, actual_p], fontsize=8, loc="lower right")
    ax_gnt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); st.pyplot(fig_gnt); plt.close(fig_gnt)

    # ── OEE KPIs ─────────────────────────────────────────────────────────────
    st.subheader("OEE — Overall Equipment Effectiveness")
    st.latex(r"OEE = Availability \times Performance \times Quality")

    downtime_hrs  = len(suppressed_alerts) * 0.5          # assume 30 min per alert
    availability  = max(0.0, (N - downtime_hrs) / N)
    performance   = max(0.0, 1.0 - float(ensemble_score.mean()))
    quality       = max(0.0, 1.0 - float(labels.mean()))
    oee           = availability * performance * quality

    ko1, ko2, ko3, ko4 = st.columns(4)
    ko1.metric("OEE",          f"{oee*100:.1f}%")
    ko2.metric("Availability", f"{availability*100:.1f}%")
    ko3.metric("Performance",  f"{performance*100:.1f}%")
    ko4.metric("Quality",      f"{quality*100:.1f}%")

    fig_oee, ax_oee = plt.subplots(figsize=(7, 4))
    oee_cats = ["OEE", "Availability", "Performance", "Quality"]
    oee_vals = [oee*100, availability*100, performance*100, quality*100]
    oee_cols = ["#9b59b6", "#3498db", "#2ecc71", "#f39c12"]
    bars_oee = ax_oee.bar(oee_cats, oee_vals, color=oee_cols, width=0.5)
    for bar, v in zip(bars_oee, oee_vals):
        ax_oee.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_oee.axhline(85, color="green", ls="--", lw=1.3, label="World-class OEE (85%)")
    ax_oee.set_ylabel("Percentage (%)"); ax_oee.set_title("OEE Component Breakdown")
    ax_oee.set_ylim(0, 112); ax_oee.legend(fontsize=8)
    ax_oee.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); st.pyplot(fig_oee); plt.close(fig_oee)

    st.caption("SignalHealth v1.0  |  Pure NumPy/Pandas/Matplotlib/Streamlit  |  Industrial IoT Anomaly Detection")
