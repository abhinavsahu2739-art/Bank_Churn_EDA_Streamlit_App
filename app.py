"""
Bank Churn Analysis Dashboard
==============================
A complete EDA dashboard built with Streamlit, Pandas, NumPy,
Matplotlib, and Seaborn. Designed for college-level presentation.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (clean, professional, presentation-ready)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #a0aec0 !important;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Main background */
    .main { background: #f7f8fc; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; color: #1a1a2e; }
    .metric-card p  { margin: 0; color: #6b7280; font-size: 0.85rem; }

    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white !important;
        padding: 0.7rem 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Chart container */
    .chart-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 8px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #1e40af;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }

    /* Streamlit adjustments */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    h1 { color: #1a1a2e !important; font-weight: 700 !important; }
    h2, h3 { color: #1a1a2e !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

# Consistent colour palette used across all charts
PALETTE  = ["#3b82f6", "#ef4444"]       # blue = stayed, red = churned
HEATMAP_CMAP = "Blues"
FIG_BG   = "#ffffff"

plt.rcParams.update({
    "figure.facecolor":  FIG_BG,
    "axes.facecolor":    "#f9fafb",
    "axes.edgecolor":    "#d1d5db",
    "axes.labelcolor":   "#374151",
    "xtick.color":       "#6b7280",
    "ytick.color":       "#6b7280",
    "text.color":        "#374151",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "font.family":       "DejaVu Sans",
})


def section(title: str):
    st.markdown(f'<div class="section-header">📊 {title}</div>', unsafe_allow_html=True)


def metric_card(col, label: str, value):
    col.markdown(
        f'<div class="metric-card"><p>{label}</p><h3>{value}</h3></div>',
        unsafe_allow_html=True,
    )


def show_plot(fig):
    """Render a matplotlib figure inside a styled white card."""
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    plt.close(fig)


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def load_and_clean(uploaded_file) -> tuple[pd.DataFrame, dict]:
    """Load CSV, clean it, and return (cleaned_df, report_dict)."""
    df = pd.read_csv(uploaded_file)
    report = {}

    # --- Drop obvious ID / row-number columns ---
    id_like = [c for c in df.columns
               if any(kw in c.lower() for kw in ["customerid", "rownum", "surname"])]
    df.drop(columns=id_like, inplace=True, errors="ignore")
    report["dropped_cols"] = id_like

    # --- Duplicates ---
    n_dup = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = int(n_dup)

    # --- Missing values ---
    missing_before = df.isnull().sum().sum()
    # Numeric columns → fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # Categorical columns → fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    report["missing_filled"] = int(missing_before)

    # --- Standardise target column name ---
    target_candidates = [c for c in df.columns if "churn" in c.lower() or "exited" in c.lower()]
    if target_candidates:
        df.rename(columns={target_candidates[0]: "Churn"}, inplace=True)
    report["target"] = "Churn"

    # --- Rename common columns for consistency ---
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "age":             rename_map[c] = "Age"
        elif cl == "balance":       rename_map[c] = "Balance"
        elif "salary" in cl or "estimatedsalary" in cl: rename_map[c] = "EstimatedSalary"
        elif cl == "gender":        rename_map[c] = "Gender"
        elif cl == "geography":     rename_map[c] = "Geography"
    df.rename(columns=rename_map, inplace=True)

    return df, report


# ─────────────────────────────────────────────
# PLOT FUNCTIONS  (each returns a Figure)
# ─────────────────────────────────────────────

def plot_countplot(df: pd.DataFrame) -> plt.Figure:
    """1. Count Plot – Churn Distribution"""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Churn"].value_counts()
    labels = ["Stayed (0)", "Churned (1)"]
    bars = ax.bar(labels, counts.values, color=PALETTE, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Churn Distribution (Count)", pad=12)
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Number of Customers")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return fig


def plot_pie(df: pd.DataFrame) -> plt.Figure:
    """2. Pie Chart – Churn Percentage"""
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df["Churn"].value_counts()
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=["Stayed", "Churned"],
        colors=PALETTE,
        autopct="%1.1f%%",
        startangle=140,
        explode=(0, 0.06),
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight("bold")
    ax.set_title("Churn Percentage", pad=14)
    fig.tight_layout()
    return fig


def plot_histograms_subplots(df: pd.DataFrame) -> plt.Figure:
    """3 & 8. Histograms (Age, Balance, Salary) in a combined subplot grid."""
    hist_cols = [c for c in ["Age", "Balance", "EstimatedSalary"] if c in df.columns]
    n = len(hist_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]

    colors = ["#3b82f6", "#10b981", "#f59e0b"]
    for ax, col, color in zip(axes, hist_cols, colors):
        ax.hist(df[col].dropna(), bins=30, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(df[col].mean(), color="#1a1a2e", linestyle="--", linewidth=1.4, label=f"Mean: {df[col].mean():.0f}")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame) -> plt.Figure:
    """4. Box Plots – Balance & Age outlier detection."""
    box_cols = [c for c in ["Balance", "Age"] if c in df.columns]
    n = len(box_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    colors = ["#3b82f6", "#8b5cf6"]
    for ax, col, color in zip(axes, box_cols, colors):
        bp = ax.boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="#1a1a2e", linewidth=2),
            flierprops=dict(marker="o", markerfacecolor="#ef4444", markersize=4, alpha=0.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        ax.set_title(f"{col} – Box Plot (Outlier Detection)")
        ax.set_ylabel(col)
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Outlier Detection via Box Plots", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    """5. Correlation Heatmap – numeric columns only."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(max(7, len(corr) * 0.9), max(5, len(corr) * 0.75)))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=mask,
        linewidths=0.5,
        linecolor="#f0f0f0",
        annot_kws={"size": 9},
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", pad=14)
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame) -> plt.Figure:
    """6. Scatter Plot – Balance vs Age, coloured by Churn."""
    if "Balance" not in df.columns or "Age" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for churn_val, label, color in zip([0, 1], ["Stayed", "Churned"], PALETTE):
        subset = df[df["Churn"] == churn_val]
        ax.scatter(subset["Age"], subset["Balance"],
                   c=color, label=label, alpha=0.45, s=18, edgecolors="none")
    ax.set_title("Balance vs Age  (Coloured by Churn)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Balance")
    ax.legend(title="Churn Status", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_bar_churn_vs_cat(df: pd.DataFrame, cat_col: str) -> plt.Figure:
    """7. Bar Plot – Churn rate by a categorical column."""
    if cat_col not in df.columns:
        return None
    grouped = df.groupby(cat_col)["Churn"].value_counts(normalize=True).unstack(fill_value=0) * 100
    if 1 not in grouped.columns:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    categories = grouped.index.tolist()
    x = np.arange(len(categories))
    width = 0.35

    bars0 = ax.bar(x - width / 2, grouped.get(0, [0] * len(categories)), width,
                   label="Stayed", color=PALETTE[0], edgecolor="white")
    bars1 = ax.bar(x + width / 2, grouped.get(1, [0] * len(categories)), width,
                   label="Churned", color=PALETTE[1], edgecolor="white")

    for bar in list(bars0) + list(bars1):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"Churn Rate by {cat_col} (%)")
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15 if len(categories) > 4 else 0)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 Bank Churn\nAnalysis Dashboard")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "UPLOAD CSV FILE",
        type=["csv"],
        help="Upload a bank customer dataset with a Churn / Exited column.",
    )

    st.markdown("---")
    st.markdown("**NAVIGATION**")
    nav = st.radio(
        "Go to section",
        [
            "🏠 Overview",
            "🧹 Data Cleaning Report",
            "📋 Summary Statistics",
            "📊 Churn Distribution",
            "📈 Histograms & Box Plots",
            "🔥 Correlation Heatmap",
            "🔵 Scatter Plot",
            "📉 Churn by Category",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Built with Streamlit · Pandas · Seaborn · Matplotlib")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

st.title("🏦 Bank Churn Analysis Dashboard")
st.markdown("*Exploratory Data Analysis  |  College Group Project*")

if uploaded_file is None:
    # ── Welcome / landing state ──
    st.markdown(
        """
        <div class="info-box">
        👈  <strong>Upload a CSV file</strong> from the sidebar to begin analysis.<br>
        The dashboard expects a standard bank-churn dataset (e.g., Kaggle's
        <em>Churn_Modelling.csv</em>) with columns such as:
        <code>Age</code>, <code>Balance</code>, <code>Gender</code>,
        <code>Geography</code>, <code>EstimatedSalary</code>, <code>Exited / Churn</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    for col, icon, label, desc in [
        (col1, "📁", "Step 1", "Upload your CSV using the sidebar"),
        (col2, "🔍", "Step 2", "Navigate sections with the sidebar menu"),
        (col3, "📊", "Step 3", "Explore all charts & statistics"),
    ]:
        col.markdown(
            f'<div class="metric-card"><p>{icon} <strong>{label}</strong></p><h3 style="font-size:1rem;">{desc}</h3></div>',
            unsafe_allow_html=True,
        )
    st.stop()


# ─── Load & clean data ───────────────────────
df, clean_report = load_and_clean(uploaded_file)

# Guard: must have Churn column
if "Churn" not in df.columns:
    st.error("❌ Could not find a 'Churn' or 'Exited' column in your dataset. "
             "Please check your file and re-upload.")
    st.stop()


# ═══════════════════════════════════════════════
# SECTION: OVERVIEW
# ═══════════════════════════════════════════════
if nav == "🏠 Overview":
    section("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total Customers", f"{len(df):,}")
    metric_card(c2, "Features", str(df.shape[1]))
    churn_rate = df["Churn"].mean() * 100
    metric_card(c3, "Churn Rate", f"{churn_rate:.1f}%")
    metric_card(c4, "Retained Rate", f"{100 - churn_rate:.1f}%")

    st.subheader("Dataset Preview  (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.dtypes.index,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
    })
    st.dataframe(dtype_df, use_container_width=True)


# ═══════════════════════════════════════════════
# SECTION: DATA CLEANING REPORT
# ═══════════════════════════════════════════════
elif nav == "🧹 Data Cleaning Report":
    section("Data Cleaning Report")

    st.subheader("What was done automatically")

    steps = [
        ("🗑️ Irrelevant Columns Dropped",
         f"`{', '.join(clean_report['dropped_cols']) or 'None'}` — ID-like columns removed."),
        ("🔁 Duplicate Rows Removed",
         f"**{clean_report['duplicates_removed']}** duplicate rows were found and removed."),
        ("🩹 Missing Values Handled",
         f"**{clean_report['missing_filled']}** missing values filled "
         f"(numeric → median, categorical → mode)."),
        ("🎯 Target Column Standardised",
         f"Churn / Exited column renamed to **`Churn`** (0 = stayed, 1 = churned)."),
    ]

    for title, desc in steps:
        st.markdown(f"**{title}**")
        st.markdown(desc)
        st.markdown("---")

    # Missing values heatmap (post-clean — should be all zeros)
    section("Missing Values After Cleaning")
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({"Column": missing.index, "Missing Values": missing.values})
    st.dataframe(missing_df[missing_df["Missing Values"] > 0]
                 if missing.sum() > 0
                 else missing_df.head(10),
                 use_container_width=True)
    if missing.sum() == 0:
        st.success("✅ No missing values remain after cleaning.")


# ═══════════════════════════════════════════════
# SECTION: SUMMARY STATISTICS
# ═══════════════════════════════════════════════
elif nav == "📋 Summary Statistics":
    section("Descriptive Statistics")

    st.subheader("Numerical Summary  (describe)")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.subheader("Dataset Info  (formatted)")
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    # Format into a readable table
    lines = [l for l in info_str.split("\n") if l.strip()]
    st.code(info_str, language="text")

    section("Categorical Columns – Value Counts")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        chosen = st.selectbox("Select a categorical column:", cat_cols)
        vc = df[chosen].value_counts().reset_index()
        vc.columns = [chosen, "Count"]
        vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2).astype(str) + "%"
        st.dataframe(vc, use_container_width=True)
    else:
        st.info("No categorical columns found.")


# ═══════════════════════════════════════════════
# SECTION: CHURN DISTRIBUTION
# ═══════════════════════════════════════════════
elif nav == "📊 Churn Distribution":
    section("Churn Distribution")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Count Plot")
        show_plot(plot_countplot(df))

    with col_b:
        st.subheader("Pie Chart")
        show_plot(plot_pie(df))

    stayed  = (df["Churn"] == 0).sum()
    churned = (df["Churn"] == 1).sum()
    st.markdown(
        f"""
        <div class="info-box">
        📌 Out of <strong>{len(df):,}</strong> customers,
        <strong style="color:#3b82f6">{stayed:,}</strong> stayed and
        <strong style="color:#ef4444">{churned:,}</strong> churned
        — a churn rate of <strong>{churned/len(df)*100:.1f}%</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════
# SECTION: HISTOGRAMS & BOX PLOTS
# ═══════════════════════════════════════════════
elif nav == "📈 Histograms & Box Plots":
    section("Histograms  (Age, Balance, Salary)")
    st.markdown("The combined subplot below shows the distribution of all key numeric features.")
    show_plot(plot_histograms_subplots(df))

    section("Box Plots  – Outlier Detection")
    st.markdown("Box plots reveal the spread, median, and outliers for Balance and Age.")
    show_plot(plot_boxplots(df))


# ═══════════════════════════════════════════════
# SECTION: CORRELATION HEATMAP
# ═══════════════════════════════════════════════
elif nav == "🔥 Correlation Heatmap":
    section("Correlation Heatmap")
    st.markdown(
        "Only **numeric** columns are used. Values close to **+1** or **−1** "
        "indicate strong correlation; values near **0** indicate little relationship."
    )
    show_plot(plot_heatmap(df))

    # Top correlations with Churn
    num_df = df.select_dtypes(include=[np.number])
    if "Churn" in num_df.columns:
        corr_churn = num_df.corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
        st.subheader("Top Features Correlated with Churn")
        corr_df = corr_churn.reset_index()
        corr_df.columns = ["Feature", "Correlation with Churn"]
        corr_df["Correlation with Churn"] = corr_df["Correlation with Churn"].round(4)
        st.dataframe(corr_df, use_container_width=True)


# ═══════════════════════════════════════════════
# SECTION: SCATTER PLOT
# ═══════════════════════════════════════════════
elif nav == "🔵 Scatter Plot":
    section("Scatter Plot – Balance vs Age")
    st.markdown(
        "Each dot is a customer. **Blue** = stayed, **Red** = churned. "
        "Look for clusters or patterns that separate the two groups."
    )
    fig = plot_scatter(df)
    if fig:
        show_plot(fig)
    else:
        st.warning("⚠️ 'Balance' or 'Age' column not found in dataset.")


# ═══════════════════════════════════════════════
# SECTION: CHURN BY CATEGORY
# ═══════════════════════════════════════════════
elif nav == "📉 Churn by Category":
    section("Churn by Gender")
    fig_g = plot_bar_churn_vs_cat(df, "Gender")
    if fig_g:
        show_plot(fig_g)
    else:
        st.warning("⚠️ 'Gender' column not found in dataset.")

    section("Churn by Geography")
    fig_geo = plot_bar_churn_vs_cat(df, "Geography")
    if fig_geo:
        show_plot(fig_geo)
    else:
        st.warning("⚠️ 'Geography' column not found in dataset.")

    # Bonus: any remaining object columns
    extra_cats = [c for c in df.select_dtypes(include="object").columns
                  if c not in ("Gender", "Geography")]
    if extra_cats:
        section("Churn by Other Categorical Features")
        chosen = st.selectbox("Select column:", extra_cats)
        fig_extra = plot_bar_churn_vs_cat(df, chosen)
        if fig_extra:
            show_plot(fig_extra)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Bank Churn Analysis Dashboard  ·  Built with Streamlit, Pandas, NumPy, Matplotlib & Seaborn")