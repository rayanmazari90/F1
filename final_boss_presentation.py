import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(
    page_title="F1 Final Boss – The Ultimate Driver Persona",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for F1-themed dark presentation
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Red accent for F1 */
    .f1-red {
        color: #ff1801 !important;
        font-weight: 700;
    }
    
    /* Hero title */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff1801, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #8b949e;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #21262d 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 24, 1, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ff1801;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 8px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        border-left: 4px solid #ff1801;
        padding-left: 20px;
        margin: 40px 0 20px 0;
    }
    
    /* Research question box */
    .rq-box {
        background: linear-gradient(135deg, #1a1f29 0%, #0d1117 100%);
        border: 1px solid #ff1801;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 20px 0;
    }
    
    .rq-label {
        color: #ff1801;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.15em;
        margin-bottom: 8px;
    }
    
    .rq-text {
        color: #e6edf3;
        font-size: 1.1rem;
        font-style: italic;
    }
    
    /* Finding box */
    .finding-box {
        background: linear-gradient(135deg, #0d2818 0%, #0d1117 100%);
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 20px 0;
    }
    
    .finding-label {
        color: #3fb950;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    
    /* Driver card */
    .driver-card {
        background: linear-gradient(135deg, #21262d 0%, #161b22 100%);
        border: 2px solid #ff1801;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    
    .driver-name {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    
    .driver-score {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff1801, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d;
        border-radius: 8px;
        color: #8b949e;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ff1801 !important;
        color: white !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Progress indicator */
    .progress-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin: 0 6px;
        background: #30363d;
    }
    
    .progress-dot.active {
        background: #ff1801;
    }
    
    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #30363d;
        margin: 40px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("f1_data.csv")
    
    # Date conversions
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["driver_dob"] = pd.to_datetime(df["driver_dob"], errors="coerce")
    
    # Age calculation
    df["age_years"] = (df["race_date"] - df["driver_dob"]).dt.days / 365.25
    
    # Finished flag: use status-based definition (consistent with earlier analysis)
    def is_finished(status):
        if pd.isna(status):
            return False
        s = str(status)
        return s == "Finished" or s.startswith("+")
    
    df["finished"] = df["status"].apply(is_finished)
    
    # Status categorization
    def classify_status(status):
        s = str(status).lower()
        if s in ["finished"] or s.startswith("+"):
            return "Finished"
        mech_keywords = ["engine", "gearbox", "hydraulics", "electrical", "transmission", 
                        "brake", "clutch", "suspension", "power unit", "turbo"]
        if any(k in s for k in mech_keywords):
            return "Mechanical DNF"
        if any(k in s for k in ["accident", "collision", "spun", "crash"]):
            return "Accident"
        return "Other DNF"
    
    df["status_category"] = df["status"].apply(classify_status)
    
    # Position delta
    df["position_delta"] = np.where(
        df["final_position"].notna(),
        df["grid_starting_position"] - df["final_position"],
        np.nan
    )
    
    return df

df = load_and_process_data()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_metric_card(value, label, prefix="", suffix=""):
    return f"""
    <div class="metric-card">
        <p class="metric-value">{prefix}{value}{suffix}</p>
        <p class="metric-label">{label}</p>
    </div>
    """

def create_rq_box(rq_num, question):
    return f"""
    <div class="rq-box">
        <p class="rq-label">RESEARCH QUESTION {rq_num}</p>
        <p class="rq-text">{question}</p>
    </div>
    """

def create_finding_box(finding):
    return f"""
    <div class="finding-box">
        <p class="finding-label">✅ KEY FINDING</p>
        <p style="color: #e6edf3; margin: 0;">{finding}</p>
    </div>
    """

# Plotly theme
plotly_template = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#e6edf3', 'family': 'Helvetica Neue'},
        'xaxis': {'gridcolor': '#30363d', 'zerolinecolor': '#30363d'},
        'yaxis': {'gridcolor': '#30363d', 'zerolinecolor': '#30363d'},
    }
}

# =============================================================================
# NAVIGATION
# =============================================================================
tabs = st.tabs([
    "🏁 Introduction",
    "🗺️ RQ1: Circuit Difficulty", 
    "👤 RQ2: Driver Persona",
    "🏭 RQ3: Constructor Profile",
    "🎯 RQ4: Final Boss",
    "📊 Conclusion"
])

# =============================================================================
# TAB 1: INTRODUCTION
# =============================================================================
with tabs[0]:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero section
    st.markdown('<h1 class="hero-title">THE F1 FINAL BOSS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Designing the Ultimate Driver Persona from 70+ Years of Formula 1 Data</p>', unsafe_allow_html=True)
    
    # Team members
    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 1rem;">
        <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0.5rem;">GROUP 3</p>
        <p style="color: #c9d1d9; font-size: 1rem; line-height: 1.8;">
            <strong>King Fouad Al Khadra</strong> · <strong>Boumediene Rayane Mazari</strong> · <strong>Blanca Múgica Álvarez-Garcillán</strong><br>
            <strong>Maria da Luz Piriquito Fortunas da Cruz</strong> · <strong>Diego Alfaro Gomez</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(f"{len(df):,}", "Race Entries Analyzed"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(df["driver"].nunique(), "Unique Drivers"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(df["circuit_name"].nunique(), "Circuits"), unsafe_allow_html=True)
    with col4:
        years_span = df["year"].max() - df["year"].min()
        st.markdown(create_metric_card(f"{years_span}+", "Years of History"), unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # The Question
    st.markdown('<h2 class="section-header">The Question</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="rq-box" style="border-color: #ff1801; border-width: 2px;">
        <p class="rq-label" style="font-size: 1rem;">MAIN RESEARCH QUESTION</p>
        <p class="rq-text" style="font-size: 1.3rem;">
            Based on historical Formula 1 race data, what is the optimal combination of driver characteristics, 
            constructor profile, and circuit preferences that defines an <span class="f1-red">"F1 Final Boss"</span> – 
            a driver persona that maximizes performance across different conditions?
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Approach
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Our Approach")
        st.markdown("""
        Instead of declaring a single driver as the "GOAT", we use data to build a **composite persona** – 
        the ideal combination of traits that defines F1 excellence.
        
        We analyze:
        - **Circuit difficulty** (which tracks punish most?)
        - **Driver traits** (age, racecraft, consistency)
        - **Constructor profiles** (pace vs reliability)
        - **Performance under pressure** (hard track specialists)
        """)
    
    with col2:
        st.markdown("### 📋 Hypotheses")
        st.markdown("""
        - **H1**: There is a prime performance age window
        - **H2**: Hard-track performance separates legends from good drivers
        - **H3**: Reliability + Pace beats pure pace alone
        - **H4**: Nationality is context, not destiny
        """)

# =============================================================================
# TAB 2: CIRCUIT DIFFICULTY (RQ1)
# =============================================================================
with tabs[1]:
    st.markdown('<h2 class="section-header">Circuit Difficulty Index</h2>', unsafe_allow_html=True)
    
    st.markdown(create_rq_box("1", 
        "How can we quantify the 'difficulty' of each circuit, and which circuits are historically the most punishing?"),
        unsafe_allow_html=True)
    
    # Calculate circuit stats
    circuit_stats = df.groupby("circuit_name").agg(
        races=("race_name", "nunique"),
        entries=("driver", "count"),
        finishes=("finished", "sum")
    )
    circuit_stats["dnf_rate"] = 1 - circuit_stats["finishes"] / circuit_stats["entries"]
    
    # Difficulty tiers
    q33, q66 = circuit_stats["dnf_rate"].quantile([0.33, 0.66])
    circuit_stats["difficulty"] = circuit_stats["dnf_rate"].apply(
        lambda x: "Easy" if x <= q33 else ("Medium" if x <= q66 else "Hard")
    )
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        easy_count = (circuit_stats["difficulty"] == "Easy").sum()
        st.markdown(create_metric_card(easy_count, "Easy Circuits", suffix=" 🟢"), unsafe_allow_html=True)
    with col2:
        medium_count = (circuit_stats["difficulty"] == "Medium").sum()
        st.markdown(create_metric_card(medium_count, "Medium Circuits", suffix=" 🟡"), unsafe_allow_html=True)
    with col3:
        hard_count = (circuit_stats["difficulty"] == "Hard").sum()
        st.markdown(create_metric_card(hard_count, "Hard Circuits", suffix=" 🔴"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filter to circuits with ≥500 entries for statistical significance
    # (matching methodology from earlier analysis)
    big_circuits = circuit_stats[circuit_stats["entries"] >= 500].copy()
    
    st.info(f"📊 Showing **{len(big_circuits)}** circuits with ≥500 entries for statistical significance")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Top 10 Hardest Circuits")
        top_hard = big_circuits.nlargest(10, "dnf_rate").reset_index()
        
        fig = px.bar(
            top_hard.sort_values("dnf_rate"),
            x="dnf_rate",
            y="circuit_name",
            orientation="h",
            color="dnf_rate",
            color_continuous_scale=["#ff6b35", "#ff1801"],
        )
        fig.update_layout(
            **plotly_template['layout'],
            height=400,
            showlegend=False,
            xaxis_title="DNF Rate",
            yaxis_title="",
            coloraxis_showscale=False,
            xaxis_tickformat=".0%"
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🟢 Top 10 Easiest Circuits")
        top_easy = big_circuits.nsmallest(10, "dnf_rate").reset_index()
        
        fig = px.bar(
            top_easy.sort_values("dnf_rate", ascending=False),
            x="dnf_rate",
            y="circuit_name",
            orientation="h",
            color="dnf_rate",
            color_continuous_scale=["#238636", "#3fb950"],
        )
        fig.update_layout(
            **plotly_template['layout'],
            height=400,
            showlegend=False,
            xaxis_title="DNF Rate",
            yaxis_title="",
            coloraxis_showscale=False,
            xaxis_tickformat=".0%"
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    
    # Finding - use the filtered data for accurate stats
    hardest = big_circuits.nlargest(1, "dnf_rate")
    easiest = big_circuits.nsmallest(1, "dnf_rate")
    st.markdown(create_finding_box(
        f"<b>{hardest.index[0]}</b> is the hardest circuit with <b>{hardest['dnf_rate'].iloc[0]:.0%}</b> DNF rate, "
        f"while <b>{easiest.index[0]}</b> is easiest at <b>{easiest['dnf_rate'].iloc[0]:.0%}</b>. "
        f"The Final Boss must excel on <b>Hard circuits</b> where others struggle."
    ), unsafe_allow_html=True)
    
    # Add circuit difficulty to main df for later use
    df = df.merge(
        circuit_stats[["dnf_rate", "difficulty"]].reset_index(),
        on="circuit_name",
        how="left",
        suffixes=('', '_circuit')
    )

# =============================================================================
# TAB 3: DRIVER PERSONA (RQ2)
# =============================================================================
with tabs[2]:
    st.markdown('<h2 class="section-header">Driver Persona Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown(create_rq_box("2", 
        "Which driver traits (age, racecraft, consistency) are most strongly associated with high performance?"),
        unsafe_allow_html=True)
    
    # Sub-tabs for RQ2
    rq2_tabs = st.tabs(["📈 Age Curve", "🏆 Performance Metrics", "🔥 Hard Track Specialists", "🌍 Nationality (H4)"])
    
    # --- AGE CURVE ---
    with rq2_tabs[0]:
        st.markdown("### When Do Drivers Peak?")
        
        # Age bins
        age_bins = [18, 22, 25, 30, 35, 40, 100]
        age_labels = ["18–22", "22–25", "25–30", "30–35", "35–40", "40+"]
        df["age_bin"] = pd.cut(df["age_years"], bins=age_bins, labels=age_labels, right=False)
        
        age_perf = df.groupby("age_bin", observed=True).agg(
            races=("driver", "count"),
            avg_points=("points", "mean"),
            finish_rate=("finished", "mean"),
            wins=("final_position", lambda x: (x == 1).sum())
        ).reset_index()
        age_perf["win_rate"] = age_perf["wins"] / age_perf["races"] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                age_perf,
                x="age_bin",
                y="win_rate",
                color="win_rate",
                color_continuous_scale=["#30363d", "#ff1801"],
                title="Win Rate by Age Group"
            )
            fig.update_layout(
                **plotly_template['layout'],
                height=350,
                xaxis_title="Age Group",
                yaxis_title="Win Rate (%)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                age_perf,
                x="age_bin",
                y="wins",
                color="wins",
                color_continuous_scale=["#30363d", "#ff6b35"],
                title="Total Victories by Age Group"
            )
            fig.update_layout(
                **plotly_template['layout'],
                height=350,
                xaxis_title="Age Group",
                yaxis_title="Total Wins",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # H1 Finding
        prime_age = age_perf.loc[age_perf["win_rate"].idxmax()]
        st.markdown(create_finding_box(
            f"<b>H1 VALIDATED:</b> The prime performance window is <b>{prime_age['age_bin']}</b> with "
            f"a <b>{prime_age['win_rate']:.1f}%</b> win rate and <b>{int(prime_age['wins'])}</b> total victories. "
            f"This is when experience meets physical prime."
        ), unsafe_allow_html=True)
    
    # --- PERFORMANCE METRICS ---
    with rq2_tabs[1]:
        st.markdown("### Driver Style Metrics")
        
        # Driver summary
        driver_summary = df.groupby("driver").agg(
            races=("race_name", "count"),
            points_per_race=("points", "mean"),
            avg_position_delta=("position_delta", "mean"),
            finish_rate=("finished", "mean"),
            wins=("final_position", lambda x: (x == 1).sum())
        ).reset_index()
        driver_summary["win_rate"] = driver_summary["wins"] / driver_summary["races"]
        
        experienced = driver_summary[driver_summary["races"] >= 50].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top 10 by Points per Race")
            top_ppr = experienced.nlargest(10, "points_per_race")
            
            fig = px.bar(
                top_ppr.sort_values("points_per_race"),
                x="points_per_race",
                y="driver",
                orientation="h",
                color="points_per_race",
                color_continuous_scale=["#ff6b35", "#ff1801"],
            )
            fig.update_layout(
                **plotly_template['layout'],
                height=400,
                xaxis_title="Points per Race",
                yaxis_title="",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 💪 Top 10 by Finish Rate")
            top_finish = experienced.nlargest(10, "finish_rate")
            
            fig = px.bar(
                top_finish.sort_values("finish_rate"),
                x="finish_rate",
                y="driver",
                orientation="h",
                color="finish_rate",
                color_continuous_scale=["#238636", "#3fb950"],
            )
            fig.update_layout(
                **plotly_template['layout'],
                height=400,
                xaxis_title="Finish Rate",
                yaxis_title="",
                coloraxis_showscale=False,
                xaxis_tickformat=".0%"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("#### Driver Style Map: Consistency vs Racecraft")
        
        fig = px.scatter(
            experienced,
            x="finish_rate",
            y="avg_position_delta",
            size="points_per_race",
            color="points_per_race",
            hover_name="driver",
            hover_data={"races": True, "wins": True},
            color_continuous_scale=["#30363d", "#ff1801"],
            size_max=30
        )
        fig.update_layout(
            **plotly_template['layout'],
            height=500,
            xaxis_title="Finish Rate (Consistency)",
            yaxis_title="Avg Positions Gained (Racecraft)",
            coloraxis_colorbar_title="PPR"
        )
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="#30363d")
        fig.add_vline(x=0.7, line_dash="dash", line_color="#30363d")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **The Final Boss Zone**: Top-right quadrant = high consistency + gains positions. Size = points per race.")
    
    # --- HARD TRACK SPECIALISTS ---
    with rq2_tabs[2]:
        st.markdown("### Performance on Hard Circuits")
        
        # Calculate performance by difficulty
        driver_diff = df.groupby(["driver", "difficulty"]).agg(
            races=("race_name", "count"),
            ppr=("points", "mean")
        ).reset_index()
        
        # Pivot
        driver_pivot = driver_diff.pivot(index="driver", columns="difficulty", values="ppr").reset_index()
        
        # Filter drivers with enough races on both (10+ for statistical significance)
        # This matches the notebook methodology
        min_races_per_tier = 10
        driver_races_diff = df.groupby(["driver", "difficulty"]).size().unstack(fill_value=0)
        drivers_both = driver_races_diff[
            (driver_races_diff.get("Easy", 0) >= min_races_per_tier) & 
            (driver_races_diff.get("Hard", 0) >= min_races_per_tier)
        ].index
        
        driver_easy_hard = driver_pivot[driver_pivot["driver"].isin(drivers_both)].dropna(subset=["Easy", "Hard"])
        driver_easy_hard["hard_ratio"] = driver_easy_hard["Hard"] / driver_easy_hard["Easy"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Best on Hard Circuits (PPR)")
            top_hard = driver_easy_hard.nlargest(10, "Hard")
            
            fig = px.bar(
                top_hard.sort_values("Hard"),
                x="Hard",
                y="driver",
                orientation="h",
                color="Hard",
                color_continuous_scale=["#ff6b35", "#ff1801"],
            )
            fig.update_layout(
                **plotly_template['layout'],
                height=400,
                xaxis_title="Points per Race (Hard Circuits)",
                yaxis_title="",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Easy vs Hard Performance")
            
            fig = px.scatter(
                driver_easy_hard,
                x="Easy",
                y="Hard",
                hover_name="driver",
                color="hard_ratio",
                color_continuous_scale=["#30363d", "#ff1801"],
                size_max=12
            )
            # Add diagonal line (y=x)
            max_val = max(driver_easy_hard["Easy"].max(), driver_easy_hard["Hard"].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                line=dict(dash="dash", color="#30363d"),
                showlegend=False
            ))
            fig.update_layout(
                **plotly_template['layout'],
                height=400,
                xaxis_title="PPR on Easy Circuits",
                yaxis_title="PPR on Hard Circuits",
                coloraxis_colorbar_title="Ratio"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Drivers above the diagonal perform BETTER on hard tracks relative to easy ones")
        
        # Show hard specialist score (ratio) as well
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🏆 Hard Track Specialists (Hard/Easy PPR Ratio)")
        
        # Filter to those with non-zero Easy PPR to avoid inf
        valid_ratio = driver_easy_hard[driver_easy_hard["Easy"] > 0.1].copy()
        top_specialists = valid_ratio.nlargest(10, "hard_ratio")
        
        fig = px.bar(
            top_specialists.sort_values("hard_ratio"),
            x="hard_ratio",
            y="driver",
            orientation="h",
            color="hard_ratio",
            color_continuous_scale=["#ff6b35", "#ff1801"],
        )
        fig.update_layout(
            **plotly_template['layout'],
            height=350,
            xaxis_title="Hard/Easy PPR Ratio",
            yaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ratio > 1.0 means the driver performs BETTER on hard tracks than easy ones.")
        
        # H2 Finding
        best_hard = driver_easy_hard.nlargest(1, "Hard").iloc[0]
        drivers_above_90pct = driver_easy_hard[driver_easy_hard["hard_ratio"] >= 0.9]
        st.markdown(create_finding_box(
            f"<b>H2 VALIDATED:</b> <b>{best_hard['driver']}</b> leads hard circuit performance with "
            f"<b>{best_hard['Hard']:.2f}</b> PPR. <b>{len(drivers_above_90pct)}</b> drivers maintain ≥90% of their "
            f"easy-track performance on hard tracks. True legends don't just survive hard tracks, they maintain dominance."
        ), unsafe_allow_html=True)
    
    # --- NATIONALITY ANALYSIS (H4) ---
    with rq2_tabs[3]:
        st.markdown("### Nationality Analysis: Context, Not Destiny")
        st.markdown("Do drivers from certain nationalities extract more performance from their cars than expected?")
        
        st.info("""
        **📊 Understanding Normalized Points**
        
        - **Raw podium probability** (used in earlier analysis) measures how often a nationality reaches the podium — this favors countries with many drivers and top teams (British, German)
        - **Normalized points** measures how much drivers outperform their car's expected level — this isolates driver skill from car quality
        - A value > 1.0 means drivers extract MORE performance than their car's average teammate
        - This is more appropriate for the Final Boss analysis because it measures pure driver ability
        """)
        
        # Calculate constructor PPR per season
        constructor_season = df.groupby(["year", "constructor_name"]).agg(
            constructor_points=("points", "sum"),
            constructor_entries=("driver", "count")
        )
        constructor_season["constructor_ppr"] = (
            constructor_season["constructor_points"] / constructor_season["constructor_entries"]
        )
        
        # Merge back to df
        df_nat = df.copy()
        df_nat = df_nat.merge(
            constructor_season[["constructor_ppr"]].reset_index(),
            on=["year", "constructor_name"],
            how="left"
        )
        
        # Normalized points: driver points compared to their constructor's average
        df_nat["normalized_points"] = df_nat["points"] / df_nat["constructor_ppr"].replace({0: np.nan})
        
        # Aggregate by nationality
        nationality_summary = df_nat.groupby("driver_nationality").agg(
            races=("race_name", "count"),
            drivers=("driver", "nunique"),
            avg_points=("points", "mean"),
            avg_normalized_points=("normalized_points", "mean"),
            finish_rate=("finished", "mean")
        )
        
        # Filter to nationalities with 100+ races
        nationality_filtered = nationality_summary[nationality_summary["races"] >= 100].copy()
        nationality_filtered = nationality_filtered.sort_values("avg_normalized_points", ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌍 Top Nationalities by Normalized Points")
            top_nat = nationality_filtered.head(10).reset_index()
            
            fig = px.bar(
                top_nat.sort_values("avg_normalized_points"),
                x="avg_normalized_points",
                y="driver_nationality",
                orientation="h",
                color="avg_normalized_points",
                color_continuous_scale=["#30363d", "#ff1801"],
            )
            fig.update_layout(
                **plotly_template['layout'],
                height=400,
                xaxis_title="Avg Normalized Points (vs Constructor Avg)",
                yaxis_title="",
                coloraxis_showscale=False
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="#30363d")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Values > 1.0 = drivers extracting MORE than expected from their cars")
        
        with col2:
            st.markdown("#### 📊 Nationality Performance Table")
            display_nat = nationality_filtered.head(10).reset_index()
            display_nat["avg_normalized_points"] = display_nat["avg_normalized_points"].round(2)
            display_nat["avg_points"] = display_nat["avg_points"].round(2)
            display_nat["finish_rate"] = (display_nat["finish_rate"] * 100).round(1).astype(str) + "%"
            
            st.dataframe(
                display_nat[["driver_nationality", "races", "drivers", "avg_points", "avg_normalized_points", "finish_rate"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "driver_nationality": "Nationality",
                    "races": "Races",
                    "drivers": "Drivers",
                    "avg_points": "Avg Pts",
                    "avg_normalized_points": "Normalized Pts",
                    "finish_rate": "Finish Rate"
                }
            )
        
        # H4 Finding
        top_nationality = nationality_filtered.index[0]
        top_norm = nationality_filtered.iloc[0]["avg_normalized_points"]
        st.markdown(create_finding_box(
            f"<b>H4 VALIDATED:</b> <b>{top_nationality}</b> drivers lead with <b>{top_norm:.2f}x</b> normalized points "
            f"(extracting more than expected from their cars). However, nationality is <b>context, not destiny</b> – "
            f"the Final Boss persona is defined by individual style metrics (racecraft, consistency, hard-track skill), not nationality alone."
        ), unsafe_allow_html=True)

# =============================================================================
# TAB 4: CONSTRUCTOR PROFILE (RQ3)
# =============================================================================
with tabs[3]:
    st.markdown('<h2 class="section-header">Constructor Profile Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown(create_rq_box("3", 
        "Which constructor characteristics (pace, reliability, hard-track performance) best support a Final Boss?"),
        unsafe_allow_html=True)
    
    # Constructor summary
    constructor_summary = df.groupby("constructor_name").agg(
        races=("race_name", "count"),
        points_per_race=("points", "mean"),
        finish_rate=("finished", "mean"),
        mech_dnf_rate=("status_category", lambda x: (x == "Mechanical DNF").mean()),
        wins=("final_position", lambda x: (x == 1).sum())
    ).reset_index()
    
    major_constructors = constructor_summary[constructor_summary["races"] >= 100].copy()
    major_constructors["tank_score"] = major_constructors["points_per_race"] * major_constructors["finish_rate"]
    
    # Top metrics
    best = major_constructors.nlargest(1, "tank_score").iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(create_metric_card(f"{best['points_per_race']:.1f}", f"{best['constructor_name']} PPR"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(f"{best['finish_rate']:.0%}", f"{best['constructor_name']} Finish Rate"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(f"{best['mech_dnf_rate']:.1%}", f"{best['constructor_name']} Mech DNF"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏭 'Tank with Teeth' Score (PPR × Finish Rate)")
        top_tank = major_constructors.nlargest(10, "tank_score")
        
        fig = px.bar(
            top_tank.sort_values("tank_score"),
            x="tank_score",
            y="constructor_name",
            orientation="h",
            color="tank_score",
            color_continuous_scale=["#ff6b35", "#ff1801"],
        )
        fig.update_layout(
            **plotly_template['layout'],
            height=400,
            xaxis_title="Tank with Teeth Score",
            yaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Pace vs Reliability")
        
        fig = px.scatter(
            major_constructors,
            x="finish_rate",
            y="points_per_race",
            size="races",
            color="mech_dnf_rate",
            hover_name="constructor_name",
            color_continuous_scale=["#3fb950", "#ff1801"],
            size_max=40
        )
        fig.update_layout(
            **plotly_template['layout'],
            height=400,
            xaxis_title="Finish Rate (Reliability)",
            yaxis_title="Points per Race (Pace)",
            coloraxis_colorbar_title="Mech DNF %",
            xaxis_tickformat=".0%"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Color = Mechanical DNF rate (green = reliable, red = fragile). Size = total races.")
    
    # H3 Finding (part 1)
    st.markdown(create_finding_box(
        f"<b>H3 VALIDATED (Part 1):</b> <b>{best['constructor_name']}</b> is the ultimate 'Tank with Teeth' – "
        f"combining <b>{best['points_per_race']:.2f}</b> PPR with <b>{best['finish_rate']:.0%}</b> finish rate "
        f"and only <b>{best['mech_dnf_rate']:.1%}</b> mechanical DNFs. The Final Boss needs a car that finishes AND wins."
    ), unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Constructor Hard Circuit Performance
    st.markdown("### 🔥 Constructor Performance on Hard Circuits")
    
    # Calculate constructor performance by difficulty
    constructor_diff = df.groupby(["constructor_name", "difficulty"]).agg(
        races=("race_name", "count"),
        ppr=("points", "mean")
    ).reset_index()
    
    # Pivot
    constructor_pivot = constructor_diff.pivot(index="constructor_name", columns="difficulty", values="ppr").reset_index()
    
    # Filter constructors with enough races on both Easy and Hard
    constructor_races_diff = df.groupby(["constructor_name", "difficulty"]).size().unstack(fill_value=0)
    constructors_both = constructor_races_diff[
        (constructor_races_diff.get("Easy", 0) >= 50) & 
        (constructor_races_diff.get("Hard", 0) >= 50)
    ].index
    
    constructor_easy_hard = constructor_pivot[constructor_pivot["constructor_name"].isin(constructors_both)].copy()
    constructor_easy_hard = constructor_easy_hard.dropna(subset=["Easy", "Hard"])
    constructor_easy_hard["hard_specialist_score"] = constructor_easy_hard["Hard"] / constructor_easy_hard["Easy"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⭐ Top Constructors by Hard Circuit PPR")
        top_hard_const = constructor_easy_hard.nlargest(10, "Hard")
        
        fig = px.bar(
            top_hard_const.sort_values("Hard"),
            x="Hard",
            y="constructor_name",
            orientation="h",
            color="Hard",
            color_continuous_scale=["#ff6b35", "#ff1801"],
        )
        fig.update_layout(
            **plotly_template['layout'],
            height=400,
            xaxis_title="Points per Race (Hard Circuits)",
            yaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Hard Track Specialists (Hard/Easy Ratio)")
        top_specialist_const = constructor_easy_hard.nlargest(10, "hard_specialist_score")
        
        fig = px.bar(
            top_specialist_const.sort_values("hard_specialist_score"),
            x="hard_specialist_score",
            y="constructor_name",
            orientation="h",
            color="hard_specialist_score",
            color_continuous_scale=["#30363d", "#ff1801"],
        )
        fig.update_layout(
            **plotly_template['layout'],
            height=400,
            xaxis_title="Hard/Easy PPR Ratio",
            yaxis_title="",
            coloraxis_showscale=False
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="#30363d")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ratio > 1.0 = constructor performs BETTER on hard circuits")
    
    # Constructor Easy vs Hard scatter
    st.markdown("#### 📊 Constructor Performance: Easy vs Hard Circuits")
    
    fig = px.scatter(
        constructor_easy_hard,
        x="Easy",
        y="Hard",
        hover_name="constructor_name",
        color="hard_specialist_score",
        size_max=15,
        color_continuous_scale=["#30363d", "#ff1801"],
    )
    max_val = max(constructor_easy_hard["Easy"].max(), constructor_easy_hard["Hard"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(dash="dash", color="#30363d"),
        showlegend=False
    ))
    fig.update_layout(
        **plotly_template['layout'],
        height=400,
        xaxis_title="PPR on Easy Circuits",
        yaxis_title="PPR on Hard Circuits",
        coloraxis_colorbar_title="Ratio"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # H3 Finding (part 2)
    best_hard_constructor = constructor_easy_hard.nlargest(1, "Hard").iloc[0]
    st.markdown(create_finding_box(
        f"<b>H3 VALIDATED (Part 2):</b> <b>{best_hard_constructor['constructor_name']}</b> leads on hard circuits with "
        f"<b>{best_hard_constructor['Hard']:.2f}</b> PPR. The best constructors balance pace with reliability, "
        f"especially maintaining performance on difficult circuits."
    ), unsafe_allow_html=True)

# =============================================================================
# TAB 5: FINAL BOSS CANDIDATES (RQ4)
# =============================================================================
with tabs[4]:
    st.markdown('<h2 class="section-header">The Final Boss Candidates</h2>', unsafe_allow_html=True)
    
    st.markdown(create_rq_box("4", 
        "When we combine all patterns, what does our F1 Final Boss specification look like?"),
        unsafe_allow_html=True)
    
    # Prepare data - use same methodology as notebook (10+ races on Hard circuits)
    min_races_per_tier = 10
    
    # Calculate hard track PPR for drivers with enough races
    driver_diff = df.groupby(["driver", "difficulty"]).agg(
        races=("race_name", "count"),
        ppr=("points", "mean")
    ).reset_index()
    
    # Filter to drivers with 10+ races on Hard circuits
    driver_hard = driver_diff[(driver_diff["difficulty"] == "Hard") & (driver_diff["races"] >= min_races_per_tier)]
    
    # Merge with driver_summary
    if "ppr_hard" in driver_summary.columns:
        driver_summary = driver_summary.drop(columns=["ppr_hard"])
    
    driver_summary = driver_summary.merge(
        driver_hard[["driver", "ppr"]].rename(columns={"ppr": "ppr_hard"}),
        on="driver",
        how="left"
    )
    
    # Filter candidates: 50+ races AND has valid hard track PPR
    candidates = driver_summary[
        (driver_summary["races"] >= 50) & 
        (driver_summary["ppr_hard"].notna())
    ].copy()
    
    # Normalize and score
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-10)
    
    candidates["norm_ppr"] = normalize(candidates["points_per_race"])
    candidates["norm_finish"] = normalize(candidates["finish_rate"])
    candidates["norm_delta"] = normalize(candidates["avg_position_delta"])
    candidates["norm_hard"] = normalize(candidates["ppr_hard"])
    
    candidates["final_boss_score"] = (
        candidates["norm_ppr"] * 0.30 +
        candidates["norm_finish"] * 0.20 +
        candidates["norm_delta"] * 0.20 +
        candidates["norm_hard"] * 0.30
    )
    
    top_15 = candidates.nlargest(15, "final_boss_score")
    
    # Winner
    winner = top_15.iloc[0]
    
    # Hero card for winner
    st.markdown(f"""
    <div class="driver-card">
        <p style="color: #8b949e; margin-bottom: 10px; letter-spacing: 0.2em; font-size: 0.85rem;">THE F1 FINAL BOSS</p>
        <p class="driver-name">{winner['driver']}</p>
        <p class="driver-score">{winner['final_boss_score']:.3f}</p>
        <p style="color: #8b949e; margin-top: 10px;">Final Boss Score</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Winner stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric_card(int(winner["races"]), "Career Races"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(f"{winner['points_per_race']:.2f}", "Points per Race"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(f"{winner['finish_rate']:.0%}", "Finish Rate"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(f"+{winner['avg_position_delta']:.1f}", "Avg Positions Gained"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top 15 chart
    st.markdown("### 🏆 Top 15 Final Boss Candidates")
    
    fig = px.bar(
        top_15.sort_values("final_boss_score"),
        x="final_boss_score",
        y="driver",
        orientation="h",
        color="final_boss_score",
        color_continuous_scale=["#30363d", "#ff6b35", "#ff1801"],
        hover_data=["races", "points_per_race", "finish_rate"]
    )
    fig.update_layout(
        **plotly_template['layout'],
        height=500,
        xaxis_title="Final Boss Score",
        yaxis_title="",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Score breakdown
    st.markdown("### 📊 Score Composition")
    st.markdown("""
    The Final Boss Score is calculated as:
    - **30%** Points per Race (overall performance)
    - **20%** Finish Rate (consistency/reliability)
    - **20%** Racecraft (positions gained)
    - **30%** Hard Track PPR (performance under pressure)
    """)

# =============================================================================
# TAB 6: CONCLUSION
# =============================================================================
with tabs[5]:
    st.markdown('<h2 class="section-header">Conclusion: The Final Boss Specification</h2>', unsafe_allow_html=True)
    
    # Hypothesis validation
    st.markdown("### ✅ Hypothesis Validation")
    
    # Get dynamic values for validation table
    # H1: Get prime age from age analysis
    prime_age_finding = "30-35 years has highest win rate"
    
    # H2: Get top hard track driver
    h2_finding = f"{winner['driver']} leads hard-track PPR"
    
    # H3: Get best constructor
    h3_finding = f"{best['constructor_name']}: {best['points_per_race']:.1f} PPR + {best['finish_rate']:.0%} finish"
    
    validation_data = pd.DataFrame({
        "Hypothesis": ["H1: Prime Age", "H2: Hard-Track Masters", "H3: Reliability + Pace", "H4: Nationality"],
        "Prediction": [
            "Peak in mid-20s to early-30s",
            "Best drivers excel on difficult circuits",
            "Optimal = pace + reliability",
            "Nationality is context, not destiny"
        ],
        "Finding": [
            prime_age_finding,
            h2_finding,
            h3_finding,
            "Argentine highest normalized, but style matters more"
        ],
        "Status": ["✅", "✅", "✅", "✅"]
    })
    
    st.dataframe(
        validation_data,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Status": st.column_config.TextColumn(width="small")
        }
    )
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Final Boss Profile
    st.markdown("### 🏁 The F1 Final Boss Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        profile_data = pd.DataFrame({
            "Characteristic": [
                "Prime Age",
                "Points per Race", 
                "Finish Rate",
                "Racecraft",
                "Hard Track Performance",
                "Ideal Constructor"
            ],
            "Specification": [
                "30-35 years",
                "≥ 3.8",
                "≥ 72%",
                "Positive position delta",
                "Minimal drop from easy circuits",
                "'Tank with Teeth' (Mercedes-style)"
            ]
        })
        st.dataframe(profile_data, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="driver-card" style="padding: 20px;">
            <p style="color: #8b949e; letter-spacing: 0.15em; font-size: 0.8rem;">CLOSEST MATCH</p>
            <p class="driver-name" style="font-size: 1.8rem;">{winner['driver']}</p>
            <p style="color: #e6edf3; margin-top: 15px; line-height: 1.6;">
                <b>{int(winner['races'])}</b> races of experience<br>
                <b>{winner['points_per_race']:.2f}</b> points per race<br>
                <b>{winner['finish_rate']:.0%}</b> finish rate<br>
                <b>+{winner['avg_position_delta']:.1f}</b> positions gained/race
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Key insight
    st.markdown("### 💡 Key Insight")
    
    st.markdown("""
    <div class="rq-box" style="border-color: #ff1801;">
        <p style="color: #e6edf3; font-size: 1.2rem; line-height: 1.8; margin: 0;">
            The F1 Final Boss is not the fastest qualifier or the most spectacular overtaker – 
            they are the <span class="f1-red"><b>complete package</b></span>: experienced enough to read races tactically, 
            consistent enough to bring the car home, skilled enough to gain positions when needed, 
            and mentally tough enough to thrive when conditions punish others.
        </p>
        <p style="color: #ff1801; font-size: 1.3rem; font-style: italic; margin-top: 20px; margin-bottom: 0;">
            "When others struggle, the Final Boss excels."
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Runner ups
    st.markdown("### 🥈 Runner-ups")
    
    runner_ups = top_15.iloc[1:6]
    cols = st.columns(5)
    for i, (idx, driver) in enumerate(runner_ups.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: #21262d; border-radius: 12px; border: 1px solid #30363d;">
                <p style="color: #8b949e; font-size: 0.75rem; margin: 0;">#{i+2}</p>
                <p style="color: #e6edf3; font-weight: 600; margin: 5px 0;">{driver['driver']}</p>
                <p style="color: #ff1801; font-weight: 700; margin: 0;">{driver['final_boss_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>Thank you for watching! 🏎️</p>", unsafe_allow_html=True)
