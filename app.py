import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InfluenceAI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN BACKGROUND
    ===================================================== */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #fff7fb 0%,
                #ffffff 45%,
                #f3f0ff 100%
            );
    }


    /* =====================================================
       MAIN CONTENT
    ===================================================== */

    .main {
        padding-top: 20px;
    }


    /* =====================================================
       HEADINGS
    ===================================================== */

    h1 {
        color: #7c3aed !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #6d28d9 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #7c3aed !important;
        font-weight: 700 !important;
    }


    /* =====================================================
       TEXT
    ===================================================== */

    p {
        color: #374151 !important;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #fff0f6,
                #f5f3ff
            );

        border-right:
            1px solid #eadcf5;
    }


    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {

        color: #6d28d9 !important;
    }


    /* =====================================================
       BUTTON
    ===================================================== */

    .stButton > button {

        width: 100%;

        border-radius: 12px;

        border: none;

        background:
            linear-gradient(
                90deg,
                #7c3aed,
                #ec4899
            );

        color: white;

        font-size: 16px;

        font-weight: 700;

        padding: 13px;

        box-shadow:
            0 5px 15px
            rgba(124, 58, 237, 0.20);
    }


    .stButton > button:hover {

        background:
            linear-gradient(
                90deg,
                #6d28d9,
                #db2777
            );

        color: white;

        transform: translateY(-1px);
    }


    /* =====================================================
       METRIC CARDS
    ===================================================== */

    div[data-testid="stMetric"] {

        background: white;

        padding: 20px;

        border-radius: 16px;

        border:
            1px solid #f3d5e5;

        box-shadow:
            0 5px 20px
            rgba(124, 58, 237, 0.08);
    }


    div[data-testid="stMetricLabel"] {

        color: #6b7280 !important;
    }


    div[data-testid="stMetricValue"] {

        color: #7c3aed !important;

        font-weight: 800 !important;
    }


    /* =====================================================
       SELECT BOX
    ===================================================== */

    div[data-baseweb="select"] > div {

        background-color: white;

        border-radius: 10px;

        border:
            1px solid #e5d4f5;
    }


    /* =====================================================
       DOWNLOAD BUTTON
    ===================================================== */

    .stDownloadButton > button {

        width: 100%;

        border-radius: 12px;

        background:
            linear-gradient(
                90deg,
                #ec4899,
                #7c3aed
            );

        color: white;

        font-weight: 700;

        border: none;
    }


    /* =====================================================
       INFO BOX
    ===================================================== */

    div[data-testid="stAlert"] {

        border-radius: 12px;
    }


    /* =====================================================
       DATAFRAME
    ===================================================== */

    [data-testid="stDataFrame"] {

        border-radius: 12px;

        overflow: hidden;
    }


    /* =====================================================
       DIVIDER
    ===================================================== */

    hr {

        border-color:
            #eadcf5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    engagement_model = joblib.load(
        "models/engagement_model.pkl"
    )

    reach_model = joblib.load(
        "models/reach_model.pkl"
    )

    sales_model = joblib.load(
        "models/sales_model.pkl"
    )

    metadata = joblib.load(
        "models/metadata.pkl"
    )

    return (
        engagement_model,
        reach_model,
        sales_model,
        metadata
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "models/cleaned_data.csv"
    )


# ============================================================
# LOAD FILES
# ============================================================

try:

    (
        engagement_model,
        reach_model,
        sales_model,
        metadata
    ) = load_models()

    df = load_data()

except Exception as e:

    st.error(
        "⚠️ Unable to load model files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🚀 InfluenceAI")

st.subheader(
    "Influencer Marketing Analysis"
)

st.write(
    "Analyze campaign performance using machine learning "
    "predictions for engagement, reach and expected sales."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎯 Campaign Setup")

    st.write(
        "Choose the campaign features to generate "
        "marketing predictions."
    )

    st.divider()

    selected_category = st.selectbox(
        "👤 Influencer Category",
        metadata["influencer_categories"]
    )

    selected_platform = st.selectbox(
        "📱 Social Media Platform",
        metadata["platforms"]
    )

    selected_campaign = st.selectbox(
        "🎯 Campaign Type",
        metadata["campaign_types"]
    )

    campaign_duration = st.slider(
        "📅 Campaign Duration",
        min_value=1,
        max_value=30,
        value=14
    )

    st.divider()

    st.info(
        "💡 The system predicts campaign performance "
        "and compares available influencer-platform "
        "combinations."
    )


# ============================================================
# CAMPAIGN INPUTS
# ============================================================

st.header("🎯 Campaign Inputs")

st.write(
    "Select your campaign details below and generate "
    "the prediction."
)


col1, col2 = st.columns(2)


with col1:

    input_category = st.selectbox(
        "👤 Influencer Category",
        metadata["influencer_categories"],
        key="main_category"
    )

    input_platform = st.selectbox(
        "📱 Social Media Platform",
        metadata["platforms"],
        key="main_platform"
    )


with col2:

    input_campaign = st.selectbox(
        "🎯 Campaign Type",
        metadata["campaign_types"],
        key="main_campaign"
    )

    input_duration = st.slider(
        "📅 Campaign Duration",
        min_value=1,
        max_value=30,
        value=14,
        key="main_duration"
    )


st.write("")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_combination(
    category,
    platform,
    campaign_type,
    duration
):

    input_data = pd.DataFrame({

        "platform": [platform],

        "influencer_category": [category],

        "campaign_type": [campaign_type],

        "campaign_duration_days": [duration]

    })


    engagement = engagement_model.predict(
        input_data
    )[0]


    reach = reach_model.predict(
        input_data
    )[0]


    sales = sales_model.predict(
        input_data
    )[0]


    return (
        max(0, engagement),
        max(0, reach),
        max(0, sales)
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🚀 Predict Campaign Performance"
)


# ============================================================
# RESULTS
# ============================================================

if predict_button:

    # ========================================================
    # SELECTED CAMPAIGN PREDICTION
    # ========================================================

    (
        engagement,
        reach,
        sales
    ) = predict_combination(

        input_category,

        input_platform,

        input_campaign,

        input_duration
    )


    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    st.divider()

    st.header("📊 Prediction Results")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "❤️ Predicted Engagement",
            f"{engagement:,.0f}"
        )


    with col2:

        st.metric(
            "👁️ Expected Reach",
            f"{reach:,.0f}"
        )


    with col3:

        st.metric(
            "🛒 Expected Sales",
            f"{sales:,.0f}"
        )


    with col4:

        st.metric(
            "📅 Duration",
            f"{input_duration} Days"
        )


    # ========================================================
    # RECOMMENDATION ENGINE
    # ========================================================

    recommendations = []


    for category in metadata[
        "influencer_categories"
    ]:

        for platform in metadata[
            "platforms"
        ]:

            (
                engagement_pred,
                reach_pred,
                sales_pred
            ) = predict_combination(

                category,

                platform,

                input_campaign,

                input_duration
            )


            recommendations.append({

                "Category":
                    category,

                "Platform":
                    platform,

                "Predicted Engagement":
                    engagement_pred,

                "Expected Reach":
                    reach_pred,

                "Expected Product Sales":
                    sales_pred

            })


    recommendation_df = pd.DataFrame(
        recommendations
    )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize(series):

        minimum = series.min()

        maximum = series.max()


        if maximum == minimum:

            return pd.Series(
                [100] * len(series),
                index=series.index
            )


        return (
            (series - minimum)
            /
            (maximum - minimum)
        ) * 100


    recommendation_df[
        "Engagement Score"
    ] = normalize(
        recommendation_df[
            "Predicted Engagement"
        ]
    )


    recommendation_df[
        "Reach Score"
    ] = normalize(
        recommendation_df[
            "Expected Reach"
        ]
    )


    recommendation_df[
        "Sales Score"
    ] = normalize(
        recommendation_df[
            "Expected Product Sales"
        ]
    )


    # ========================================================
    # SUITABILITY SCORE
    # ========================================================

    recommendation_df[
        "Suitability Score"
    ] = (

        recommendation_df[
            "Engagement Score"
        ] * 0.30

        +

        recommendation_df[
            "Reach Score"
        ] * 0.30

        +

        recommendation_df[
            "Sales Score"
        ] * 0.40

    )


    recommendation_df = (

        recommendation_df

        .sort_values(
            "Suitability Score",
            ascending=False
        )

        .reset_index(drop=True)

    )


    best = recommendation_df.iloc[0]


    # ========================================================
    # RECOMMENDED CAMPAIGN
    # ========================================================

    st.divider()

    st.header("🎯 Recommended Campaign")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "👤 Category",
            best["Category"]
        )


    with col2:

        st.metric(
            "📱 Platform",
            best["Platform"]
        )


    with col3:

        st.metric(
            "⭐ Suitability",
            f"{best['Suitability Score']:.1f}%"
        )


    st.info(
        f"The calculated suitability score for "
        f"**{best['Category']} × {best['Platform']}** "
        f"is **{best['Suitability Score']:.1f}%**."
    )


    # ========================================================
    # SCORE PROGRESS
    # ========================================================

    st.subheader("⭐ Campaign Suitability")

    st.progress(
        int(best["Suitability Score"])
    )


    # ========================================================
    # CAMPAIGN COMPARISON
    # ========================================================

    st.divider()

    st.header("📋 Campaign Comparison")


    display_df = recommendation_df[
        [
            "Category",
            "Platform",
            "Predicted Engagement",
            "Expected Reach",
            "Expected Product Sales",
            "Suitability Score"
        ]
    ].copy()


    display_df[
        "Predicted Engagement"
    ] = display_df[
        "Predicted Engagement"
    ].round(0)


    display_df[
        "Expected Reach"
    ] = display_df[
        "Expected Reach"
    ].round(0)


    display_df[
        "Expected Product Sales"
    ] = display_df[
        "Expected Product Sales"
    ].round(0)


    display_df[
        "Suitability Score"
    ] = display_df[
        "Suitability Score"
    ].round(2)


    st.dataframe(
        display_df.head(10),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # VISUAL ANALYTICS
    # ========================================================

    st.divider()

    st.header("📈 Marketing Visual Analytics")


    chart_df = recommendation_df.head(10).copy()


    chart_df["Combination"] = (
        chart_df["Category"]
        + " • "
        + chart_df["Platform"]
    )


    # ========================================================
    # 1. ENGAGEMENT BAR CHART
    # ========================================================

    st.subheader("❤️ Predicted Engagement")


    fig_engagement = px.bar(

        chart_df,

        x="Combination",

        y="Predicted Engagement",

        title="Predicted Engagement by Campaign",

        text_auto=".0f",

        color="Predicted Engagement",

        color_continuous_scale="Purples"

    )


    fig_engagement.update_layout(

        height=450,

        xaxis_title="Campaign Combination",

        yaxis_title="Engagement",

        paper_bgcolor="white",

        plot_bgcolor="white"

    )


    st.plotly_chart(
        fig_engagement,
        use_container_width=True
    )


    # ========================================================
    # 2. REACH BAR CHART
    # ========================================================

    st.subheader("👁️ Expected Reach")


    fig_reach = px.bar(

        chart_df,

        x="Combination",

        y="Expected Reach",

        title="Expected Reach by Campaign",

        text_auto=".0f",

        color="Expected Reach",

        color_continuous_scale="Blues"

    )


    fig_reach.update_layout(

        height=450,

        xaxis_title="Campaign Combination",

        yaxis_title="Reach",

        paper_bgcolor="white",

        plot_bgcolor="white"

    )


    st.plotly_chart(
        fig_reach,
        use_container_width=True
    )


    # ========================================================
    # 3. SALES BAR CHART
    # ========================================================

    st.subheader("🛒 Expected Product Sales")


    fig_sales = px.bar(

        chart_df,

        x="Combination",

        y="Expected Product Sales",

        title="Expected Product Sales by Campaign",

        text_auto=".0f",

        color="Expected Product Sales",

        color_continuous_scale="Pinkyl"

    )


    fig_sales.update_layout(

        height=450,

        xaxis_title="Campaign Combination",

        yaxis_title="Expected Sales",

        paper_bgcolor="white",

        plot_bgcolor="white"

    )


    st.plotly_chart(
        fig_sales,
        use_container_width=True
    )


    # ========================================================
    # 4. SUITABILITY BAR CHART
    # ========================================================

    st.subheader("⭐ Campaign Suitability")


    fig_score = px.bar(

        chart_df,

        x="Combination",

        y="Suitability Score",

        title="Suitability Score by Campaign",

        text_auto=".1f",

        color="Suitability Score",

        color_continuous_scale="Sunset"

    )


    fig_score.update_layout(

        height=450,

        xaxis_title="Campaign Combination",

        yaxis_title="Suitability (%)",

        paper_bgcolor="white",

        plot_bgcolor="white"

    )


    st.plotly_chart(
        fig_score,
        use_container_width=True
    )


    # ========================================================
    # 5. PIE CHART
    # ========================================================

    st.divider()

    st.subheader("🥧 Platform Distribution")


    platform_counts = (
        recommendation_df["Platform"]
        .value_counts()
        .reset_index()
    )


    platform_counts.columns = [
        "Platform",
        "Count"
    ]


    fig_pie = px.pie(

        platform_counts,

        names="Platform",

        values="Count",

        title="Influencer Platform Distribution",

        hole=0.35

    )


    fig_pie.update_layout(

        height=450,

        paper_bgcolor="white"

    )


    st.plotly_chart(

        fig_pie,

        use_container_width=True

    )


    # ========================================================
    # 6. HISTOGRAM
    # ========================================================

    st.subheader("📊 Engagement Distribution")


    fig_hist = px.histogram(

        recommendation_df,

        x="Predicted Engagement",

        nbins=10,

        title="Distribution of Predicted Engagement",

        color_discrete_sequence=[
            "#8b5cf6"
        ]

    )


    fig_hist.update_layout(

        height=450,

        xaxis_title="Predicted Engagement",

        yaxis_title="Number of Campaign Combinations",

        paper_bgcolor="white",

        plot_bgcolor="white"

    )


    st.plotly_chart(

        fig_hist,

        use_container_width=True

    )


    # ========================================================
    # 7. LINE CHART
    # ========================================================

    st.subheader("📈 Engagement vs Reach")


    line_df = recommendation_df.copy()


    line_df["Combination"] = (

        line_df["Category"]

        + " • "

        + line_df["Platform"]

    )


    fig_line = px.line(

        line_df,

        x="Combination",

        y=[
            "Predicted Engagement",
            "Expected Reach"
        ],

        markers=True,

        title="Engagement and Reach Across Campaigns"

    )


    fig_line.update_layout(

        height=500,

        xaxis_title="Campaign Combination",

        yaxis_title="Predicted Value",

        paper_bgcolor="white",

        plot_bgcolor="white"

    )


    st.plotly_chart(

        fig_line,

        use_container_width=True

    )


    # ========================================================
    # 8. SCATTER PLOT
    # ========================================================

    st.subheader("🔵 Reach vs Expected Sales")


    fig_scatter = px.scatter(

        recommendation_df,

        x="Expected Reach",

        y="Expected Product Sales",

        color="Platform",

        size="Predicted Engagement",

        hover_data=[

            "Category",

            "Platform",

            "Predicted Engagement",

            "Expected Reach",

            "Expected Product Sales"

        ],

        title="Relationship Between Reach and Product Sales"

    )


    fig_scatter.update_layout(

        height=500,

        xaxis_title="Expected Reach",

        yaxis_title="Expected Product Sales",

        paper_bgcolor="white",

        plot_bgcolor="white"

    )


    st.plotly_chart(

        fig_scatter,

        use_container_width=True

    )


    # ========================================================
    # 9. BOX PLOT
    # ========================================================

    st.subheader("📦 Sales Distribution")


    fig_box = px.box(

        recommendation_df,

        x="Platform",

        y="Expected Product Sales",

        color="Platform",

        points="all",

        title="Expected Product Sales Distribution"

    )


    fig_box.update_layout(

        height=500,

        xaxis_title="Platform",

        yaxis_title="Expected Product Sales",

        paper_bgcolor="white",

        plot_bgcolor="white",

        showlegend=False

    )


    st.plotly_chart(

        fig_box,

        use_container_width=True

    )


    # ========================================================
    # 10. RADAR CHART
    # ========================================================

    st.divider()

    st.header("🎯 Recommended Campaign Profile")


    radar_categories = [

        "Engagement",

        "Reach",

        "Sales"

    ]


    radar_values = [

        best["Engagement Score"],

        best["Reach Score"],

        best["Sales Score"]

    ]


    radar_values.append(

        radar_values[0]

    )


    radar_categories.append(

        radar_categories[0]

    )


    fig_radar = go.Figure()


    fig_radar.add_trace(

        go.Scatterpolar(

            r=radar_values,

            theta=radar_categories,

            fill="toself",

            name="Campaign Profile"

        )

    )


    fig_radar.update_layout(

        polar=dict(

            radialaxis=dict(

                visible=True,

                range=[0, 100]

            )

        ),

        height=500,

        paper_bgcolor="white"

    )


    st.plotly_chart(

        fig_radar,

        use_container_width=True

    )


    # ========================================================
    # DATASET
    # ========================================================

    st.divider()

    with st.expander(
        "🔍 Explore Dataset"
    ):

        st.write(

            f"Dataset contains "
            f"**{df.shape[0]:,} rows** "
            f"and **{df.shape[1]} columns**."

        )

        st.dataframe(

            df,

            use_container_width=True,

            hide_index=True

        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.divider()

    st.header("📥 Download Results")


    csv = recommendation_df.to_csv(

        index=False

    ).encode("utf-8")


    st.download_button(

        "⬇️ Download Campaign Recommendations",

        csv,

        "influencer_campaign_results.csv",

        "text/csv"

    )


# ============================================================
# BEFORE PREDICTION
# ============================================================

else:

    st.info(
        "👆 Select your campaign features above and click "
        "**Predict Campaign Performance** to generate the "
        "results and visual analytics."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚀 InfluenceAI | Social Network Analysis + "
    "Machine Learning + Influencer Marketing"
)