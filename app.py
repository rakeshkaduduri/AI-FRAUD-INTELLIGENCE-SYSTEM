# AI FRAUD INTELLIGENCE SYSTEM

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model


# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Fraud Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* =====================================================
BACKGROUND
===================================================== */

.stApp {
    background:
    radial-gradient(
        circle at top left,
        #07111f 0%,
        #020617 45%,
        #000000 100%
    );
    color: white;
}


/* =====================================================
REMOVE STREAMLIT
===================================================== */

#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
header {
    visibility: hidden;
}



/* =====================================================
MAIN TITLE
REALISTIC MODERN GLOW EFFECT
===================================================== */

.main-title {
    font-size: 3.9rem;
    font-weight: 900;
    text-align: center;
    margin-top: 18px;
    margin-bottom : 18px;
    letter-spacing: 1.2px;
    position: relative;
    color: #E2E8F0;
    animation:
    titleGlow 3.5s ease-in-out infinite,
    titleFloat 5s ease-in-out infinite;
    transition: 0.4s ease;
}

            
# AURA PULSE
@keyframes auraPulse {

    0% {
        opacity: 0.55;
        transform:
        translate(-50%, -50%)
        scale(0.96);
    }

    50% {
        opacity: 1;
        transform:
        translate(-50%, -50%)
        scale(1.05);
    }

    100% {
        opacity: 0.55;
        transform:
        translate(-50%, -50%)
        scale(0.96);
    }
}


/* =====================================================
TITLE GLOW ANIMATION
===================================================== */

@keyframes titleGlow {

    0% {
        color: #CBD5E1;
        text-shadow:
        0px 0px 4px rgba(56,189,248,0.10),
        0px 0px 12px rgba(56,189,248,0.05);
        opacity: 0.96;
    }

    25% {
        color: #E2E8F0;
        text-shadow:
        0px 0px 8px rgba(56,189,248,0.18),
        0px 0px 22px rgba(56,189,248,0.08),
        0px 0px 40px rgba(56,189,248,0.04);
        opacity: 1;
    }

    50% {
        color: #93C5FD;
        text-shadow:
        0px 0px 12px rgba(125,211,252,0.28),
        0px 0px 30px rgba(125,211,252,0.12),
        0px 0px 55px rgba(125,211,252,0.06);
        opacity: 1;
    }

    75% {
        color: #E2E8F0;
        text-shadow:
        0px 0px 8px rgba(56,189,248,0.18),
        0px 0px 22px rgba(56,189,248,0.08),
        0px 0px 40px rgba(56,189,248,0.04);
        opacity: 1;
    }

    100% {
        color: #93C5FD;
        text-shadow:
        0px 0px 4px rgba(56,189,248,0.10),
        0px 0px 12px rgba(56,189,248,0.05);
        opacity: 0.96;
    }
}


/* =====================================================
FLOATING EFFECT
===================================================== */

@keyframes titleFloat {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-2px);
    }

    100% {
        transform: translateY(0px);
    }
}


/* =====================================================
UNDERLINE GLOW
===================================================== */

.main-title::after {
    content: "";
    display: block;
    width: 180px;
    height: 3px;
    margin: 16px auto 0 auto;
    border-radius: 999px;
    background:
    linear-gradient(
        90deg,
        transparent,
        #38BDF8,
        #7DD3FC,
        #38BDF8,
        transparent
    );

    box-shadow:
    0px 0px 10px rgba(56,189,248,0.18),
    0px 0px 20px rgba(56,189,248,0.08);

    animation:
    linePulse 3.5s ease-in-out infinite;
}


/* =====================================================
UNDERLINE PULSE
===================================================== */

@keyframes linePulse {

    0% {
        opacity: 0.4;
        width: 140px;
    }

    50% {
        opacity: 0.95;
        width: 240px;
    }

    100% {
        opacity: 0.4;
        width: 140px;
    }
}


/* =====================================================
OPTIONAL HOVER EFFECT
===================================================== */

.main-title:hover {
    transform: scale(1.01);
    cursor: default;
}
            
.sub-title {
    text-align: center;
    color: #64748B;
    font-size: 1rem;
    font-weight: 400;
    margin-top: 12px;
    margin-bottom: 38px;
    letter-spacing: 0.2px;
}


/* =====================================================
SECTION TITLE
===================================================== */

.section-title {
    color: #00F5FF;
    font-size: 2.22rem;
    font-weight: 700;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 26px;
    letter-spacing: 0.5px;
    position: relative;
}


/* GLOW UNDERLINE */

.section-title::after {
    content: "";
    display: block;
    width: 90px;
    height: 2px;
    margin: 10px auto 0 auto;
    background: linear-gradient(
        90deg,
        transparent,
        #00F5FF,
        transparent
    );

    border-radius: 999px;
    box-shadow:
    0px 0px 12px #00F5FF;
}


/* =====================================================
METRIC CARDS
===================================================== */

.metric-card {
    background:
    rgba(255,255,255,0.04);
    border:
    1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 24px;
    backdrop-filter: blur(18px);
    transition: 0.3s;
    box-shadow:
    0px 0px 18px rgba(0,255,255,0.08);
    height: 100px;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow:
    0px 0px 28px rgba(0,255,255,0.18);
}


/* =====================================================
METRIC TITLE
===================================================== */

.metric-title {
    color: #94A3B8;
    font-size: 0.9rem;
}


/* =====================================================
METRIC VALUE
===================================================== */

.metric-value {
    color: #00F5FF;
    font-size: 2rem;
    font-weight: bold;
    margin-top: 10px;
}

/* =====================================================
LIVE STATUS PANEL
===================================================== */

.status-panel {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 38px;
    background:
    linear-gradient(
        135deg,
        rgba(0,255,180,0.05),
        rgba(0,180,255,0.04)
    );

    border:
    1px solid rgba(0,255,180,0.14);
    padding: 26px 38px;
    border-radius: 24px;
    backdrop-filter: blur(18px);
    transition: all 0.35s ease;
    position: relative;
    overflow: hidden;
    margin-bottom: 38px;
    box-shadow:
    0px 0px 20px rgba(0,255,180,0.04);
}


/* HOVER EFFECT */

.status-panel:hover {
    transform: translateY(-3px);
    border:
    1px solid rgba(0,255,180,0.28);
    box-shadow:
    0px 0px 35px rgba(0,255,180,0.12),
    0px 0px 70px rgba(0,180,255,0.06);
}


/* GLOW OVERLAY */

.status-panel::before {
    content: "";
    position: absolute;
    top: 0;
    left: -120%;
    width: 70%;
    height: 100%;
    background:
    linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.05),
        transparent
    );
    transition: 0.7s;
}


.status-panel:hover::before {
    left: 130%;
}


/* LEFT */

.status-left {
    display: flex;
    align-items: center;
    gap: 18px;
}


/* PULSE DOT */

.pulse-dot {
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: #00FF99;
    box-shadow:
    0px 0px 15px #00FF99;
    animation: pulse 1.8s infinite;
}


/* DOT ANIMATION */

@keyframes pulse {

    0% {
        transform: scale(1);
        opacity: 1;
    }

    50% {
        transform: scale(1.18);
        opacity: 0.7;
    }

    100% {
        transform: scale(1);
        opacity: 1;
    }
}


/* LABEL */

.status-label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 5px;
}


/* MAIN */

.status-main {
    font-size: 1.22rem;
    font-weight: 700;
    color: white;
}


/* METRICS */

.status-metric {
    min-width: 110px;
    text-align: center;
}


.status-metric-label {
    font-size: 0.78rem;
    color: #94A3B8;
    margin-bottom: 6px;
}


.status-metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #00F5FF;
}


/* DIVIDER */

.status-divider {
    width: 1px;
    height: 44px;
    background:
    rgba(255,255,255,0.08);
}


/* =====================================================
ALERT BOX
===================================================== */

.alert-box {

    background:
    linear-gradient(
        135deg,
        rgba(80,0,20,0.65),
        rgba(35,0,10,0.92)
    );

    border:
    1px solid rgba(255,75,75,0.22);

    border-left:
    4px solid #FF4B4B;

    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(14px);

    box-shadow:
    0px 0px 25px rgba(255,75,75,0.08);
}


/* =====================================================
TABLE
===================================================== */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border:
    1px solid rgba(255,255,255,0.08);
}


/* =====================================================
DOWNLOAD BUTTON
===================================================== */

.stDownloadButton button {
    width: 100%;
    border-radius: 12px;
    height: 52px;
    font-weight: bold;
    background: linear-gradient(
        90deg,
        #00F5FF,
        #0099FF
    );

    color: black;
    border: none;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class='main-title'>
 AI FRAUD INTELLIGENCE SYSTEM
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
Real-Time Deep Learning Threat Monitoring Platform
</div>
""", unsafe_allow_html=True)


# =========================================================
# POSITIONAL ENCODING LAYER
# =========================================================

@tf.keras.utils.register_keras_serializable()
class PositionalEncodingLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def positional_encoding(
        self,
        seq_len,
        d_model
    ):

        positions = tf.range(
            seq_len,
            dtype=tf.float32
        )

        positions = tf.expand_dims(
            positions,
            axis=1
        )

        dimensions = tf.range(
            d_model,
            dtype=tf.float32
        )

        dimensions = tf.expand_dims(
            dimensions,
            axis=0
        )

        angle_rates = 1 / tf.pow(

            10000.0,

            (
                2 *
                tf.floor(dimensions / 2)
            ) /

            tf.cast(d_model, tf.float32)
        )

        angle_rads = (
            positions *
            angle_rates
        )

        sines = tf.sin(
            angle_rads[:, 0::2]
        )

        cosines = tf.cos(
            angle_rads[:, 1::2]
        )

        pos_encoding = tf.concat(
            [sines, cosines],
            axis=-1
        )

        pos_encoding = tf.expand_dims(
            pos_encoding,
            axis=0
        )

        return pos_encoding

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        pos_encoding = self.positional_encoding(
            seq_len,
            d_model
        )

        return inputs + pos_encoding


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        "fraud_detection_model.keras",  

        custom_objects={
            "PositionalEncodingLayer":
            PositionalEncodingLayer
        },
        compile=False
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    return model, scaler


model, scaler = load_artifacts()


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)


# =========================================================
# CREATE SEQUENCES
# =========================================================

SEQUENCE_LENGTH = 5

def create_sequences(data, seq_len):
    sequences = []

    for i in range(
        len(data) - seq_len
    ):

        sequences.append(
            data[i:i+seq_len]
        )

    return np.array(sequences)


# =========================================================
# MAIN LOGIC
# =========================================================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # =====================================================
    # VALIDATION
    # =====================================================

    if "Amount" not in df.columns:
        st.error(
            "CSV must contain Amount column"
        )

        st.stop()


    # =====================================================
    # SORT TIME
    # =====================================================

    if "Time" in df.columns:
        df = df.sort_values(
            "Time"
        )


    # =====================================================
    # REMOVE CLASS COLUMN
    # =====================================================

    if "Class" in df.columns:
        df = df.drop(
            "Class",
            axis=1
        )


    # =====================================================
    # SCALE AMOUNT
    # =====================================================

    df["Amount"] = scaler.transform(
        df[["Amount"]]
    )


    # =====================================================
    # CREATE INPUT
    # =====================================================

    data = df.values

    X_input = create_sequences(
        data,
        SEQUENCE_LENGTH
    )

    if len(X_input) == 0:
        st.error(
            "Upload more than 5 rows"
        )
        st.stop()


    # =====================================================
    # PREDICTIONS
    # =====================================================

    probabilities = model.predict(
        X_input,
        verbose=0
    ).flatten()


    # =====================================================
    # RESULT DATAFRAME
    # =====================================================

    result_df = df.iloc[
        SEQUENCE_LENGTH:
    ].copy()

    result_df[
        "Fraud_Probability"
    ] = probabilities


    # =====================================================
    # RISK LEVELS
    # =====================================================

    result_df["Risk_Level"] = np.where(
        result_df["Fraud_Probability"] > 0.8,

        "CRITICAL",
        np.where(
            result_df["Fraud_Probability"] > 0.5,
            "SUSPICIOUS",
            "SAFE"
        )
    )


    # =====================================================
    # KPI VALUES
    # =====================================================

    total_transactions = len(
        result_df
    )

    critical_count = len(
        result_df[
            result_df["Risk_Level"]
            == "CRITICAL"
        ]
    )

    suspicious_count = len(
        result_df[
            result_df["Risk_Level"]
            == "SUSPICIOUS"
        ]
    )

    avg_risk = result_df[
        "Fraud_Probability"
    ].mean()

    max_risk = result_df[
        "Fraud_Probability"
    ].max()


    # =====================================================
    # STATUS BAR
    # =====================================================

    st.markdown(f"""
    <div class='status-panel'>
    <div class='status-left'>
    <div class='pulse-dot'></div>
    <div>

    <div class='status-label'>
    Live Monitoring
    </div>

    <div class='status-main'>
    AI Fraud Detection Active
    </div>
    </div>
    </div>


    <div class='status-divider'></div>
    <div class='status-metric'>

    <div class='status-metric-label'>
    Transactions
    </div>

    <div class='status-metric-value'>
    {total_transactions}
    </div>
    </div>


    <div class='status-divider'></div>
    <div class='status-metric'>
    <div class='status-metric-label'>
    Threat Level
    </div>

    <div class='status-metric-value'
    style='color:#FFD700;'>

    Moderate
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='height:18px'></div>",
        unsafe_allow_html=True
    )


    # =====================================================
    # KPI ROW
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(f"""
        <div class='metric-card'>

        <div class='metric-title'>
        Transactions
        </div>

        <div class='metric-value'>
        {total_transactions}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class='metric-card'>

        <div class='metric-title'>
        Critical Threats
        </div>

        <div class='metric-value'
        style='color:#FF4B4B;'>

        {critical_count}

        </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class='metric-card'>

        <div class='metric-title'>
        Suspicious Activity
        </div>

        <div class='metric-value'
        style='color:#FFD700;'>

        {suspicious_count}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:

        st.markdown(f"""
        <div class='metric-card'>

        <div class='metric-title'>
        Threat Score
        </div>

        <div class='metric-value'>
        {max_risk:.2f}
        </div>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # SYSTEM THREAT LEVEL
    # =====================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'>
    System Threat Level
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns([1, 1.8, 1])

    with g2:
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=max_risk * 100,
                number={
                    'font': {
                        'size': 42,
                        'color': "#FFFFFF"
                    }
                },

                gauge={
                    'axis': {
                        'range': [0,100]
                    },

                    'bar': {
                        'color': "#00F5FF",
                        'thickness': 0.35
                    },

                    'steps': [

                        {
                            'range': [0,40],
                            'color': "#00AA66"
                        },

                        {
                            'range': [40,70],
                            'color': "#FFD700"
                        },

                        {
                            'range': [70,100],
                            'color': "#FF4B4B"
                        }
                    ]
                }
            )
        )

        gauge_fig.update_layout(
            template="plotly_dark",
            height=260,
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(
                l=0,
                r=0,
                t=10,
                b=0
            )
        )

        st.plotly_chart(
            gauge_fig,
            use_container_width=True
        )


    # =====================================================
    # TOP THREAT DETECTION
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'>
    Top Threat Detection
    </div>
    """, unsafe_allow_html=True)

    top_alert = result_df.sort_values(
        "Fraud_Probability",
        ascending=False
    ).head(1)

    for _, row in top_alert.iterrows():
        c1, c2, c3 = st.columns([1, 2.2, 1])

        with c2:

            st.markdown(f"""
            <div class='alert-box'>

            <div style='
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:20px;
            '>

            <div>

            <div style='
            font-size:1rem;
            font-weight:700;
            color:white;
            '>

            🚨 HIGH RISK TRANSACTION

            </div>

            <div style='
            margin-top:8px;
            color:#94A3B8;
            font-size:0.9rem;
            '>

            Classification:
            <b>{row['Risk_Level']}</b>

            </div>

            </div>

            <div style='
            text-align:right;
            min-width:90px;
            '>

            <div style='
            font-size:2rem;
            font-weight:800;
            color:#FF4B4B;
            line-height:1;
            '>

            {row['Fraud_Probability']:.2f}

            </div>

            <div style='
            font-size:0.75rem;
            color:#94A3B8;
            margin-top:6px;
            '>

            Threat Score

            </div>

            </div>

            </div>

            </div>
            """, unsafe_allow_html=True)


    # =====================================================
    # RISK INTELLIGENCE ANALYSIS
    # =====================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'>
    Risk Intelligence Analysis
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns(2)


    # =====================================================
    # RISK CLUSTER ANALYSIS
    # =====================================================

    with r1:
        cluster_fig = px.scatter(
            result_df,
            x=result_df.index,
            y="Fraud_Probability",
            color="Risk_Level",
            color_discrete_map={
                "SAFE":"#00FF99",
                "SUSPICIOUS":"#FFD700",
                "CRITICAL":"#FF4B4B"
            },
            template="plotly_dark"
        )

        cluster_fig.update_traces(
            marker=dict(
                size=8,
                opacity=0.82
            )
        )

        cluster_fig.update_layout(
            height=340,
            yaxis_range=[0.52, 0.56],
            legend_title_text='',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            )
        )

        st.plotly_chart(
            cluster_fig,
            use_container_width=True
        )


    # =====================================================
    # RISK DENSITY
    # =====================================================

    with r2:
        density_fig = px.histogram(
            result_df,
            x="Fraud_Probability",
            nbins=20,
            template="plotly_dark"
        )

        density_fig.update_traces(
            marker_color="#00F5FF",
            marker_line_width=0
        )

        density_fig.update_layout(
            height=340,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )

        st.plotly_chart(
            density_fig,
            use_container_width=True
        )


    # =====================================================
    # SECOND ANALYTICS ROW
    # =====================================================

    st.markdown("<br><br>", unsafe_allow_html=True)
    a1, a2 = st.columns(2)


    # =====================================================
    # THREAT COMPOSITION
    # =====================================================

    with a1:

        st.markdown("""
        <div class='section-title'>
        Threat Composition
        </div>
        """, unsafe_allow_html=True)

        distribution = result_df[
            "Risk_Level"
        ].value_counts()

        composition_fig = go.Figure()
        composition_fig.add_trace(

            go.Bar(
                x=distribution.values,
                y=distribution.index,
                orientation='h',
                marker_color=[
                    "#00FF99",
                    "#FFD700",
                    "#FF4B4B"
                ]
            )
        )

        composition_fig.update_layout(
            template="plotly_dark",
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )

        st.plotly_chart(
            composition_fig,
            use_container_width=True
        )


    # =====================================================
    # THREAT TIMELINE
    # =====================================================

    with a2:

        st.markdown("""
        <div class='section-title'>
        Threat Timeline Intelligence
        </div>
        """, unsafe_allow_html=True)

        timeline_fig = go.Figure()
        timeline_fig.add_trace(

            go.Scatter(
                y=result_df[
                    "Fraud_Probability"
                ],
                mode='lines',
                fill='tozeroy',
                line=dict(
                    color='#00F5FF',
                    width=3
                )
            )
        )

        timeline_fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="#FFD700"
        )

        timeline_fig.update_layout(
            template="plotly_dark",
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            )
        )

        st.plotly_chart(
            timeline_fig,
            use_container_width=True
        )


    # =====================================================
    # RECENT CRITICAL TRANSACTIONS
    # =====================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'>
    Recent Critical Transactions
    </div>
    """, unsafe_allow_html=True)

    top_threats = result_df[
        result_df["Risk_Level"] != "SAFE"
    ].sort_values(

        "Fraud_Probability",
        ascending=False

    ).head(10)

    st.dataframe(
        top_threats,
        use_container_width=True
    )


    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)
    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download AI Threat Report",
        data=csv,
        file_name="fraud_report.csv",
        mime="text/csv"
    )