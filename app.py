import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="CGPA Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).parent / "student_data.csv"

# ---------- Custom UI ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #111827, #1e3a8a);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
    }
    .hero p {
        margin: .45rem 0 0 0;
        color: #dbeafe;
        font-size: 1rem;
    }
    .kpi-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 7px 20px rgba(15, 23, 42, 0.06);
        min-height: 118px;
    }
    .kpi-label {
        color: #64748b;
        font-size: .88rem;
        margin-bottom: .25rem;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .kpi-sub {
        color: #475569;
        font-size: .82rem;
        margin-top: .25rem;
    }
    .section-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 7px 20px rgba(15, 23, 42, 0.05);
    }
    .result-box {
        background: linear-gradient(135deg, #eff6ff, #ecfeff);
        border: 1px solid #bfdbfe;
        padding: 1.1rem;
        border-radius: 18px;
        margin-top: .6rem;
    }
    div[data-testid="stSidebar"] {
        background: #0f172a;
    }
    div[data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        min-height: 44px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE)

    features = [
        "previous_cgpa",
        "attendance",
        "study_hours",
        "assignment_score",
        "midterm_score",
        "final_score",
    ]

    X = df[features]
    y = df["cgpa"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return model, mae, r2, df

model, mae, r2, df = train_model()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎓 Student Input")
    st.caption("Enter academic details below")

    previous_cgpa = st.number_input(
        "Previous CGPA",
        min_value=0.0,
        max_value=4.0,
        value=3.00,
        step=0.01,
    )

    attendance = st.slider("Attendance (%)", 0, 100, 82)
    study_hours = st.number_input(
        "Study Hours / Day",
        min_value=0.0,
        max_value=15.0,
        value=3.5,
        step=0.5,
    )
    assignment_score = st.slider("Assignment Score (%)", 0, 100, 78)
    midterm_score = st.slider("Midterm Score (%)", 0, 100, 72)
    final_score = st.slider("Final Exam Score (%)", 0, 100, 76)

    st.markdown("---")
    predict_btn = st.button("🚀 Predict CGPA", use_container_width=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>CGPA Prediction Dashboard</h1>
    <p>Machine Learning based academic performance prediction system</p>
</div>
""", unsafe_allow_html=True)

# ---------- Dataset KPIs ----------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Training Records</div>
        <div class="kpi-value">{len(df)}</div>
        <div class="kpi-sub">Sample students in dataset</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Model R² Score</div>
        <div class="kpi-value">{r2:.2f}</div>
        <div class="kpi-sub">Prediction quality indicator</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Model MAE</div>
        <div class="kpi-value">{mae:.2f}</div>
        <div class="kpi-sub">Average prediction error</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    avg_cgpa = float(df["cgpa"].mean())
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Dataset Avg CGPA</div>
        <div class="kpi-value">{avg_cgpa:.2f}</div>
        <div class="kpi-sub">Average of sample dataset</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------- Prediction ----------
left, right = st.columns([1.1, 1])

with left:
    st.markdown("### 📌 Current Student Profile")

    profile_df = pd.DataFrame({
        "Metric": [
            "Previous CGPA",
            "Attendance",
            "Study Hours",
            "Assignment",
            "Midterm",
            "Final Exam",
        ],
        "Value": [
            previous_cgpa,
            attendance,
            study_hours,
            assignment_score,
            midterm_score,
            final_score,
        ],
    })

    st.dataframe(profile_df, hide_index=True, use_container_width=True)

with right:
    st.markdown("### 🔮 Prediction Result")

    if predict_btn:
        input_df = pd.DataFrame(
            [[
                previous_cgpa,
                attendance,
                study_hours,
                assignment_score,
                midterm_score,
                final_score,
            ]],
            columns=[
                "previous_cgpa",
                "attendance",
                "study_hours",
                "assignment_score",
                "midterm_score",
                "final_score",
            ],
        )

        pred = float(model.predict(input_df)[0])
        pred = max(0.0, min(4.0, pred))

        if pred >= 3.5:
            category = "Excellent"
            message = "Strong academic performance expected."
        elif pred >= 3.0:
            category = "Very Good"
            message = "Good performance with room for improvement."
        elif pred >= 2.5:
            category = "Good"
            message = "Stable performance. Improve consistency."
        elif pred >= 2.0:
            category = "Average"
            message = "Extra academic effort is recommended."
        else:
            category = "Needs Improvement"
            message = "Student may need additional support."

        st.session_state.history.insert(0, {
            "Predicted CGPA": round(pred, 2),
            "Category": category,
            "Attendance": attendance,
            "Study Hours": study_hours,
            "Final Score": final_score,
        })

        st.markdown(f"""
        <div class="result-box">
            <div class="kpi-label">Predicted CGPA</div>
            <div class="kpi-value">{pred:.2f} / 4.00</div>
            <div class="kpi-sub"><b>{category}</b> — {message}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int((pred / 4.0) * 100))

        rc1, rc2 = st.columns(2)
        rc1.metric("Performance", category)
        rc2.metric("Prediction %", f"{(pred / 4.0) * 100:.1f}%")

    else:
        st.info("Enter student data from the sidebar and click **Predict CGPA**.")

st.write("")

# ---------- Charts ----------
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("### 📊 Academic Scores")
    score_chart = pd.DataFrame({
        "Component": ["Attendance", "Assignment", "Midterm", "Final"],
        "Score": [attendance, assignment_score, midterm_score, final_score],
    }).set_index("Component")
    st.bar_chart(score_chart)

with chart2:
    st.markdown("### 📈 Dataset CGPA Distribution")
    hist_counts, bin_edges = np.histogram(df["cgpa"], bins=10, range=(0, 4))
    hist_df = pd.DataFrame({
        "CGPA Range": [
            f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
            for i in range(len(hist_counts))
        ],
        "Students": hist_counts
    }).set_index("CGPA Range")
    st.bar_chart(hist_df)

st.write("")

# ---------- History ----------
st.markdown("### 🕘 Recent Predictions")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history[:10])
    st.dataframe(history_df, hide_index=True, use_container_width=True)
else:
    st.caption("No predictions yet. Your recent results will appear here.")

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "Demo academic project • Python • Streamlit • Pandas • Scikit-learn • Linear Regression"
)
