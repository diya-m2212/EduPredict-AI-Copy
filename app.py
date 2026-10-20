
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"

PASSING_MARKS = 35

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignment_Score",
    "Previous_Test_Average",
    "Assignment_Completion_Rate",
]

PAGES = [
    "Dashboard",
    "Predict Performance",
    "Academic Analytics",
    "Student Records",
    "About Project",
]

# --------------------------------------------------
# THEME / CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, #28204c 0%, transparent 34%),
            linear-gradient(135deg, #0b1020 0%, #11152b 55%, #17132d 100%);
        color: #f4f2ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17132c 0%, #111326 100%);
        border-right: 1px solid #30294c;
    }

    [data-testid="stSidebar"] * {
        color: #f0edff;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid #4c3d77;
        border-radius: 22px;
        background: linear-gradient(120deg, #35245e 0%, #25224d 48%, #19213e 100%);
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
    }

    .hero h1 {
        font-size: 38px;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: #ffffff;
    }

    .hero p {
        font-size: 15px;
        color: #d7d1f5;
        margin: 0;
        line-height: 1.7;
    }

    .eyebrow {
        color: #c4a8ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #ffffff;
        margin: 18px 0 12px 0;
    }

    .section-subtitle {
        color: #b8b5d2;
        font-size: 13px;
        margin-top: -7px;
        margin-bottom: 16px;
    }

    .info-card {
        background: linear-gradient(145deg, #201d3c, #191b31);
        border: 1px solid #393354;
        border-radius: 16px;
        padding: 18px;
        min-height: 125px;
    }

    .info-label {
        color: #bdb8d7;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .info-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 800;
        line-height: 1.2;
    }

    .info-foot {
        color: #a9a1ce;
        font-size: 11px;
        margin-top: 9px;
    }

    .panel {
        background: #191a31;
        border: 1px solid #35314f;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }

    .panel h3 {
        color: #ffffff;
        margin-top: 0;
        font-size: 17px;
    }

    .panel p, .panel li {
        color: #c6c2dd;
        line-height: 1.7;
        font-size: 13px;
    }

    .purple-note {
        padding: 15px 17px;
        background: #251f43;
        border-left: 4px solid #a78bfa;
        border-radius: 10px;
        color: #e8e1ff;
        margin: 12px 0;
        line-height: 1.6;
    }

    .result-card {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #6950a8;
        background: linear-gradient(135deg, #32215b, #20213e);
        text-align: center;
    }

    .result-number {
        color: #ffffff;
        font-size: 48px;
        font-weight: 800;
    }

    .result-caption {
        color: #d5c8ff;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: #201d3c;
        border: 1px solid #393354;
        border-radius: 14px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #c4bfdc;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid #8b70d1;
        background: linear-gradient(90deg, #7653c8, #6243ad);
        color: white;
    }

    .stButton > button:hover {
        border-color: #c4a8ff;
        background: linear-gradient(90deg, #8968db, #7353c4);
        color: white;
    }

    .small-muted {
        color: #aaa6c7;
        font-size: 12px;
    }

    h1, h2, h3, h4 {
        color: #ffffff;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #393354;
        border-radius: 10px;
    }

    hr {
        border-color: #35314f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# DATA AND MODEL HELPERS
# --------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_PATH)

    numeric_columns = [
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Assignment_Score",
        "Final_Score",
        "Previous_Test_Average",
        "Assignment_Completion_Rate",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def get_predictor():
    try:
        from src.predict import predict_performance
        return predict_performance
    except Exception as error:
        st.error(
            "The prediction module could not be loaded. "
            "Check that src/predict.py exists and contains "
            "predict_performance()."
        )
        with st.expander("Technical details"):
            st.code(str(error))
        return None


def go_to_page(page_name):
    st.session_state["navigation"] = page_name


def metric_card(label, value, foot):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
            <div class="info-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8e5f7", family="Inter"),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ded9f2"),
        ),
        title_font=dict(size=16, color="#ffffff"),
    )
    fig.update_xaxes(
        gridcolor="#34314b",
        zerolinecolor="#34314b",
    )
    fig.update_yaxes(
        gridcolor="#34314b",
        zerolinecolor="#34314b",
    )
    return fig


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Dashboard"

with st.sidebar:
    st.markdown("## 🎓 EduPredict AI")
    st.caption("Student Performance Intelligence")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        PAGES,
        key="navigation",
    )

    st.divider()
    st.markdown("**PROJECT OVERVIEW**")
    st.caption("Machine Learning • Academic Analytics")
    st.caption("Model: Linear Regression")
    st.caption(f"Passing score: {PASSING_MARKS}/100")
    st.divider()
    st.caption("Academic project • Illustrative dataset")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_data()

if df.empty:
    st.error(
        "Student data could not be loaded. Please check that this file exists:"
    )
    st.code("data/student_data.csv")
    st.info(
        "Run the app from your EduPredict-AI project folder and ensure "
        "the CSV contains your student dataset."
    )
    st.stop()

if "Final_Score" not in df.columns:
    st.error("The dataset must contain a 'Final_Score' column.")
    st.stop()

df = df.dropna(subset=["Final_Score"]).copy()

if df.empty:
    st.error("No valid final-score records were found in the dataset.")
    st.stop()

if "Student_ID" not in df.columns:
    df["Student_ID"] = [f"STU-{i:03d}" for i in range(1, len(df) + 1)]

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-POWERED ACADEMIC INSIGHTS</div>
            <h1>EduPredict AI 🎓</h1>
            <p>
                Understand academic performance, explore student data,
                and estimate final scores using a machine-learning model.
                Turn student information into useful learning insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_students = len(df)
    average_score = df["Final_Score"].mean()
    pass_rate = (df["Final_Score"] >= PASSING_MARKS).mean() * 100

    average_attendance = (
        df["Attendance"].mean()
        if "Attendance" in df.columns
        else None
    )

    average_study = (
        df["Study_Hours"].mean()
        if "Study_Hours" in df.columns
        else None
    )

    st.markdown('<div class="section-title">📊 Academic Overview</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A quick summary of the loaded student dataset.</div>',
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)

    with metric_columns[0]:
        metric_card("STUDENT RECORDS", f"{total_students:,}", "Records in dataset")

    with metric_columns[1]:
        metric_card("AVERAGE FINAL SCORE", f"{average_score:.1f}/100", "Mean final score")

    with metric_columns[2]:
        attendance_text = (
            f"{average_attendance:.1f}%"
            if average_attendance is not None
            else "N/A"
        )
        metric_card("AVERAGE ATTENDANCE", attendance_text, "Mean attendance rate")

    with metric_columns[3]:
        study_text = (
            f"{average_study:.1f} hrs"
            if average_study is not None
            else "N/A"
        )
        metric_card("AVERAGE STUDY HOURS", study_text, "Mean reported study hours")

    st.markdown('<div class="section-title">📈 Performance Insights</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        score_fig = px.histogram(
            df,
            x="Final_Score",
            nbins=10,
            title="Final Score Distribution",
            labels={"Final_Score": "Final Score"},
        )
        score_fig.update_traces(
            marker_color="#9b7af5",
            marker_line_color="#d8c9ff",
            marker_line_width=1,
        )
        score_fig.add_vline(
            x=PASSING_MARKS,
            line_dash="dash",
            line_color="#ffb86b",
            annotation_text="Pass threshold",
        )
        st.plotly_chart(
            chart_layout(score_fig),
            use_container_width=True,
        )

    with right:
        if "Attendance" in df.columns:
            attendance_fig = px.scatter(
                df,
                x="Attendance",
                y="Final_Score",
                title="Attendance vs Final Score",
                labels={
                    "Attendance": "Attendance (%)",
                    "Final_Score": "Final Score",
                },
                hover_data=["Student_ID"],
            )
            attendance_fig.update_traces(
                marker=dict(size=11, color="#a78bfa", line=dict(width=1, color="#e9ddff"))
            )
            st.plotly_chart(
                chart_layout(attendance_fig),
                use_container_width=True,
            )
        else:
            st.info("Add an Attendance column to explore this relationship.")

    left, right = st.columns(2)

    with left:
        if "Study_Hours" in df.columns:
            study_fig = px.scatter(
                df,
                x="Study_Hours",
                y="Final_Score",
                title="Study Hours vs Final Score",
                labels={
                    "Study_Hours": "Study Hours",
                    "Final_Score": "Final Score",
                },
                hover_data=["Student_ID"],
            )
            study_fig.update_traces(
                marker=dict(size=11, color="#67d8c4", line=dict(width=1, color="#d1fff5"))
            )
            st.plotly_chart(
                chart_layout(study_fig),
                use_container_width=True,
            )
        else:
            st.info("Add a Study_Hours column to explore this relationship.")

    with right:
        pass_counts = pd.Series(
            {
                "Pass": int((df["Final_Score"] >= PASSING_MARKS).sum()),
                "Below threshold": int((df["Final_Score"] < PASSING_MARKS).sum()),
            }
        )
        pass_fig = px.pie(
            names=pass_counts.index,
            values=pass_counts.values,
            title=f"Score Threshold Overview (≥ {PASSING_MARKS})",
            hole=0.60,
        )
        pass_fig.update_traces(
            textinfo="label+percent",
            marker=dict(
                colors=["#8e72e8", "#e7a06b"],
                line=dict(color="#191a31", width=3),
            ),
        )
        st.plotly_chart(
            chart_layout(pass_fig),
            use_container_width=True,
        )

    st.markdown('<div class="section-title">💡 Dataset Highlights</div>',
                unsafe_allow_html=True)

    best_score = df["Final_Score"].max()
    lowest_score = df["Final_Score"].min()
    below_threshold = int((df["Final_Score"] < PASSING_MARKS).sum())

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="panel">
                <h3>🏆 Highest Score</h3>
                <p style="font-size:28px;color:#c4a8ff;font-weight:800">
                    {best_score:.1f}/100
                </p>
                <p>Highest recorded final score in this dataset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="panel">
                <h3>📘 Lowest Score</h3>
                <p style="font-size:28px;color:#f0b77d;font-weight:800">
                    {lowest_score:.1f}/100
                </p>
                <p>Lowest recorded final score in this dataset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="panel">
                <h3>🎯 Support Needed</h3>
                <p style="font-size:28px;color:#e7a06b;font-weight:800">
                    {below_threshold}
                </p>
                <p>Records below the configured threshold of {PASSING_MARKS}/100.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        '<div class="section-title">🚀 Explore EduPredict AI</div>',
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3, gap="medium")

    with b1:
        st.markdown(
            """
            <div class="panel" style="height:170px; box-sizing:border-box;">
                <h3>🔮 Predict Performance</h3>
                <p>Enter academic details and estimate a final score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Go to Prediction",
            key="dashboard_predict",
            on_click=go_to_page,
            args=("Predict Performance",),
            use_container_width=True,
        )

    with b2:
        st.markdown(
            """
            <div class="panel" style="height:170px; box-sizing:border-box;">
                <h3>📊 Academic Analytics</h3>
                <p>Explore score patterns, distributions, and relationships.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Explore Analytics",
            key="dashboard_analytics",
            on_click=go_to_page,
            args=("Academic Analytics",),
            use_container_width=True,
        )

    with b3:
        st.markdown(
            """
            <div class="panel" style="height:170px; box-sizing:border-box;">
                <h3>🗂️ Student Records</h3>
                <p>Search, filter, and review available student records.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "View Records",
            key="dashboard_records",
            on_click=go_to_page,
            args=("Student Records",),
            use_container_width=True,
        )

    st.caption(
        "Note: These results describe the dataset currently loaded. "
        "They should not be treated as verified real-world academic findings."
    )


# --------------------------------------------------
# PREDICT PERFORMANCE
# --------------------------------------------------

elif page == "Predict Performance":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">PERSONALIZED SCORE ESTIMATION</div>
            <h1>🔮 Predict Performance</h1>
            <p>Enter a student's academic details to estimate their final score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="purple-note">Fill in all six inputs. The model uses '
        'study hours, attendance, previous scores, and assignment information.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("### 📚 Student Academic Details")

        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.number_input(
                "Daily Study Hours",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5,
                help="Average number of hours studied per day.",
            )

            attendance = st.slider(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=80,
            )

            previous_score = st.slider(
                "Previous Score",
                min_value=0,
                max_value=100,
                value=70,
            )

        with col2:
            assignment_score = st.slider(
                "Assignment Score",
                min_value=0,
                max_value=100,
                value=75,
            )

            previous_test_average = st.slider(
                "Previous Test Average",
                min_value=0,
                max_value=100,
                value=70,
            )

            assignment_completion_rate = st.slider(
                "Assignment Completion Rate (%)",
                min_value=0,
                max_value=100,
                value=85,
            )

        submitted = st.form_submit_button(
            "✨ Predict Final Score",
            use_container_width=True,
        )

    if submitted:
        predictor = get_predictor()

        if predictor is not None:
            try:
                prediction = predictor(
                    study_hours=study_hours,
                    attendance=attendance,
                    previous_score=previous_score,
                    assignment_score=assignment_score,
                    previous_test_average=previous_test_average,
                    assignment_completion_rate=assignment_completion_rate,
                )

                if hasattr(prediction, "item"):
                    prediction = prediction.item()

                prediction = max(0.0, min(100.0, float(prediction)))

                st.markdown("### 🎯 Prediction Result")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-caption">ESTIMATED FINAL SCORE</div>
                        <div class="result-number">{prediction:.1f}<span style="font-size:22px">/100</span></div>
                        <div class="result-caption">
                            {"At or above the configured score threshold" if prediction >= PASSING_MARKS else "Below the configured score threshold"}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(int(round(prediction)))

                if prediction >= PASSING_MARKS:
                    st.success(
                        f"The estimated score is at or above the configured "
                        f"threshold of {PASSING_MARKS}/100."
                    )
                else:
                    st.warning(
                        f"The estimated score is below the configured "
                        f"threshold of {PASSING_MARKS}/100. Consider additional study support."
                    )

                st.markdown("### 🧭 Suggested Improvement Plan")

                suggestions = []

                if study_hours < 2:
                    suggestions.append(
                        "Build a consistent study routine and gradually increase daily study time."
                    )
                elif study_hours < 4:
                    suggestions.append(
                        "Try focused study sessions with short breaks and weekly revision."
                    )

                if attendance < 75:
                    suggestions.append(
                        "Improve class attendance and review missed lessons."
                    )

                if previous_score < 60:
                    suggestions.append(
                        "Revise foundational concepts and practice questions from previous topics."
                    )

                if assignment_score < 60:
                    suggestions.append(
                        "Review assignment feedback and practise similar questions."
                    )

                if previous_test_average < 60:
                    suggestions.append(
                        "Use short practice tests to identify and improve weak areas."
                    )

                if assignment_completion_rate < 80:
                    suggestions.append(
                        "Set small deadlines and aim to complete assignments on time."
                    )

                if not suggestions:
                    suggestions.append(
                        "Maintain your current routine, revise regularly, and use practice tests to monitor progress."
                    )

                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")

                with st.expander("View input details"):
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Feature": [
                                    "Study Hours",
                                    "Attendance",
                                    "Previous Score",
                                    "Assignment Score",
                                    "Previous Test Average",
                                    "Assignment Completion Rate",
                                ],
                                "Value": [
                                    study_hours,
                                    attendance,
                                    previous_score,
                                    assignment_score,
                                    previous_test_average,
                                    assignment_completion_rate,
                                ],
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    "This is a model-based estimate, not a guaranteed result. "
                    "Its reliability depends on the quality and size of the training data."
                )

            except Exception as error:
                st.error("Prediction could not be completed.")
                with st.expander("Technical details"):
                    st.code(str(error))


# --------------------------------------------------
# ACADEMIC ANALYTICS
# --------------------------------------------------

elif page == "Academic Analytics":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">EXPLORE THE DATA</div>
            <h1>📊 Academic Analytics</h1>
            <p>Explore score patterns and relationships in the available student records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        st.warning("No numeric columns are available for analytics.")
    else:
        st.markdown("### Select an analysis")

        available_columns = list(numeric_df.columns)

        default_x = (
            "Study_Hours" if "Study_Hours" in available_columns
            else available_columns[0]
        )
        default_y = (
            "Final_Score" if "Final_Score" in available_columns
            else available_columns[-1]
        )

        xcol, ycol = st.columns(2)

        with xcol:
            x_axis = st.selectbox(
                "X-axis",
                available_columns,
                index=available_columns.index(default_x),
            )

        with ycol:
            y_axis = st.selectbox(
                "Y-axis",
                available_columns,
                index=available_columns.index(default_y),
            )

        
        if x_axis == y_axis:
            plot_df = numeric_df[[x_axis]].dropna().copy()

            histogram = px.histogram(
                plot_df,
                x=x_axis,
                nbins=12,
                title=f"Distribution of {x_axis}",
            )
            histogram.update_traces(marker_color="#9b7af5")

            st.plotly_chart(
                chart_layout(histogram, 380),
                use_container_width=True,
            )

            st.info(
                "You selected the same feature for both axes. "
                "Showing its distribution instead of a relationship chart."
            )

        else:
            plot_df = numeric_df[[x_axis, y_axis]].dropna().copy()

            scatter = px.scatter(
                plot_df,
                x=x_axis,
                y=y_axis,
                title=f"{x_axis} vs {y_axis}",
            )
            scatter.update_traces(
                marker=dict(size=10, color="#a78bfa")
            )

            st.plotly_chart(
                chart_layout(scatter, 380),
                use_container_width=True,
            )

            correlation = plot_df[x_axis].corr(plot_df[y_axis])

            if pd.notna(correlation):
                st.metric(
                    "Pearson correlation",
                    f"{correlation:.3f}",
                )

                if abs(correlation) < 0.3:
                    interpretation = (
                        "The linear relationship in this dataset appears weak."
                    )
                elif abs(correlation) < 0.7:
                    interpretation = (
                        "The dataset shows a moderate linear relationship."
                    )
                else:
                    interpretation = (
                        "The dataset shows a strong linear relationship."
                    )

                st.markdown(
                    f'<div class="purple-note">{interpretation} '
                    'Correlation alone does not prove causation.</div>',
                    unsafe_allow_html=True,
                )



        st.markdown("### Numeric Feature Correlation")

        corr = numeric_df.corr(numeric_only=True)

        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale=[
                    [0.0, "#e7a06b"],
                    [0.5, "#282542"],
                    [1.0, "#9b7af5"],
                ],
                text=corr.round(2).values,
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            )
        )
        heatmap.update_layout(title="Correlation Matrix")
        st.plotly_chart(
            chart_layout(heatmap, 480),
            use_container_width=True,
        )

        st.markdown("### Descriptive Statistics")
        st.dataframe(
            numeric_df.describe().round(2),
            use_container_width=True,
        )


# --------------------------------------------------
# STUDENT RECORDS
# --------------------------------------------------

elif page == "Student Records":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">DATA EXPLORER</div>
            <h1>🗂️ Student Records</h1>
            <p>Search, filter, and review the student dataset loaded into the application.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered_df = df.copy()

    search_text = st.text_input(
        "🔎 Search by Student ID",
        placeholder="Enter a student ID...",
    )

    if search_text.strip():
        filtered_df = filtered_df[
            filtered_df["Student_ID"]
            .astype(str)
            .str.contains(search_text.strip(), case=False, na=False)
        ]

    score_range = st.slider(
        "Filter by Final Score",
        min_value=0,
        max_value=100,
        value=(0, 100),
    )

    filtered_df = filtered_df[
        filtered_df["Final_Score"].between(
            score_range[0],
            score_range[1],
        )
    ]

    if "Attendance" in filtered_df.columns:
        attendance_range = st.slider(
            "Filter by Attendance (%)",
            min_value=0,
            max_value=100,
            value=(0, 100),
        )

        filtered_df = filtered_df[
            filtered_df["Attendance"].between(
                attendance_range[0],
                attendance_range[1],
            )
        ]

    st.markdown(f"**Records found:** {len(filtered_df)}")

    display_df = filtered_df.copy()

    if "Final_Score" in display_df.columns:
        display_df["Performance_Status"] = display_df["Final_Score"].apply(
            lambda score: (
                "At/above threshold"
                if score >= PASSING_MARKS
                else "Below threshold"
            )
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Records (CSV)",
        data=csv_data,
        file_name="edupredict_student_records.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        f"Performance status uses the project's configured threshold of {PASSING_MARKS}/100."
    )


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">PROJECT DOCUMENTATION</div>
            <h1>💜 About EduPredict AI</h1>
            <p>A student performance prediction and academic analytics project using machine learning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Project Objective")

    st.markdown(
        """
        EduPredict AI is designed to explore student academic information and
        estimate final scores from selected academic indicators. It also
        provides charts and data summaries to help users understand patterns
        in the available dataset.
        """
    )

    st.markdown("### ⚙️ Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown(
            """
            <div class="panel">
                <h3>🐍 Python</h3>
                <p>Core programming language for data handling and model integration.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech2:
        st.markdown(
            """
            <div class="panel">
                <h3>🧠 Scikit-learn</h3>
                <p>Machine-learning tools, including the Linear Regression model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech3:
        st.markdown(
            """
            <div class="panel">
                <h3>📊 Streamlit & Plotly</h3>
                <p>Interactive application pages, charts, and data exploration.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🧪 Model Information")

    st.markdown(
        f"""
        <div class="panel">
            <h3>Linear Regression</h3>
            <p>
                The prediction page uses the project's trained Linear Regression
                model to estimate a final score from six inputs.
            </p>
            <p><b>Input features:</b> Study Hours, Attendance, Previous Score,
            Assignment Score, Previous Test Average, and Assignment Completion Rate.</p>
            <p><b>Target:</b> Final Score (0–100).</p>
            <p><b>Configured score threshold:</b> {PASSING_MARKS}/100.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📁 Project Structure")

    st.code(
        """
EduPredict-AI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── student_data.csv
├── models/
│   └── student_performance_model.pkl
└── src/
    ├── __init__.py
    ├── train_model.py
    ├── predict.py
    └── add_features.py
        """.strip(),
        language="text",
    )

st.markdown("### ⚠️ Important Limitations")

st.info(
    "The current dataset contains only a small number of illustrative records. "
    "Model predictions and visual patterns may not generalize to real students. "
    "A larger, representative dataset and independent evaluation are needed "
    "before using the model for real academic decisions."
    )
st.markdown("### 🔧 Useful Commands")

st.code(
    """python src/train_model.py
streamlit run app.py""",
    language="bash",
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:8px 0 18px 0; color:#aaa6c7; font-size:12px;">
        🎓 EduPredict AI · Student Performance Prediction System<br>
        Built with Python, Scikit-learn, Streamlit, Pandas, and Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
