import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import io

from mining import (
    load_and_preprocess_data,
    run_prefixspan,
    run_gsp,
    build_ml_dataset,
    train_and_evaluate_models
)

st.set_page_config(
    page_title="Student Learning Behavior Analysis",
    page_icon="🎓",
    layout="wide"
)

st.title("Student Learning Behavior Sequence Analysis")
st.caption(
    "Analyzing student learning behavior using Sequential Pattern Mining "
    "(PrefixSpan & GSP) and Machine Learning models."
)

# Sidebar
st.sidebar.markdown("## Analysis Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Student Interaction Dataset (CSV)",
    type="csv"
)

# Minimum Support
min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=1,
    max_value=20,
    value=2
)

# Helper Functions
def get_top_sequences(df, label, top_n=5):
    seqs = (
        df[df['success_label'] == label]
        .groupby('student_id')['activity']
        .apply(lambda x: " → ".join(x[:3]))
        .value_counts(normalize=True)
        .head(top_n) * 100
    )
    return seqs


def plot_sequence_distribution(df, label, title):
    seqs = get_top_sequences(df, label)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(seqs.index[::-1], seqs.values[::-1])
    ax.set_xlabel("Percentage (%)")
    ax.set_title(title)

    for i, v in enumerate(seqs.values[::-1]):
        ax.text(v + 1, i, f"{v:.1f}%", va='center')

    return fig


# Main Logic
if uploaded_file is not None:

    df, sequences = load_and_preprocess_data(uploaded_file)

    if sequences.empty:
        st.error("No valid sequences found in the dataset.")
        st.stop()

    # Dataset Summary
    st.markdown("## Dataset Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", sequences['student_id'].nunique())
    c2.metric("Average Sequence Length", round(sequences['sequence_length'].mean(), 2))
    c3.metric("High Performers", int(sequences['success_label'].sum()))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset",
        "Visualizations",
        "Pattern Mining",
        "Modeling"
    ])

    # Tab 1: Dataset
    with tab1:
        st.subheader("Sample of Dataset")
        st.dataframe(df.head())

        st.markdown("##  Data Quality Check")

        q1, q2, q3 = st.columns(3)

        # ---- Missing Values ----
        missing_count = df.isna().sum().sum()
        if missing_count == 0:
            q1.success(" No Missing Values (NaN)")
        else:
            q1.error(f" Missing Values: {missing_count}")

        dup_count = df.duplicated().sum()
        if dup_count == 0:
            q2.success(" No Duplicated Rows")
        else:
            q2.error(f" Duplicates Found: {dup_count}")

        q3.info(f" Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        st.markdown("## Dataset Information")
        with st.expander("Show Data Info"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

        st.markdown("## Statistical Summary")
        with st.expander("Show Describe Statistics"):
            st.dataframe(df.describe().round(2))

    # Tab 2: Visualizations
    with tab2:

        st.subheader("Student Performance Distribution")

        perf = sequences['success_label'].value_counts().sort_index()
        labels = ['Low', 'High']
        total = perf.sum()

        fig0, ax0 = plt.subplots(figsize=(5, 3))
        ax0.bar(labels, perf.values)
        ax0.set_ylabel("Number of Students")

        for i, v in enumerate(perf.values):
            ax0.text(i, v + 0.5, f"{v} ({v/total:.0%})", ha='center')

        st.pyplot(fig0)

        st.caption(
            "The dataset is balanced, which helps avoid bias in classification models."
        )

        st.subheader("Top Learning Sequences by Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### High Performing Students")
            st.pyplot(plot_sequence_distribution(df, 1, "High Performance Sequences"))

        with col2:
            st.markdown("#### Low Performing Students")
            st.pyplot(plot_sequence_distribution(df, 0, "Low Performance Sequences"))

        st.subheader("Sequence Length vs Performance")

        fig3, ax3 = plt.subplots(figsize=(5, 3))
        sequences.boxplot(
            column='sequence_length',
            by='success_label',
            ax=ax3
        )
        ax3.set_xlabel("Performance (0 = Low, 1 = High)")
        ax3.set_ylabel("Sequence Length")
        plt.suptitle("")
        st.pyplot(fig3)

        st.caption(
            "Performance differences are driven more by the type and order of "
            "learning activities rather than the number of actions."
        )

    # Tab 3: Pattern Mining
    with tab3:
        st.subheader("Sequential Pattern Mining")

        high_p, low_p = run_prefixspan(sequences, min_support)

        c1, c2 = st.columns(2)

        with c1:
            st.success("High Performers (PrefixSpan)")
            for s, p in high_p[:5]:
                st.write(f"Support: {s} → {p}")

        with c2:
            st.error("Low Performers (PrefixSpan)")
            for s, p in low_p[:5]:
                st.write(f"Support: {s} → {p}")

        st.markdown("### GSP Patterns")
        for s, p in run_gsp(sequences, min_support)[:5]:
            st.write(f"Support: {s} → {p}")

    # Tab 4: Modeling
    with tab4:
        st.subheader("Model Performance")

        X, y = build_ml_dataset(df, sequences)
        results = train_and_evaluate_models(X, y)

        names = list(results.keys())
        accs = [results[m]['accuracy'] for m in names]

        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.bar(names, accs)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel("Accuracy")
        st.pyplot(fig4)

        with st.expander("Classification Reports"):
            for name, res in results.items():
                st.markdown(f"### {name}")
                st.dataframe(
                    pd.DataFrame(res['report_dict']).transpose().round(2)
                )

else:
    st.info("Upload a dataset to start the analysis.")
