# Code by Ishan Katoch

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler


# Function to format the classification report as a DataFrame
def format_classification_report(report):
    report_data = []
    lines = report.split("\n")
    for line in lines[2:-5]:
        row_data = line.split()
        row_data[0] = str(row_data[0])  
        row_data[1:] = [float(val) for val in row_data[1:]]
        report_data.append(row_data)

    df = pd.DataFrame(
        report_data, columns=["Class", "Precision", "Recall", "F1-Score", "Support"]
    )
    return df


st.title("StreamTune - Interactive Machine Learning Model Tuning")
st.write(
    "Upload your dataset, select models, and tune hyperparameters to optimize model performance."
)

# Sidebar - Dataset Upload and Sample Datasets
st.sidebar.title("Upload or Use Sample Dataset")
sample_dataset = st.sidebar.selectbox(
    "Select a sample dataset", ["Iris", "Breast Cancer", "Custom"]
)

if sample_dataset == "Iris":
    data = load_iris()  # Add the missing dataset loading function
elif sample_dataset == "Breast Cancer":
    data = load_breast_cancer()  # Add the missing dataset loading function
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV or Excel file", type=["csv", "xlsx"]
    )
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)  # Use pd.read_excel() for Excel files
    elif uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    else:
        data = None

if data is not None:
    if isinstance(data, dict):
        df = pd.DataFrame(data=data["data"], columns=data["feature_names"])
        df["target"] = data["target"]
        target_mapping = {
            i: data["target_names"][i] for i in range(len(data["target_names"]))
        }
        df["target"] = df["target"].map(target_mapping)
    else:
        df = data

    st.write("Data Preview:")
    st.dataframe(df.head())

    st.sidebar.title("Select Target Variable")
    target_variable = st.sidebar.selectbox("Select the target variable", df.columns)

    st.sidebar.title("Tune Hyperparameters")

    models_with_tuning = {
        "Random Forest": {
            "n_estimators": st.sidebar.slider(
                "Number of Trees (Random Forest)", 1, 100, 10
            ),
            "max_depth": st.sidebar.slider("Max Depth (Random Forest)", 1, 20, 10),
        },
        "k-Nearest Neighbors": {
            "n_neighbors": st.sidebar.slider(
                "Number of Neighbors (k-Nearest Neighbors)", 1, 20, 5
            )
        },
        "Gradient Boosting": {
            "n_estimators_gb": st.sidebar.slider(
                "Number of Estimators (Gradient Boosting)", 1, 100, 10
            ),
            "max_depth_gb": st.sidebar.slider(
                "Max Depth (Gradient Boosting)", 1, 20, 3
            ),
        },
    }

    non_tunable_models = ["Logistic Regression", "Support Vector Machine"]

    st.write("## Model Comparison and Evaluation:")
    selected_models = st.sidebar.multiselect(
        "Select models for comparison",
        list(models_with_tuning.keys()) + non_tunable_models,
    )

    if len(selected_models) == 0:
        st.warning("Please select models for comparison from the sidebar.")
    else:
        # Data Splitting and Model Training
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for selected_model in selected_models:
            st.write(f"### {selected_model}")

            if selected_model in models_with_tuning:
                model = models_with_tuning[selected_model]

                if selected_model == "Random Forest":
                    n_estimators_rf = model["n_estimators"]
                    max_depth_rf = model["max_depth"]
                    model_instance = RandomForestClassifier(
                        n_estimators=n_estimators_rf,
                        max_depth=max_depth_rf,
                        random_state=42,
                    )

                elif selected_model == "k-Nearest Neighbors":
                    n_neighbors_knn = model["n_neighbors"]
                    model_instance = KNeighborsClassifier(n_neighbors=n_neighbors_knn)

                elif selected_model == "Gradient Boosting":
                    n_estimators_gb = model["n_estimators_gb"]
                    max_depth_gb = model["max_depth_gb"]
                    model_instance = GradientBoostingClassifier(
                        n_estimators=n_estimators_gb,
                        max_depth=max_depth_gb,
                        random_state=42,
                    )

            elif selected_model == "Logistic Regression":
                model_instance = LogisticRegression(max_iter=1000)

            elif selected_model == "Support Vector Machine":
                model_instance = SVC(kernel="linear", C=1.0, random_state=42)

            # Fit the model
            if selected_model == "k-Nearest Neighbors":
                model_instance.fit(X_train_scaled, y_train)  # Use scaled data
            else:
                model_instance.fit(X_train, y_train)

            # Make predictions
            y_pred = model_instance.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Center the accuracy value and add a white border around it
            st.markdown(
                f"<p style='text-align:center; padding: 10px; border: 1px solid white; border-radius: 5px;'>"
                f"<b style='font-size: 18px;'>Accuracy:</b> {accuracy:.2f}</p>",
                unsafe_allow_html=True,
            )

            st.write("#### Confusion Matrix:")

            query_params = st.experimental_get_query_params()
            is_dark_theme = query_params.get("theme", ["light"])[0] == "dark"

            background_color = "#1a1a1a" if is_dark_theme else "#f0f0f0"
            text_color = "#ffffff" if is_dark_theme else "#000000"

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))  # Adjust the figure size for better visibility
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues" if is_dark_theme else "Greens",
                cbar=False,
                linewidths=1,
                linecolor="white",
                annot_kws={"color": text_color},
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            plt.gca().set_facecolor(background_color)

            plt.gca().tick_params(axis="both", colors=text_color)

            st.pyplot(plt)

            st.write("#### Classification Report:")
            report = classification_report(y_test, y_pred)
            df_report = format_classification_report(report)
            st.table(df_report)

            # Add some distance between the classification report and the feedback form
            st.write("")  # Empty line
            st.write("")  # Empty line
            st.write("")  # Empty line
            st.write("")  # Empty line
            st.write("")  # Empty line
            st.write("")  # Empty line
            st.write("")  # Empty line

# User Feedback Form
feedback_expander = st.expander("Feedback Form", expanded=False)
with feedback_expander.form(key="feedback_form"):
    st.markdown(
        "<p style='font-size: 24px; font-weight: bold; color: #ff6600; text-align: center;'>We'd Love to Hear from You!</p>",
        unsafe_allow_html=True,
    )
    feedback_text = st.text_area("", "Please share your feedback here.", height=150)
    submit_button = st.form_submit_button(label="Submit Feedback")

# Process user feedback
if submit_button:
    with open("feedback.txt", "a") as f:
        f.write(feedback_text + "\n")

    st.success("Feedback submitted successfully. Thank you!")
