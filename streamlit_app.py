import streamlit as st

from model_service import FEATURE_NAMES, load_artifact, predict_species_from_artifact


DEFAULT_FEATURES = [5.1, 3.5, 1.4, 0.2]
FEATURE_LIMITS = {
    "sepal length (cm)": (0.1, 10.0),
    "sepal width (cm)": (0.1, 10.0),
    "petal length (cm)": (0.1, 10.0),
    "petal width (cm)": (0.1, 10.0),
}


@st.cache_resource
def get_artifact():
    return load_artifact()


def main() -> None:
    st.set_page_config(page_title="Iris Classifier", layout="centered")
    st.title("Iris Classifier")

    try:
        artifact = get_artifact()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    metrics = artifact["metrics"]
    metric_columns = st.columns(2)
    metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    metric_columns[1].metric("F1-score", f"{metrics['f1_score']:.3f}")

    with st.form("prediction"):
        columns = st.columns(2)
        features = []

        for index, feature_name in enumerate(FEATURE_NAMES):
            min_value, max_value = FEATURE_LIMITS[feature_name]
            value = columns[index % 2].number_input(
                feature_name.title(),
                min_value=min_value,
                max_value=max_value,
                value=DEFAULT_FEATURES[index],
                step=0.1,
            )
            features.append(value)

        submitted = st.form_submit_button("Predict")

    if submitted:
        result = predict_species_from_artifact(features, artifact)
        st.success(result["predicted_label"].title())
        st.caption(f"Class id: {result['predicted_class']}")


if __name__ == "__main__":
    main()
