import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(
    page_title="AI Medicine Detection System",
    page_icon="💊",
    layout="wide"
)

MODEL_PATH = "medicine_model.keras"
CLASS_FILE = "class_indices.json"
MEDICINE_FILE = "medicines.json"
DATASET_PATH = "prepared_images"

IMAGE_SIZE = (224, 224)


@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


@st.cache_data
def load_classes():

    with open(
        CLASS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    class_names = [None] * len(data)

    for name, index in data.items():

        class_names[int(index)] = name

    return class_names


@st.cache_data
def load_medicines():

    with open(
        MEDICINE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


@st.cache_resource
def create_feature_model():

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])


@st.cache_data
def create_image_database():

    feature_model = create_feature_model()

    database_features = []
    database_names = []

    extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    for medicine in os.listdir(DATASET_PATH):

        medicine_path = os.path.join(
            DATASET_PATH,
            medicine
        )

        if not os.path.isdir(
            medicine_path
        ):
            continue

        for filename in os.listdir(
            medicine_path
        ):

            if not filename.lower().endswith(
                extensions
            ):
                continue

            image_path = os.path.join(
                medicine_path,
                filename
            )

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                image = image.resize(
                    IMAGE_SIZE
                )

                image_array = np.array(
                    image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                    image_array
                )

                feature = feature_model.predict(
                    image_array,
                    verbose=0
                )[0]

                feature = feature / (
                    np.linalg.norm(feature) + 1e-8
                )

                database_features.append(
                    feature
                )

                database_names.append(
                    medicine
                )

            except Exception:
                continue

    return (
        np.array(database_features),
        database_names
    )


model = load_model()
class_names = load_classes()
medicines = load_medicines()

feature_model = create_feature_model()

database_features, database_names = create_image_database()


if "history" not in st.session_state:

    st.session_state.history = []


def find_medicine(
    medicine_name
):

    if isinstance(
        medicines,
        dict
    ):

        for name, data in medicines.items():

            if name.lower() == medicine_name.lower():

                return data

    elif isinstance(
        medicines,
        list
    ):

        for data in medicines:

            if not isinstance(
                data,
                dict
            ):

                continue

            name = data.get(
                "name",
                ""
            )

            if name.lower() == medicine_name.lower():

                return data

    return None


def get_image_features(
    image
):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    feature = feature_model.predict(
        image_array,
        verbose=0
    )[0]

    feature = feature / (
        np.linalg.norm(feature) + 1e-8
    )

    return feature


def detect_medicine(
    image
):

    uploaded_feature = get_image_features(
        image
    )

    similarities = np.dot(
        database_features,
        uploaded_feature
    )

    best_index = int(
        np.argmax(similarities)
    )

    detected_medicine = database_names[
        best_index
    ]

    return detected_medicine


st.sidebar.title(
    "💊 Medicine Detection"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Detect Medicine",
        "History",
        "About",
        "Safety"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI-based educational medicine identification system."
)


if page == "Home":

    st.title(
        "💊 AI Medicine Detection System"
    )

    st.subheader(
        "Identify Medicine from Image"
    )

    st.write(
        "Upload or capture a medicine image "
        "to identify the medicine and view "
        "its information."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Medicine Classes",
            "5"
        )

    with col2:

        st.metric(
            "Reference Images",
            "123"
        )

    with col3:

        st.metric(
            "AI Technology",
            "MobileNetV2"
        )

    st.markdown("---")

    st.subheader(
        "Supported Medicines"
    )

    for medicine in class_names:

        st.write(
            f"💊 {medicine}"
        )

    st.markdown("---")

    st.warning(
        "This application is an educational college project. "
        "Always verify the medicine name on the original "
        "packaging before using any medicine."
    )


elif page == "Detect Medicine":

    st.title(
        "🔍 Detect Medicine"
    )

    language = st.selectbox(
        "Select Information Language",
        [
            "English",
            "Hindi",
            "Marathi"
        ]
    )

    information_type = st.multiselect(
        "Select Information",
        [
            "Symptoms",
            "Uses",
            "Price",
            "Pack Information",
            "Age Information",
            "Warning"
        ],
        default=[
            "Symptoms",
            "Uses",
            "Price",
            "Pack Information",
            "Age Information",
            "Warning"
        ]
    )

    input_method = st.radio(
        "Choose Image Source",
        [
            "Upload Image",
            "Use Camera"
        ]
    )

    uploaded_file = None

    if input_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload medicine image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    else:

        uploaded_file = st.camera_input(
            "Take a picture of the medicine"
        )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Selected Medicine Image",
            width="stretch"
        )

        if st.button(
            "🔍 Identify Medicine",
            width="stretch"
        ):

            with st.spinner(
                "Comparing medicine image..."
            ):

                detected_medicine = detect_medicine(
                    image
                )

            medicine_data = find_medicine(
                detected_medicine
            )

            st.markdown("---")

            st.success(
                f"Detected Medicine: {detected_medicine}"
            )

            if medicine_data is not None:

                st.subheader(
                    "💊 Medicine Information"
                )

                medicine_name = medicine_data.get(
                    "name",
                    detected_medicine
                )

                st.markdown(
                    f"### {medicine_name}"
                )

                if "Symptoms" in information_type:

                    if language == "English":

                        text = medicine_data.get(
                            "symptoms",
                            "Information not available."
                        )

                    elif language == "Hindi":

                        text = medicine_data.get(
                            "symptoms_hindi",
                            medicine_data.get(
                                "symptoms",
                                "Information not available."
                            )
                        )

                    else:

                        text = medicine_data.get(
                            "symptoms_marathi",
                            medicine_data.get(
                                "symptoms",
                                "Information not available."
                            )
                        )

                    st.markdown(
                        "#### 🩺 Symptoms"
                    )

                    st.write(text)

                if "Uses" in information_type:

                    if language == "English":

                        text = medicine_data.get(
                            "uses",
                            "Information not available."
                        )

                    elif language == "Hindi":

                        text = medicine_data.get(
                            "uses_hindi",
                            medicine_data.get(
                                "uses",
                                "Information not available."
                            )
                        )

                    else:

                        text = medicine_data.get(
                            "uses_marathi",
                            medicine_data.get(
                                "uses",
                                "Information not available."
                            )
                        )

                    st.markdown(
                        "#### 💡 Uses"
                    )

                    st.write(text)

                if "Price" in information_type:

                    st.markdown(
                        "#### 💰 Price"
                    )

                    st.write(
                        medicine_data.get(
                            "price",
                            "Information not available."
                        )
                    )

                if "Pack Information" in information_type:

                    st.markdown(
                        "#### 📦 Pack Information"
                    )

                    st.write(
                        medicine_data.get(
                            "pack_information",
                            "Information not available."
                        )
                    )

                if "Age Information" in information_type:

                    st.markdown(
                        "#### 👤 Age Information"
                    )

                    st.write(
                        medicine_data.get(
                            "age_information",
                            "Information not available."
                        )
                    )

                if "Warning" in information_type:

                    st.markdown(
                        "#### ⚠️ Warning"
                    )

                    st.warning(
                        medicine_data.get(
                            "warning",
                            "Consult a doctor or pharmacist."
                        )
                    )

                st.session_state.history.append(
                    {
                        "medicine": detected_medicine
                    }
                )

            else:

                st.error(
                    "Medicine information is not available."
                )

            st.markdown("---")

            st.info(
                "Please verify the medicine name on "
                "the original packaging before using "
                "any medicine."
            )


elif page == "History":

    st.title(
        "📜 Detection History"
    )

    if not st.session_state.history:

        st.info(
            "No medicine identification history available."
        )

    else:

        for i, item in enumerate(
            reversed(
                st.session_state.history
            ),
            start=1
        ):

            st.write(
                f"### {i}. {item['medicine']}"
            )

            st.markdown("---")

        if st.button(
            "Clear History"
        ):

            st.session_state.history = []

            st.success(
                "History cleared successfully."
            )

            st.rerun()


elif page == "About":

    st.title(
        "ℹ️ About the Project"
    )

    st.subheader(
        "AI Medicine Detection System"
    )

    st.write(
        "This project is a Python-based AI application "
        "developed to identify five medicine categories "
        "from images."
    )

    st.write(
        "The system uses MobileNetV2 image features "
        "and compares the uploaded image with existing "
        "reference medicine images."
    )

    st.markdown("---")

    st.subheader(
        "Technology Used"
    )

    st.write("Python")
    st.write("Streamlit")
    st.write("TensorFlow / Keras")
    st.write("MobileNetV2")
    st.write("NumPy")
    st.write("Pillow")

    st.markdown("---")

    st.subheader(
        "Supported Medicines"
    )

    for medicine in class_names:

        st.write(
            f"💊 {medicine}"
        )

    st.markdown("---")

    st.subheader(
        "Model Testing"
    )

    st.write(
        "The original trained model was tested using "
        "123 prepared images from five medicine categories."
    )

    st.write(
        "Overall classification accuracy: 66.67%"
    )

    st.info(
        "The project uses a small dataset and is intended "
        "for educational demonstration only."
    )


elif page == "Safety":

    st.title(
        "⚠️ Safety Information"
    )

    st.warning(
        "This application is an educational college project "
        "and is not a medical diagnostic system."
    )

    st.subheader(
        "Important Guidelines"
    )

    st.write(
        "• Always verify the medicine name on the original packaging."
    )

    st.write(
        "• Do not use the application to decide which medicine to take."
    )

    st.write(
        "• Consult a qualified doctor or pharmacist when uncertain."
    )

    st.write(
        "• Do not use the application for dosage or treatment decisions."
    )

    st.write(
        "• Always check the medicine label before use."
    )

    st.markdown("---")

    st.subheader(
        "Project Limitation"
    )

    st.write(
        "The system supports five medicine categories. "
        "Image quality, lighting, camera angle and packaging "
        "appearance can affect identification."
    )

    st.success(
        "For educational demonstration only."
    )