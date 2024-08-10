import os
import pickle

import numpy as np
import pandas as pd
import requests

import deepmreye
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")


def main():
    st.image(
        os.path.join(os.getcwd(), "media/deepmreye_logo_t.png"),
        caption=None,
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
    st.markdown(
        """

        This app enables reconstructing gaze position from the MR-signal of the eyeballs.
        Load your fMRI data below, pick one of the pretrained models,
        and download the decoded gaze coordinates shortly after.

        Please read the [paper](https://doi.org/10.1038/s41593-021-00947-w)
        and [user recommendations](https://deepmreye.slite.com/p/channel/MUgmvViEbaATSrqt3susLZ/notes/kKdOXmLqe)
        before using it.
        """
    )

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a NIFTI file (head motion-corrected)", type=["nii"])

    # Choose model weights
    model_str = st.selectbox(
        "Choose pretrained model",
        (
            "datasets_1to5",
            "datasets_1to6",
            "dataset1_guided_fixations",
            "dataset2_pursuit",
            "dataset3_pursuit",
            "dataset4_pursuit",
            "dataset5_free_viewing",
            # "dataset6_openclosed",
        ),
    )

    st.markdown(
        """

        The models have been trained on following datasets:

        * dataset1_guided_fixations: [Alexander et al. 2017](https://doi.org/10.1038/sdata.2017.181)
        * dataset2_pursuit: [Nau et al. 2018](https://doi.org/10.1016/j.neuroimage.2018.04.012)
        * dataset3_pursuit: [Polti & Nau et al. 2022](https://elifesciences.org/articles/79027)
        * dataset4_pursuit: [Nau et al. 2018](https://doi.org/10.1038/s41593-017-0050-8)
        * dataset5_free_viewing: [Julian et al. 2018](https://doi.org/10.1038/s41593-017-0049-1)
        * dataset6: 4 Participants of [Frey & Nau et al. 2021](https://www.nature.com/articles/s41593-021-00947-w)
        * datasets_1to6: Datasets 1-6
        * datasets_1to5: Datasets 1-5 (recommended)
    """
    )

    # Show the uploaded file
    if uploaded_file:
        # Create folders
        create_folders()
        # Download weights
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "weights", model_str + ".h5")):
            with st.spinner(f"Downloading model weights {model_str}..."):
                download_model_weights(model_str)

        with st.spinner("Preprocess file..."):
            path_to_nifti = save_uploaded_file(uploaded_file)
        run_participant(path_to_nifti, model_str)
        # Clean folders
        clean_folders()


def run_participant(path_to_nifti, model_str):
    participant_string = os.path.splitext(os.path.basename(path_to_nifti))[0]
    # Preprocess nifti file
    (eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = deepmreye.preprocess.get_masks()
    with st.spinner(f"Transform {path_to_nifti} into MNI space..."):
        deepmreye.preprocess.run_participant(
            path_to_nifti,
            dme_template,
            eyemask_big,
            eyemask_small,
            x_edges,
            y_edges,
            z_edges,
            transforms=["Affine", "Affine", "SyNAggro"],
        )
    # Show preprocessed results
    html_results = os.path.join(os.path.dirname(path_to_nifti), "report_" + participant_string + ".html")
    show_html(html_results)

    # Save into npz file for easy loading
    this_mask = os.path.join(os.path.dirname(path_to_nifti), "mask_" + participant_string + ".p")
    this_mask = pickle.load(open(this_mask, "rb"))
    with st.spinner("Normalize Image..."):
        this_mask = deepmreye.preprocess.normalize_img(this_mask)

    # Add dummy labels
    this_label = np.zeros(
        (this_mask.shape[3], 10, 2)
    )  # 10 is the number of subTRs used in the pretrained weights, 2 is XY
    this_id = [participant_string] * this_label.shape[0], [0] * this_label.shape[0]
    deepmreye.preprocess.save_data(
        participant_string, [this_mask], [this_label], [this_id], os.path.dirname(path_to_nifti), center_labels=False
    )
    fn_participant = os.path.join(os.path.dirname(path_to_nifti), participant_string + ".npz")

    # Now load model weights
    with st.spinner("Loading model weights..."):
        model, generators = load_model(fn_participant, model_str)

    # Run through tensorflow model
    with st.spinner("Predicting gaze coordinates..."):
        (evaluation, _) = deepmreye.train.evaluate_model(
            dataset=participant_string,
            model=model,
            generators=generators,
            save=False,
            verbose=0,
            percentile_cut=80,
        )
        df_pred_median, df_pred_subtr = adapt_evaluation(evaluation[list(evaluation.keys())[0]])

    st.success(f"Inference done for {participant_string}")

    tab1, tab2 = st.tabs(["Gaze coordinates", "Sub-TR gaze coordinates"])
    with tab1:
        st.dataframe(df_pred_median.style.format("{:.4}"))
    with tab2:
        st.dataframe(df_pred_subtr.style.format("{:.4}"))

    # Offer results as download
    st.download_button(
        "Download gaze coordinates (one per TR)",
        convert_df(df_pred_median),
        f"predicted_gaze_median_{participant_string}.csv",
        "text/csv",
        key="download-median-csv",
    )
    st.download_button(
        "Download sub-TR gaze coordinates (10 per TR)",
        convert_df(df_pred_subtr),
        f"predicted_gaze_subTR_{participant_string}.csv",
        "text/csv",
        key="download-subTR-csv",
    )


# ------------------------------------------------------------------------------
# -------------------------HELPERS----------------------------------------------
# ------------------------------------------------------------------------------
def save_uploaded_file(uploadedfile):
    path_to_file = os.path.join(os.path.dirname(__file__), "tmp", uploadedfile.name)
    with open(path_to_file, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return path_to_file


def create_folders():
    weights_folder = os.path.join(os.path.dirname(__file__), "weights")
    tmp_folder = os.path.join(os.path.dirname(__file__), "tmp")
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)


def clean_folders():
    tmp_folder = os.path.join(os.path.dirname(__file__), "tmp")
    for file in os.listdir(tmp_folder):
        os.remove(os.path.join(tmp_folder, file))


@st.cache_data
def convert_df(df):
    return df.to_csv(index=True).encode("utf-8")


def adapt_evaluation(participant_evaluation):
    pred_y = participant_evaluation["pred_y"]
    pred_y_median = np.nanmedian(pred_y, axis=1)
    pred_uncertainty = abs(participant_evaluation["euc_pred"])
    pred_uncertainty_median = np.nanmedian(pred_uncertainty, axis=1)
    df_pred_median = pd.DataFrame(
        np.concatenate((pred_y_median, pred_uncertainty_median[..., np.newaxis]), axis=1),
        columns=["X", "Y", "Uncertainty"],
    )
    # With subTR
    subtr_values = np.concatenate((pred_y, pred_uncertainty[..., np.newaxis]), axis=2)
    index = pd.MultiIndex.from_product(
        [range(subtr_values.shape[0]), range(subtr_values.shape[1])], names=["TR", "subTR"]
    )
    df_pred_subtr = pd.DataFrame(
        subtr_values.reshape(-1, subtr_values.shape[-1]), index=index, columns=["X", "Y", "pred_error"]
    )

    return df_pred_median, df_pred_subtr


def download_model_weights(model_str):
    if model_str == "datasets_1to6":
        url = "https://osf.io/download/mr87v"
    elif model_str == "datasets_1to5":
        url = "https://osf.io/download/23t5v"
    elif model_str == "dataset1_guided_fixations":
        url = "https://osf.io/download/cqf74"
    elif model_str == "dataset2_pursuit":
        url = "https://osf.io/download/4f6m7"
    elif model_str == "dataset3_pursuit":
        url = "https://osf.io/download/e89wp"
    elif model_str == "dataset4_pursuit":
        url = "https://osf.io/download/96nyp"
    elif model_str == "dataset5_free_viewing":
        url = "https://osf.io/download/89nky"
    elif model_str == "dataset6_openclosed":
        url = "https://osf.io/download/8cr2j"
    # Download file with wget and store in weights folder
    weights_folder = os.path.join(os.path.dirname(__file__), "weights", model_str + ".h5")
    # Download file with requests and save in weights folder
    r = requests.get(url, allow_redirects=True)
    with open(weights_folder, "wb") as f:
        f.write(r.content)


# @st.cache
def load_model(fn_participant, model_str):
    opts = deepmreye.util.model_opts.get_opts()
    test_participants = [fn_participant]
    generators = deepmreye.util.data_generator.create_generators(test_participants, test_participants)
    generators = (*generators, test_participants, test_participants)  # Add participant list
    (model, model_inference) = deepmreye.train.train_model(
        dataset="example_data", generators=generators, opts=opts, return_untrained=True
    )
    model_weights = os.path.join(os.path.dirname(__file__), "weights", model_str + ".h5")
    model.load_weights(model_weights)

    return model_inference, generators


def show_html(html_file):
    with open(html_file, "r", encoding="utf-8") as f:
        source_code = f.read()
        components.html(source_code, scrolling=True, height=650)


if __name__ == "__main__":
    main()
