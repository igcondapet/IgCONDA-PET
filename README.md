# IgCONDA-PET: Implicitly-Guided Counterfactual Diffusion for Detecting Anomalies in PET Images


## Introduction
This codebase is related to our submission to MICCAI 2024:<br>
> Anonymous authors, _IgCONDA-PET: Implicitly-Guided Counterfactual Diffusion for Detecting Anomalies in PET Images_.

<p align="justify">
Minimizing the need for pixel-level annotated data for training PET anomaly segmentation networks is crucial, particularly due to time and cost constraints related to expert annotations. Current un-/weakly-supervised anomaly detection methods rely on autoencoder or generative adversarial networks trained only on healthy data, although these are more challenging to train. In this work, we present a weakly-supervised and <b>I</b>mplicitly <b>g</b>uided <b>CO</b>u<b>N</b>terfactual diffusion model for <b>D</b>etecting <b>A</b>nomalies in <b>PET</b> images, branded as <b>IgCONDA-PET</b>. The training is conditioned on image class labels (healthy vs. unhealthy) along with implicit guidance [3] to generate counterfactuals for an unhealthy image with anomalies. The counterfactual generation [4] process synthesizes the healthy counterpart for a given unhealthy image, and the difference between the two facilitates the identification of anomaly locations. In this work, we use two publicly available oncological FDG PET datasets, AutoPET [1] and HECKTOR [2].
</p>

<p align="center">
<img src="./assets/method_scheme.png" alt="Figure" height="300" />
</p>
<p align="justify">
    Figure 1: IgCONDA-PET: Implicitly-guided counterfactual DPM sampling methodology for domain translation between an unhealthy image and its healthy counterfactual for PET. The anomaly map is defined as the absolute difference between the unhealthy and corresponding reconstructed healthy image.
</p>

<p align="center">
  <img src="./assets/plot_comparing_igcondapet_to_other_methods.png" alt="Figure" height="350" />
</p>
<p align="justify">
    Figure 2: Qualitative comparison between the anomaly maps generated by different methods on PET slices. GT represents the physician’s dense ground truth. Here, we set the noise level <i>D</i> = 400 and <i>w</i> = 3.0.
</p>

## How to get started?
Follow the intructions given below to set up the necessary conda environment, install packages, preprocess dataset in the correct format so it can be accepted as inputs by the code, train model and perform anomaly detection on test set using the trained models. 

- **Clone the repository, create conda environment and install necessary packages:**  
    The first step is to clone this GitHub codebase in your local machine, create a conda environment, and install all the necessary packages. This code base was developed primarily using python=3.8.10, PyTorch=1.11.0, monai=1.3.0, with CUDA 11.3 on an Ubuntu 20.04 virtual machine, so the codebase has been tested only with these configurations. We hope that it will run in other suitable combinations of different versions of python, PyTorch, monai, and CUDA, but we cannot guarantee that. Proceed with caution!
    ```
    git clone 'https://github.com/igcondapet/IgCONDA-PET.git'
    cd IgCONDA-PET
    conda env create --file requirements.yaml
    ```
    The last step above creates a conda environment named `igcondapet`. Make sure you have conda installed. Next, activate the conda environment
    ```
    conda activate igcondapet
    ```


# References

<a id="1">[1]</a> 
Gatidis, S., Kuestner, T., "A whole-body FDG-PET/CT dataset with manually annotated tumor lesions (FDG-PET-CT-Lesions)" [Dataset] (2022), The Cancer Imaging Archive. 
[![DOI](./assets/autopet_data_zenodo.svg)](https://doi.org/10.7937/gkr0-xv29)

<a id="2">[2]</a> 
Andrearczyk, V., Oreiller, V., Boughdad, S., Rest, C.C.L., Elhalawani, H., Jreige, M., Prior, J.O., Valli`eres, M., Visvikis, D., Hatt, M., et al., "Overview of the hecktor challenge at miccai 2021: automatic head and neck tumor segmentation and outcome prediction in pet/ct images", In: 3D head and neck tumor segmentation in PET/CT challenge, pp. 1–37. Springer (2021). 
[(doi)](https://doi.org/10.1007/978-3-030-98253-9_1)

<a id="3">[3]</a> 
Ho, J., Salimans, T.: "Classifier-free diffusion guidance". arXiv preprint arXiv:2207.12598 (2022)

<a id="4">[4]</a> 
Sanchez, P., Kascenas, A., Liu, X., O’Neil, A.Q., Tsaftaris, S.A.: "What is healthy? generative counterfactual diffusion for lesion localization". In: MICCAI Workshop on Deep Generative Models. pp. 34–44. Springer (2022)