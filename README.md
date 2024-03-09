# IgCONDA-PET: Implicitly-Guided Counterfactual Diffusion for Detecting Anomalies in PET Images


## Introduction

<p align="center">
  <img src="./assets/method_scheme.png" alt="Figure" height="300" />
  <br>
  <em>
  <strong>
    Figure 1: IgCONDA-PET: Implicitly-guided counterfactual DPM sampling methodology for domain translation between an unhealthy image and its healthy counterfactual for PET. The anomaly map is defined as the absolute difference between the unhealthy and corresponding reconstructed healthy image.
   </strong>
  </em>
</p>


<p align="center">
  <img src="./assets/plot_comparing_igcondapet_to_other_methods.png" alt="Figure" height="300" />
  <br>
    Figure 2: Qualitative comparison between the anomaly maps generated by different methods on PET slices. GT represents the physician’s dense ground truth. Here, we set the noise level D = 400 and w = 3.0.
</p>