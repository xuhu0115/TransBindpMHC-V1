# TransBindpMHC-V1

- [TransBindpMHC-V1](#transbindpmhc-v1)
  - [Introduction](#introduction)
    - [Architecture](#architecture)
  - [Clone the Project](#clone-the-project)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Cloud Visualization Platform](#cloud-visualization-platform)
    - [Local Deployment](#local-deployment)
  - [Visualization](#visualization)
    - [Model Preference for Peptide Lengths](#model-preference-for-peptide-lengths)
    - [Model Preference for MHC Types](#model-preference-for-mhc-types)
  - [Notes](#notes)
  - [References](#references)
  - [Cite](#cite)

🌐 **Read in other languages**: [中文版 README_zh.md](README_zh.md)

---

## Introduction

Cancer is the second leading cause of human death. Despite rapid medical advances, traditional therapies have limited efficacy against advanced and metastatic tumors. In recent years, cancer immunotherapies such as chimeric antigen receptor T-cell (CAR-T) therapy and immune checkpoint inhibitors have shown promising results, becoming a major research focus. Among these, tumor neoantigens are considered ideal targets for immunotherapy, and accurately identifying them remains a significant challenge.

Accurate prediction of peptide binding to Major Histocompatibility Complex class I (MHC-I) is crucial for improving neoantigen identification efficiency. MHC-bound peptides identified by mass spectrometry (MS) provide richer information about naturally presented antigens compared to affinity-based assays. Moreover, deep learning-based computational methods offer faster and more accurate predictions than traditional experimental approaches.

In this study, we developed **TransBindpMHC**, a novel peptide-MHC binding prediction method based on the Transformer architecture, trained on MS-derived data. We comprehensively evaluated TransBindpMHC's performance. Results show that on generalization datasets with rare or unseen MHC alleles, TransBindpMHC achieves an accuracy of **0.9055**, demonstrating strong generalization ability. On diverse benchmark datasets — including multi-batch, distribution-shifted, and large-scale multi-MHC datasets — TransBindpMHC performs comparably or better than six state-of-the-art algorithms across multiple metrics. Furthermore, on experimentally validated HPV and tumor neoantigen datasets, TransBindpMHC achieves excellent predictive accuracy, with an accuracy of **0.932** on the neoantigen dataset.

We also analyzed TransBindpMHC's prediction preferences across different peptide lengths and MHC types, showing robust performance across conditions. To enable code-free access, we provide a publicly available web server: [https://xuhu-transbindpmhc.streamlit.app/](https://xuhu-transbindpmhc.streamlit.app/)

In summary, TransBindpMHC is a pan-MHC, cross-species (currently human and mouse), peptide-length flexible (8–15 amino acids) method capable of accurately modeling most common HLA types in the Chinese population. It demonstrates superior performance and stability across multiple evaluation tasks and shows promising potential for clinical applications in precision cancer immunotherapy.

#### Architecture

<p align="center">
  <img src="images/framework.jpeg" alt="TransBindpMHC_Framework" width="600"/>
</p>

<p align="center">
  <img src="images/structure.png" alt="TransBindpMHC_structure" width="600"/>
</p>

## Clone the Project

```bash
git clone https://github.com/xuhu0115/TransBindpMHC-V1
cd TransBindpMHC-V1
```

## Installation

- Install via pip:
```bash
pip install -r requirements.txt
```

- Install via conda:
```bash
conda env create -f environment.yml
```

## Usage

#### Cloud Visualization Platform

🔗 [TransBindpMHC-V1 Web App](https://xuhu-transbindpmhc.streamlit.app/)

#### Local Deployment

Prepare the input data in the formats described below, then run the prediction script from the command line. Modify the main function of `predict.py` to adjust parameters or specify output paths.

- **Input: Two FASTA files**

  - Peptide sequences (FASTA format):
    ```fasta
    >Peptide_1
    AEAFIQPI
    >Peptide_2
    KILRGVAK
    >Peptide_3
    MVWIQLGL
    ```

  - MHC alleles (FASTA format):
    ```fasta
    >HLA-1
    HLA-A*02:01
    >HLA-2
    HLA-B*07:02
    >HLA-3
    HLA-C*04:01
    ```

  - Expected output:
    ```
    HLA         HLA_sequence  peptide    y_pred  y_prob
    HLA-A*02:01 ASNENM...ETM     AEAFIQPI   1       0.9213
    HLA-A*02:01 ASNENM...ETM     KILRGVAK   0       0.2541
    ...
    ```

- **Direct input of peptide and MHC allele**

  - Peptide: `AEAFIQPI`
  - MHC: `HLA-A*11:01`

  - Expected output:
    ```
    HLA         HLA_sequence  peptide    y_pred  y_prob
    HLA-A*11:01 ASNENM...ETM  AEAFIQPI     1     0.9213
    ```

## Visualization

#### Model Preference for Peptide Lengths

See usage instructions in:  
`方法的批量评估/5_模型对不同肽长度的预测偏好.ipynb`

#### Model Preference for MHC Types

See usage instructions in:  
`方法的批量评估/5_模型对不同MHC的预测偏好.ipynb`

## Notes

## References

## Cite
