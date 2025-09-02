# TransBindpMHC-V1（中文版）

- [TransBindpMHC-V1](#transbindpmhc-v1)
  - [简介](#简介)
    - [架构](#架构)
  - [克隆项目](#克隆项目)
  - [安装](#安装)
  - [使用方法](#使用方法)
    - [云端可视化平台](#云端可视化平台)
    - [本地部署](#本地部署)
  - [可视化分析](#可视化分析)
    - [模型对不同多肽长度的预测偏好](#模型对不同多肽长度的预测偏好)
    - [模型对不同 MHC 类型的预测偏好](#模型对不同-mhc-类型的预测偏好)
  - [注意事项](#注意事项)
  - [参考文献](#参考文献)
  - [引用](#引用)

🔙 **Back to English**: [README.md](README.md)

---

## 简介

癌症是导致人类死亡的第二大疾病，尽管医学发展迅速，但传统治疗方法对晚期及转移性肿瘤疗效有限。近年来，肿瘤免疫治疗如嵌合抗原受体T细胞疗法（CAR-T）和免疫检查点抑制剂等展现出良好的疗效，成为新的研究热点。其中，肿瘤新抗原（neoantigen）被认为是免疫治疗的理想靶标，如何精准鉴定新抗原成为一大挑战。

准确预测主要组织相容性复合物I类（MHC-I）与多肽的结合能力，是提高新抗原鉴定效率的关键。相比亲和力实验，质谱技术鉴定的 MHC 结合多肽能更全面地反映细胞内自然呈递的抗原信息。此外，与传统实验方法相比，基于深度学习的计算预测方法可更快速、准确地完成该任务。

本研究基于Transformer模型，利用质谱来源的数据进行建模，构建了一种新型多肽-MHC结合预测方法——**TransBindpMHC**。我们对其性能进行了全面评估：在训练中未见过的稀有MHC类型泛化数据集上，模型准确率达到 **0.9055**，初步验证了其泛化能力；在多批次、分布差异大的测试集以及大规模、多MHC类型的评估数据集中，TransBindpMHC在多种指标上表现优于或媲美六种主流算法；在实验验证的HPV及肿瘤新抗原数据集上，预测精度同样优异，尤其在新抗原数据集上的准确率高达 **0.932**。

为进一步探究模型在不同条件下的适用性，我们分析了其在不同多肽长度和MHC类型上的预测偏好，结果显示该方法在各种条件下均具备较强的预测能力。同时，为方便用户零代码使用，我们提供了一个可直接访问的在线平台：[https://xuhu-transbindpmhc.streamlit.app/](https://xuhu-transbindpmhc.streamlit.app/)

综上所述，TransBindpMHC是一种泛MHC类型、支持多物种（目前支持人、小鼠）、适用于8–15个氨基酸长度的多肽序列、能够良好表征大多数中国人常见HLA分子类型的pMHC结合预测新方法。在多项评估任务中均表现出优异的预测性能与稳定性，具备良好的临床应用前景，为精准肿瘤免疫治疗提供了有力的计算工具。

#### 架构

<p align="center">
  <img src="images/framework.jpeg" alt="TransBindpMHC_Framework" width="600"/>
</p>

<p align="center">
  <img src="images/structure.png" alt="TransBindpMHC_structure" width="600"/>
</p>

## 克隆项目

```bash
git clone https://github.com/xuhu0115/TransBindpMHC-V1
cd TransBindpMHC-V1
```

## 安装

- 使用 pip 安装：
```bash
pip install -r requirements.txt
```

- 使用 conda 安装：
```bash
conda env create -f environment.yml
```

## 使用方法

#### 云端可视化平台

🔗 [TransBindpMHC-V1 在线平台](https://xuhu-transbindpmhc.streamlit.app/)

#### 本地部署

请准备如下格式的数据，并运行预测脚本。如需修改参数或指定输出路径，请调整 `predict.py` 的主函数。

- **输入两个 FASTA 文件**

  - 多肽序列文件（FASTA 格式）：
    ```fasta
    >Peptide_1
    AEAFIQPI
    >Peptide_2
    KILRGVAK
    >Peptide_3
    MVWIQLGL
    ```

  - MHC 类型文件（FASTA 格式）：
    ```fasta
    >HLA-1
    HLA-A*02:01
    >HLA-2
    HLA-B*07:02
    >HLA-3
    HLA-C*04:01
    ```

  - 预期输出：
    ```
    HLA         HLA_sequence  peptide    y_pred  y_prob
    HLA-A*02:01 ASNENM...ETM     AEAFIQPI   1       0.9213
    HLA-A*02:01 ASNENM...ETM     KILRGVAK   0       0.2541
    ...
    ```

- **直接输入多肽序列和MHC类型**

  - 多肽序列：`AEAFIQPI`
  - MHC 类型：`HLA-A*11:01`

  - 预期输出：
    ```
    HLA         HLA_sequence  peptide    y_pred  y_prob
    HLA-A*11:01 ASNENM...ETM  AEAFIQPI     1     0.9213
    ```

## 可视化分析

#### 模型对不同多肽长度的预测偏好

请查看并运行：  
`方法的批量评估/5_模型对不同肽长度的预测偏好.ipynb`

#### 模型对不同 MHC 类型的预测偏好

请查看并运行：  
`方法的批量评估/5_模型对不同MHC的预测偏好.ipynb`

## 注意事项

## 参考文献

## 引用
