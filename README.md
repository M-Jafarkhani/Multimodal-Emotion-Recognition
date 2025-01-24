# Multimodal Emotion Recognition
This repository explores a range of architectures for Multimodal Emotion Recognition (MER), emphasizing the integration of multiple modalities (text, audio, video) to improve sentiment analysis. Each architecture offers unique strengths and trade-offs concerning accuracy, efficiency, and resilience to challenges like misaligned or missing modalities. Transformer-based methods consistently demonstrate the highest effectiveness for MSA tasks.

# Datasets
### [CMU-MOSI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
The **MOSI (Multimodal Opinion Sentiment and Intensity)** dataset is a widely used benchmark in multimodal sentiment and emotion analysis. It consists of short video clips where speakers express their opinions and emotions, combining three modalities: **text** (transcriptions of spoken words), **audio** (vocal tone and pitch), and **visual** (facial expressions). Each segment is annotated with sentiment intensity scores ranging from -3 (strongly negative) to +3 (strongly positive).

MOSI is commonly used for tasks like multimodal fusion, sentiment prediction, and emotion recognition, making it a crucial dataset for advancing research in human-computer interaction and affective computing.

### [CMU-MOSEI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
The **MOSEI (Multimodal Sentiment Analysis and Emotion Intensity)** dataset is an extension of MOSI, designed to be larger and more diverse. It contains over 23,000 video clips from more than 1,000 speakers, covering various topics and languages. Each clip provides annotations for **sentiment** (ranging from -3 to +3, like MOSI) and **emotion intensity** across six primary emotions: happiness, sadness, anger, fear, surprise, and disgust.

MOSEI is also multimodal, combining **text**, **audio**, and **visual** data, making it suitable for tasks like sentiment analysis, emotion recognition, and multimodal fusion. Its scale and diversity make it a key resource for advancing multimodal natural language processing and understanding real-world affective expressions.

# Architectures and Obeservations
-   **Early Fusion**:
    -   Combines features from all modalities right after feature extraction.
    -   Utilizing Gated Recurrent Units (GRU) and Transformers for improved sequential data processing.
    -   Achieved moderate to good performance, with Transformers outperforming GRUs.

-   **Late Fusion**:
    -   Processes each modality independently until the decision stage, where outputs are combined.
    -   Similar architecture as Early Fusion but delayed integration led to slightly improved performance for some models.

-   **Tensor Fusion**:
    -   Employs Tensor Fusion to capture intra- and inter-modal interactions.
    -   Achieved relatively the same performance as Early and Late Fusion techniques.

-   **Low-Rank Tensor Fusion**:
    -   A more efficient variant of Tensor Fusion that projects features into a low-rank tensor space.
    -   Much more efficient than Tensor Fusion, with the same accuracy.

-   **Multimodal Factorization Model**:
    -   Separates representations into shared multimodal factors and modality-specific generative factors.
    -   Incorporates modality-specific decoders to reconstruct inputs.
    -   Suffered from overfitting, leading to a discrepancy between training and test accuracies.

-   **Multimodal Cyclic Translation Network**:
    -   Uses cyclic translation between modalities to create robust joint representations.
    -   Captures shared and complementary information across modalities effectively.
    -   The most parameter-efficient model
    -   Acceptable accurarcy on MOSI but performed poorly on MOSEI

-   **Multimodal Transformer (MulT)**:
    -   Utilizes a crossmodal attention mechanism to dynamically fuse information across time steps.
    -   Handles misalignments between modalities efficiently.
    -   Demonstrated good performance among the architectures tested.
	
# Experiments Result

## Accuracy (%) 
| Architecture                          | CMU-MOSI    | CMU-MOSEI   |
|---------------------------------------|-------------|-------------|
| Early Fusion (Transformer)            | [75.65][L01]| [71.91][L02]|
| Late Fusion (GRU)                     | [75.21][L03]| [71.60][L04]|
| Multimodal Transformer                | [75.21][L05]| [70.40][L06]|
| Late Fusion (Transformer)             | [73.32][L07]| [68.49][L08]|
| Multimodal Cyclic Translation Network | [72.44][L09]| [59.49][L10]|
| Tensor Fusion                         | [72.30][L11]| [70.45][L12]|
| Low Rank Tensor Fusion                | [72.01][L13]| [70.95][L14]|
| Unimodal                              | [71.28][L15]| [70.01][L16]|
| Early Fusion (GRU)                    | [66.90][L17]| [49.01][L18]|
| Multimodal Factorization              | [63.70][L19]| [56.88][L20]|

## Inefernce Params
| Architecture                          | Parameters (Million)|
|---------------------------------------|---------------------|
| Early Fusion (Transformer)            | [~8.1][L01]         |
| Late Fusion (GRU)                     | [~2.5][L03]         |
| Multimodal Transformer                | [~3.0][L05]         |
| Late Fusion (Transformer)             | [~20 ][L07]         |
| Multimodal Cyclic Translation Network | [~0.2][L09]         |
| Tensor Fusion                         | [~5.4][L11]         |
| Low Rank Tensor Fusion                | [~1.5][L13]         |
| Unimodal                              | [~1.9][L15]         |
| Early Fusion (GRU)                    | [~1.6][L17]         |
| Multimodal Factorization              | [~1.4][L19]         |

# Setup
All experiments were carried out on [Google Colab Pro](https://colab.research.google.com/), utilizing a A100/T4 GPU with High RAM.

# References
 - [Codebase Repository (MultiBench)](https://github.com/Klodivio355/MultiBench)
 - [MultiModal Transformer](https://github.com/yaohungt/Multimodal-Transformer) 
 - [Multimodal Cyclic Translations](https://arxiv.org/pdf/1812.07809.pdf)
 - [Tensor Fusion](https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py)
 - [Multimodal Factorization](https://arxiv.org/pdf/1806.06176)
 - [Low Rank Tensor Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)


[L01]: src/notebooks/MOSI/Early_Fusion_Transformer.ipynb
[L02]: src/notebooks/MOSEI/Early_Fusion_Transformer.ipynb
[L03]: src/notebooks/MOSI/Late_Fusion.ipynb
[L04]: src/notebooks/MOSEI/Late_Fusion.ipynb
[L05]: src/notebooks/MOSI/Multimodal_Transformer.ipynb
[L06]: src/notebooks/MOSEI/Multimodal_Transformer.ipynb
[L07]: src/notebooks/MOSI/Late_Fusion_Transformer.ipynb
[L08]: src/notebooks/MOSEI/Late_Fusion_Transformer.ipynb
[L09]: src/notebooks/MOSI/Multimodal_Cyclic_Translation_Network.ipynb
[L10]: src/notebooks/MOSEI/Multimodal_Cyclic_Translation_Network.ipynb
[L11]: src/notebooks/MOSI/Tensor_Fusion.ipynb
[L12]: src/notebooks/MOSEI/Tensor_Fusion.ipynb
[L13]: src/notebooks/MOSI/Low_Rank_Tensor_Fusion.ipynb
[L14]: src/notebooks/MOSEI/Low_Rank_Tensor_Fusion.ipynb
[L15]: src/notebooks/MOSI/Unimodal.ipynb
[L16]: src/notebooks/MOSEI/Early_Fusion.ipynb
[L17]: src/notebooks/MOSI/Early_Fusion.ipynb
[L18]: src/notebooks/MOSEI/Early_Fusion.ipynb
[L19]: src/notebooks/MOSI/Multimodal_Factorization.ipynb
[L20]: src/notebooks/MOSEI/Multimodal_Factorization.ipynb