# Multimodal Sentiment Analysis
This repository explores a range of architectures for Multimodal Sentiment Analysis (MSA), emphasizing the integration of multiple modalities (text, audio, video) to improve sentiment analysis. Each architecture offers unique strengths and trade-offs concerning accuracy, efficiency, and resilience to challenges like misaligned or missing modalities. Transformer-based methods consistently demonstrate the highest effectiveness for MSA tasks.

# Datasets
### [CMU-MOSI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
The **MOSI (Multimodal Opinion Sentiment and Intensity)** dataset is a widely used benchmark in multimodal sentiment analysis. It consists of short video clips where speakers express their opinions and emotions, combining three modalities: **text** (transcriptions of spoken words), **audio** (vocal tone and pitch), and **visual** (facial expressions). Each segment is annotated with sentiment intensity scores ranging from -3 (strongly negative) to +3 (strongly positive).

MOSI is commonly used for tasks like multimodal fusion, sentiment prediction, and emotion recognition, making it a crucial dataset for advancing research in human-computer interaction and affective computing.

### [CMU-MOSEI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
The **MOSEI (Multimodal Sentiment Analysis and Emotion Intensity)** dataset is an extension of MOSI, designed to be larger and more diverse. It contains over 23,000 video clips from more than 1,000 speakers, covering various topics and languages. Each clip provides annotations for **sentiment** (ranging from -3 to +3, like MOSI) and **emotion intensity** across six primary emotions: happiness, sadness, anger, fear, surprise, and disgust.

MOSEI is multimodal, combining **text**, **audio**, and **visual** data, making it suitable for tasks like sentiment analysis, emotion recognition, and multimodal fusion. Its scale and diversity make it a key resource for advancing multimodal natural language processing and understanding real-world affective expressions.

### [MELD Dataset](https://github.com/declare-lab/MELD/blob/master/README.md)
The **MELD (Multimodal EmotionLines Dataset)** is a dataset created for multimodal emotion recognition and sentiment analysis. It contains over **13,000 utterances** from more than **1,400 dialogues** sourced from the TV show _Friends_. Each utterance is annotated with one of seven emotion categories: **anger, disgust, fear, joy, neutral, sadness, and surprise**, along with sentiment labels (**positive, negative, neutral**).

MELD provides data in three modalities:

-   **Text**: Transcripts of utterances.
-   **Audio**: Speech recordings.
-   **Video**: Facial expressions and body gestures.

It is widely used for tasks involving **contextual emotion understanding**, as the conversations include both isolated utterances and contextual dialogue flow, making it a rich resource for emotion recognition in multimodal, conversational settings.

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
    -   Achieved significant performance improvement over Early and Late Fusion techniques.
-   **Low-Rank Tensor Fusion**:
    
    -   A more efficient variant of Tensor Fusion that projects features into a low-rank tensor space.
    -   Sacrificed some accuracy for computational efficiency.
-   **Multimodal Factorization Model (MFM)**:
    
    -   Separates representations into shared multimodal factors and modality-specific generative factors.
    -   Incorporates modality-specific decoders to reconstruct inputs.
    -   Suffered from overfitting, leading to a discrepancy between training and test accuracies.
-   **Multimodal Cyclic Translation Network (MCTN)**:
    
    -   Uses cyclic translation between modalities to create robust joint representations.
    -   Captures shared and complementary information across modalities effectively.
-   **Multimodal Transformer (MulT)**:
    
    -   Utilizes a crossmodal attention mechanism to dynamically fuse information across time steps.
    -   Handles misalignments between modalities efficiently.
    -   Demonstrated state-of-the-art performance among the architectures tested.
 
 - **[HiTrans](https://aclanthology.org/2020.coling-main.370.pdf)**:
 
	HiTrans is a hierarchical transformer-based model for Emotion Detection in Conversations. 
	-   **Hierarchical Design**: Combines a low-level BERT transformer for local utterance representations and a high-level transformer for capturing global context across conversations.
	-   **Global Context Encoding**: Integrates long-distance dependencies between utterances with positional embeddings in the high-level transformer.
	-   **Emotion Detection Module**: Uses a Multi-Layer Perceptron (MLP) to classify emotions for each utterance based on contextualized representations.
	-   **Speaker Sensitivity**: Employs a Pairwise Utterance Speaker Verification (PUSV) task with a biaffine classifier to determine speaker relationships and enhance emotion detection.
	-   **Multi-Task Learning**: Trains jointly on emotion detection and speaker verification tasks with dynamically weighted loss to improve overall model performance
	
# Experiments Result

## Accuracy (%) 
| Architecture                             | CMU-MOSI | CMU-MOSEI |
|------------------------------------------|----------|-----------|
| Early Fusion (Transformer)               | [75.65](src/notebooks/MOSI/Early_Fusion.ipynb)    | [69.63]()     |
| Late Fusion                              | [75.21]()    | [71.60]()     |
| Multimodal Transformer                   | [75.21]()    | [70.40]()     |
| Late Fusion (Transformer)                | [73.32]()    | [68.49]()     |
| Multimodal Cyclic Translation Network    | [72.44]()    | -     |
| Tensor Fusion                            | [72.30]()    | [70.45]()     |
| Low Rank Tensor Fusion                   | [72.01]()    | [70.44]()     |
| Unimodal                                 | [71.28]()    | [70.01]()     |
| Early Fusion                             | [66.90]()    | [49.01]()     |
| Multimodal Factorization                 | [63.70]()    | -     |

## MELD
| Architecture                             | F1 Score (%) |
|------------------------------------------|--------------|
| HiTrans                   	 	       | 55.81        |

# References
 - [Baseline Repository](https://github.com/rugvedmhatre/Multimodal-Sentiment-Analysis)
 - [MultiModal Transformer](https://github.com/yaohungt/Multimodal-Transformer) 
 - [HiTrans](https://github.com/ljynlp/HiTrans)
 - [GRU](https://arxiv.org/pdf/1812.07809.pdf)
 - [Tensor Fusion](https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py)
 - [Low Rank Tensor Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)
