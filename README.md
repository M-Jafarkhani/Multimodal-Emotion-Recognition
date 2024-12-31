# Multimodal Sentiment Analysis
In this repository, we explore multiple architectures on multiple datasets to recognize sentiment and emotions from multimodal data, including text, images and audios. We examined different architectures on diffrenet datasets.

# Datasets
## [CMU-MOSI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
The **MOSI (Multimodal Opinion Sentiment and Intensity)** dataset is a widely used benchmark in multimodal sentiment analysis. It consists of short video clips where speakers express their opinions and emotions, combining three modalities: **text** (transcriptions of spoken words), **audio** (vocal tone and pitch), and **visual** (facial expressions). Each segment is annotated with sentiment intensity scores ranging from -3 (strongly negative) to +3 (strongly positive).

MOSI is commonly used for tasks like multimodal fusion, sentiment prediction, and emotion recognition, making it a crucial dataset for advancing research in human-computer interaction and affective computing.

## [CMU-MOSEI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
The **MOSEI (Multimodal Sentiment Analysis and Emotion Intensity)** dataset is an extension of MOSI, designed to be larger and more diverse. It contains over 23,000 video clips from more than 1,000 speakers, covering various topics and languages. Each clip provides annotations for **sentiment** (ranging from -3 to +3, like MOSI) and **emotion intensity** across six primary emotions: happiness, sadness, anger, fear, surprise, and disgust.

MOSEI is multimodal, combining **text**, **audio**, and **visual** data, making it suitable for tasks like sentiment analysis, emotion recognition, and multimodal fusion. Its scale and diversity make it a key resource for advancing multimodal natural language processing and understanding real-world affective expressions.

## [MELD Dataset](https://github.com/declare-lab/MELD/blob/master/README.md)
The **MELD (Multimodal EmotionLines Dataset)** is a dataset created for multimodal emotion recognition and sentiment analysis. It contains over **13,000 utterances** from more than **1,400 dialogues** sourced from the TV show _Friends_. Each utterance is annotated with one of seven emotion categories: **anger, disgust, fear, joy, neutral, sadness, and surprise**, along with sentiment labels (**positive, negative, neutral**).

MELD provides data in three modalities:

-   **Text**: Transcripts of utterances.
-   **Audio**: Speech recordings.
-   **Video**: Facial expressions and body gestures.

It is widely used for tasks involving **contextual emotion understanding**, as the conversations include both isolated utterances and contextual dialogue flow, making it a rich resource for emotion recognition in multimodal, conversational settings.

## Architectures
### Early Fusion (With Transformers)