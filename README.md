# GPT-generative-text

## Introduction
Generative AI has emerged as a transformative force, reshaping the landscape of natural language processing and text generation. The advent of powerful models like OpenAI's GPT-3 and the fundamental insights presented in the seminal paper "Attention is All You Need" by Vaswani et al. have paved the way for innovative applications across various domains. This report delves into a specific facet of generative AI, focusing on a decoder-only transformer model trained on wikitext2 data. By harnessing the principles outlined in "Attention is All You Need," our project aims to develop an auto-regressive text generating model, peeking into the ongoing evolution of language models and their practical implications.
</br>
</br>
</br>

## Framework
- Deep Learning Framework: PyTorch
- Environment: Jupyter Notebook
- Neural Network Architecture: Transformer
</br>
</br>

## About the data - Wikitext2
The Wikitext-2 dataset is a valuable resource in the field of natural language processing (NLP), offering a comprehensive and diverse collection of text data. It consists of a subset of Wikipedia articles, specifically the second version of the Wikitext dump. The dataset is designed to be used for language modelling tasks, making it an essential benchmark for researchers and practitioners in the NLP community. Comprising a wide range of topics, genres, and writing styles, Wikitext-2 provides a rich and varied linguistic landscape for training and evaluating language models. Its content includes not only informative articles but also discussions, debates, and diverse forms of textual expression found on Wikipedia. The dataset's significance extends beyond simple language modelling, as it serves as a testbed for understanding contextual nuances, syntactic structures, and semantic intricacies in a real-world, dynamic linguistic environment. The dataset can be accessed from the torchtext.datasets library.
</br>
</br>

## Transformers and their advantage over RNN and LSTM
Transformers represent a paradigm shift in natural language processing (NLP) and language modelling, presenting distinct advantages over traditional recurrent neural networks (RNNs) and long short-term memory networks (LSTMs). Unlike RNNs and LSTMs, transformers leverage a self-attention mechanism, enabling them to capture long-range dependencies and contextual information more effectively. This attention mechanism allows transformers to process input sequences in parallel, making them highly scalable and efficient in handling sequential data. Additionally, transformers excel in capturing contextual information, as they consider the entire input sequence simultaneously rather than relying on sequential processing. This characteristic is particularly advantageous for language modelling tasks, where understanding context is crucial for generating coherent and contextually relevant text.


Moreover, transformers have demonstrated superior performance in handling variable-length sequences, mitigating the vanishing gradient problem that can hinder the training of RNNs and LSTMs. The self-attention mechanism in transformers enables them to assign different weights to different parts of the input sequence dynamically, allowing for more effective information retention and utilization.


The attention mechanism in transformers also facilitates better modelling of relationships between words and contextual nuances, resulting in improved language understanding and generation. This has led to the widespread adoption of transformer-based models, such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer), which have achieved state-of-the-art results in various NLP tasks.
</br>
</br>

## Decoder-only Transformer
A decoder-only transformer refers to a modified version of the original transformer architecture that consists solely of the decoder component. In the standard transformer architecture, there are two main components: the encoder and the decoder. The encoder processes the input sequence, extracting relevant features, while the decoder generates the output sequence based on those features.


In a decoder-only transformer, the encoder component is omitted, and only the decoder is retained. This architecture is commonly used in tasks where the model is required to generate sequences autoregressively, such as language generation or sequence-to-sequence tasks where the output is generated step by step.


The decoder-only transformer employs self-attention mechanisms to attend to different parts of the decoded sequence, allowing it to capture dependencies and relationships within the generated sequence. This architecture has been notably used in models like GPT-2 (Generative Pre-trained Transformer 2) and GPT-3, where the focus is on generating coherent and contextually rich text.

</br>
</br>

## Byte Pair Encoding
Tokenization and encoding methods are fundamental processes in natural language processing that involve breaking down raw text into manageable units and representing them in a format suitable for machine learning models. In the context of this project, I have employed the Byte Pair Encoding (BPE) method for tokenization. BPE is a data compression technique widely used in the field, operating by iteratively merging the most frequent pair of adjacent bytes or characters in a given corpus until a predefined vocabulary size is reached. The outcome is a sub-word vocabulary that adeptly handles rare or out-of-vocabulary words. BPE has gained popularity, particularly in the realm of neural network-based language models, for its ability to efficiently capture both common and rare linguistic patterns. By dynamically adapting to the data during the merging process, BPE-based tokenization significantly contributes to improved language representation, making it a valuable technique in modern natural language processing pipelines.
</br>
</br>

## Conclusion
In conclusion, we successfully developed an auto-regressive text generating model based on a decoder-only transformer trained on wikitext2 data. Despite the limitations of GPU resources from the free tier of online platforms, the model demonstrated the ability to approximate the English language. The cross-entropy loss was 2.16 on the test data and 1.99 on the training data. While acknowledging the room for improvement, this project highlights the effectiveness of transformer-based architectures in text generation tasks.


Check the code [here]()


Try the model! Generate some text with this [web application](http://indranuj.pythonanywhere.com/)
</br>
</br>

## References
1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, & Illia Polosukhin. (2023). Attention Is All You Need.
2. Andrej Karpathy - [NanoGPT](https://github.com/karpathy/nanoGPT)
