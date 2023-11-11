1)	Introduction:
RNN, LSTM, and Gated Recurrent Units represent state-of-the-art techniques for addressing the challenges of sequence modeling and traversal, especially in applications such as machine translation.


2)	Background:
The goal of minimizing sequential computation is achieved through models such as extended neural GPUs, all of which use CNNs as basic components to simultaneously compute hidden representations for all input locations and output. In these models, computational complexity increases as the distance between input or output locations increases, posing challenges for dependent learning. Transformer solves this problem by using self-attention, also known as internal attention, which ensures a constant amount of activity to link signals from any two locations. Self-attention has been shown to be effective in various tasks such as summarizing, text interaction, and the acquisition of task-independent sentence representations. Meanwhile, the front-end memory network relies on a recurrent attention mechanism.


3)	Model Architecture:
Most modern neural chain transmission models follow an encoder-decoder architecture.
 The encoder transforms a sequence of input symbolic representations into a continuous sequence of representations, while the decoder produces an output sequence of symbols, one element at a time.

      3.1 Encoder and Decoder:

Encoder: The encoder is made up of 6 identical layers. Each class has two subclasses.
 The first is a multi-head self-attention mechanism, while the second is simple, the output of each sublayer is which is the function implemented by the sublayer itself. All subclasses of the model, as well as the integration classes, produce outputs of size = 512.
Decoder: The decoder also consists of 6 identical layers, the decoder inserts a third sub-layer, which performs the attention to multiple inputs and outputs of the encoder stack. Like the encoder, we use residual connections around each sublayer followed by layer normalization.
   3.2 Attention:
Essentially, the attention function can be defined as a process that executes a query and a set of key-value pairs and produces an output. Calculating this result involves determining a weighted sum of the corresponding values, with weights assigned based on each key's relevance to the given query.
      3.2.1 Scaled Dot-Product Attention:
 We call our particular attention "Scaled Dot-Product Attention". The input consists of
queries and keys of dimension, and values of dimension. We compute the dot products of the Scaled Dot-Product Attention. 

      3.2.2 Multi-Head Attention:
It is beneficial to linearly project queries, keys, and values multiple times (h times) with separate linear transformations. These converters are then connected and subjected to another projection.
Using multi-head attention in a model allows it to simultaneously focus on information from different representational subspaces in different locations. This approach improves the model's ability to capture diverse and nuanced relationships in input data.

     3.3 Embeddings and SoftMax:
 In the paper, the learned embeddings play an important role in converting input tokens and output tokens into vectors. This involves using the standard learned linear transform and the SoftMax function to transform the decoder output. Notably, the model adopts the strategy of sharing the same weight matrix in both the embedding layer and the SoftMax pre-linear transformation.
This shared parameterization contributes to the overall consistency and efficiency of the model's representation and output generation process.

4)	Why Self-Attention:
 The paper performs a comprehensive comparison of self-attention layers with regression and convolutional layers commonly used to map variable-length sequences of symbolic representations.
An important aspect examined is the path length between long-range dependencies in the network, which constitutes an important challenge in many sequence conversion tasks. The ability to learn these extended dependencies is significantly affected by the length of the paths that forward and reverse signals must travel through the network. Shorter paths between any pair of positions in the input and output chains simplify learning long-range dependencies, emphasizing the importance of efficient signal propagation for effective chain modeling.

    5) Training:

  5.1 Training Data and Batching:

 In the model, the training process is performed on the standard English-German dataset WMT 2014, which includes about 4.5 million sentence pairs. The vocabulary in this dataset consists of approximately 37,000 tokens. For the English-French translation task, the significantly larger English-French dataset from WMT 2014 was used, containing 36 million sentences. Tokenization involves dividing tokens into a vocabulary of 32,000 words. Each training session consists of sets of sentence pairs, for a total of approximately 25,000 source tokens and 25,000 target tokens, thus providing a substantial and diverse pool of training data for the model.

   5.2 Hardware and Schedule:
The machine was used with 8 NVIDIA P100 GPUs. each training step took about 0.4 seconds. they 
trained the base models for a total of 100,000 steps or 12 hours. 

   5.3 Optimizer
The optimizer that used was Adam. the varied the learning rate over the course of training, according to a specific formula.


    6)  Results:

  6.1 Machine Translation:

 In the WMT 2014 English-to-German translation task, the large transformer model demonstrated superior performance compared to previously reported results. It has surpassed the most famous benchmarks in this task. Meanwhile, in the English-to-French translation task of the same dataset, the model achieved an impressive BLEU score of 41.0.

 For the baseline models, a single model was used that averaged the last 5 test scores, recorded at 10-minute intervals during training. In the inference phase, a beam search with a beam size of 4 and a penalty length of 0.6 was used.

 To estimate the computational effort involved in training the model, the authors calculated the number of floating-point operations by multiplying the training time, the number of GPUs used, and the precision estimate Unique determination of each GPU's maintained floating-point capacity. This approach provides a quantitative measure of the computational resources required to train the model.

