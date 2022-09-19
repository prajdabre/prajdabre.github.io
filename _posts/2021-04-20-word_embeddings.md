---
layout: distill
title: Word Embeddings
description: This blog goes through the two popular techniques used for embeddings i.e. Word2Vec and GloVe in detail understanding the math along with the code implementation in PyTorch.
date: 2021-04-20
tags: [python, pytorch, word-embeddings, word2vec, glove]

authors:
  - name: Jay Gala
    url: https://jaygala24.github.io
    affiliations:
      name: University of Mumbai

bibliography: 2021-04-20-word_embeddings.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Word2Vec
    subsections:
    - name: Overview
    - name: Skip-Gram Model
    - name: Negative Sampling
    - name: Subsampling Frequent Words
    - name: Skip-Gram Implementation
  - name: GloVe
    subsections:
    - name: Overview
    - name: Co-occurrence matrix
    - name: Mathematics
    - name: GloVe Implementation
  - name: References


# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  @media (min-width: 576px) {
    .output-plot img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
  }
  .citations {
    display: none;
  }
---

Humans use language as a way of communication for exchanging ideas and knowledge with others. Words are an integral part of the language that represents the [denotational semantics](https://en.wikipedia.org/wiki/Denotational_semantics), i.e., the meaning of a word. Humans are good at processing and understanding the idea that the words convey. Hence, we can share information about an image or an incident using a short string, assuming that we have some previous context. For example, a single word, “traffic,” conveys the information equivalent to a picture representing several vehicles stuck in a jam as shown in the figure below. However, computers are not good at understanding the ideas from the words, and we need to way to encode these ideas which computers can understand, i.e., in the form of numbers. These encoded representation are helpful for solving complex NLP tasks and are known as word embeddings (word vectors of a certain dimensions representing the idea of the word).
This encoding is known as word embeddings (word vectors) which can help us use these representations for solving the complex NLP tasks.

{% include figure.html path="assets/img/word_embeddings/traffic_jam_pic.jpeg" class="img-fluid rounded" zoomable=true %}

<div class="caption">
    Picture of a traffic jam (<a href="https://bit.ly/3aq0qL9">image source</a>)
</div>


## Word2Vec

### Overview

Previously, neural language models involved the first stage as learning a distributed representation for words and using these representations in the later stages for obtaining prediction. The main idea of the word2vec is also based on these neural language models where we use the hidden layer of a neural network to learn continuous representations of words which we call embeddings. Here, we discard the output layer of a trained neural network. Word2Vec model presents two algorithms:
- Continuous Bag of Words (CBOW): Predict center word based on the context words.
- Skip Gram: Predict the context words based on the center word.

The figure below illustrates the two algorithms:

{% include figure.html path="assets/img/word_embeddings/word2vec_algorithms.jpeg" class="img-fluid rounded" zoomable=true %}

<div class="caption">
    Word2Vec Model Architectures (<a href="https://arxiv.org/abs/1301.3781">image source</a>)
</div>


### Skip-Gram Model

From now onwards, we will look into the skip-gram model for word embeddings. The task is formulated as predicting the context words within a fixed window of size m given the center word. The visual illustration of above idea is shown below.

{% include figure.html path="assets/img/word_embeddings/skip_gram_overview.jpeg" class="img-fluid rounded" zoomable=true %}

<div class="caption">
    Skip-Gram Model Overview (<a href="https://stanford.io/32wNQWe">image source</a>)
</div>

Word pairs from the large corpus of text for a fixed window size $m$ are used for training the neural network. These word pairs are formed by looking at a fixed window size $m$ before and after the center word. This window size is a hyperparameter that you can play around with, but the authors found that window size 5 seems to work well in general. Having a smaller window size means that you are capturing minimal context. On the other hand, if your window size is too large, you are capturing too much context, which might not help you obtain specific meanings. For the above example with window size 2, the training pairs would be the following.

```
    [(into, problems), (into, turning), (into, banking), (into, crises)]
```

Since every word is represented by a vector, so the objective is to iteratively maximize the probability of context words $o$ given the center word $c$ for each position $t$ and adjust the word vectors.

$$
L(\theta) = \prod_{t=1}^{T}\prod_{\substack{-m \leq j \leq m \\ j \neq 0}} P(w_{t+j}\ |\ w_t\ ;\ \theta)
$$

where $L$ is the likelihood estimation and $\theta$ is the vector representation to be optimized.

In order to avoid the floating-point overflow and simple gradient calculation, we take the apply logarithm to the above likelihood estimation. The cost function is given as follows:

$$
J(\theta) = - \frac{1}{T}\ log\ L(\theta) = - \frac{1}{T}\ \prod_{t=1}^{T}\prod_{\substack{-m \leq j \leq m \\ j \neq 0}} P(w_{t+j}\ |\ w_t\ ;\ \theta)
$$

Here we are minimizing the cost function, which means that we are maximizing the likelihood estimation, i.e., predictive accuracy.

Likelihood estimation for a context word $o$ given the center word $c$ is as follows:

$$
P(o\ |\ c) = \frac{exp(u_o^T v_c)}{\sum_{w \in V}\ exp(u_w^T v_c)}
$$

where
- $v_w$ and $u_w$ are the center word and context word vector representations
- $u_o^T v_c$ represents the dot product which is used as a similarity measure between context word $o$ and center word $c$
- $V$ represents the vocabulary

In order to express this similarity measure in terms of probability, we normalize over the entire vocabulary (the idea of using softmax) and $exp$ is used to quantify the dot product to a positive value.

Computing the normalizing factor for every word is too much expensive, which is why the authors came up with some tricks which reduce the computational cost and speed up the training.


### Negative Sampling

The main idea of the negative sampling is to differentiate data from noise, i.e., train a binary logistic regression for classifying a true pair (center word and context word) against several noise pairs (center word and random word). So now our problem is reduced to $K + 1$ labels classification instead of $V$ words ($K \ll V$), which means that weights will only be updated for $K + 1$ words whereas weights for all the words were updated. In general, we choose 5 negative words other than the context window around the center word ($K = 5$). We want the context words to have a higher probability than the sampled negative words.
 
The new objective function (cost function) is given as follows:
 
$$
J_{neg-sample}(\theta) = -\ log\ (\sigma(u_o^T v_c)) - \sum_{k=1}^{K}\ log\ (\sigma(-u_k^T v_c))
$$
 
where
- $\sigma$ represents sigmoid
- first term represents the estimation for true pair
- second term represents the estimation for negative samples
 
The authors found that the unigram distribution $U(w)^{3/4}$ works well than the other unigram and uniform distribution choices for sampling noise. The intuition is that raised to $3/4$ factor brings down the probability for more frequent words.
 
$$
P_n(w) = \frac{U(w)^{3/4}}{Z}
$$
 
where $Z$ is the normalization term.


### Subsampling Frequent Words

Word2vec has been trained on a very large corpus of text in which frequently occurring words do not contribute significantly to the meaning of a word. Common function words such as "the", "as", "a" provide structure to the sentence but don’t help in learning good quality word representation as they occur in context with many words in the corpus. For example, the co-occurrence of "New", "York" benefits the model in capturing better meaningful representation than the co-occurrence of "New", "the". The authors introduce a subsampling technique that discards the high-frequency words based on the probability formula computed for each word $w_i$ which is given below:
 
$$
P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$

where $t$ is a chosen threshold, typically around $10^{-5}$.


### Skip-Gram Implementation

Here we will be using text corpus of cleaned wikipedia articles provided by Matt Mahoney.

```python
!wget https://s3.amazonaws.com/video.udacity-data.com/topher/2018/October/5bbe6499_text8/text8.zip
!unzip text8.zip
```

```python
%matplotlib inline
%config InlineBackend.figure_format = "retina"

import time
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

```python
# check if gpu is available since training is faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
class Word2VecDataset(object):
    def __init__(self, corpus, min_count=5, window_size=5, threshold=1e-5):
        """ Prepares the training data for the word2vec neural network.
            Params:
                corpus (string): corpus of words
                min_count (int): words with minimum occurrence to consider
                window_size (int): context window size for generating word pairs
                threshold (float): threshold used for subsampling words
        """
        self.window_size = window_size
        self.min_count = min_count
        self.threshold = threshold

        tokens = corpus.split(" ")
        word_counts = Counter(tokens)
        # only consider the words that occur atleast 5 times in the corpus 
        word_counts = Counter({word:count for word, count in word_counts.items() if count >= min_count})
        
        self.word2idx = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # create prob dist based on word frequency
        word_freq = np.array(list(word_counts.values()))
        self.unigram_dist = word_freq / word_freq.sum()

        # create prob dist for negative sampling
        self.noise_dist = self.unigram_dist ** 0.75
        self.noise_dist = self.noise_dist / self.noise_dist.sum()

        # get prob for drop words
        self.word_drop_prob = 1 - np.sqrt(threshold / word_freq)

        # create the training corpus subsampling frequent words
        self.token_ids = [self.word2idx[word] for word in tokens 
                          if word in self.word2idx and random.random() > self.word_drop_prob[self.word2idx[word]]]

        # create word pairs for corpus
        self.generate_word_pairs()
    

    def generate_word_pairs(self):
        """ Creates the pairs of center and context words based on the context window size.
        """
        word_pair_ids = []
        for current_idx, word_id in enumerate(self.token_ids):
            # find the start and end of context window
            left_boundary = max(current_idx - self.window_size, 0)
            right_boundary = min(current_idx + self.window_size + 1, len(self.token_ids))

            # obtain the context words and center words based on context window
            context_word_ids = self.token_ids[left_boundary:current_idx] + self.token_ids[current_idx + 1:right_boundary]
            center_word_id = self.token_ids[current_idx]
            
            # add the word pair to the training set
            for context_word_id in context_word_ids:
                word_pair_ids.append((center_word_id, context_word_id))
        
        self.word_pair_ids = word_pair_ids


    def get_batches(self, batch_size):
        """ Creates the batches for training the network.
            Params:
                batch_size (int): size of the batch
            Returns:
                batch (torch tensor of shape (batch_size, 2)): tensor of word pair ids for a given batch
        """
        for i in range(0, len(self.word_pair_ids), batch_size):
            yield torch.tensor(self.word_pair_ids[i: i+batch_size], dtype=torch.long)
    
    
    def get_negative_samples(self, batch_size, n_samples):
        """ Samples negative word ids for a given batch.
            Params:
                batch_size (int): size of the batch
                n_samples (int): number of negative samples
            Returns:
                neg_samples (torch tensor of shape (batch_size, n_samples)): tensor of negative sample word ids
                    for a given batch
        """
        neg_samples_ids = np.random.choice(len(self.word2idx), size=(batch_size, n_samples), 
                                       replace=False, p=self.noise_dist)
        return torch.tensor(neg_samples_ids, dtype=torch.long)

```

```python
# read the file and initialize the Word2VecDataset
with open("text8", encoding="utf-8") as f:
    corpus = f.read()

dataset = Word2VecDataset(corpus)
```

```python
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """ Skip Gram variant of Word2Vec with negative sampling for learning word 
            embeddings. Uses the concept of predicting context words given the 
            center word.
            Params:
                vocab_size (int): number of words in the vocabulary
                embed_dim (int): embeddings of dimension to be generated
        """
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # embedding layers for input (center) and output (context) words
        self.embed_in = nn.Embedding(vocab_size, embed_dim)
        self.embed_out = nn.Embedding(vocab_size, embed_dim)

        # initialize the embeddings with uniform dist
        self.embed_in.weight.data.uniform_(-1, 1)
        self.embed_out.weight.data.uniform_(-1, 1)


    def forward(self, in_ids, pos_out_ids, neg_out_ids):
        """ Trains the Skip Gram variant model and updates the weights based on the
            criterion.
            Params:
                in_ids (torch tensor of shape (batch_size,)): indexes of the input words for a batch
                pos_out_ids (torch tensor of shape (batch_size,)): indexes of the output words (true pairs) for a batch
                neg_out_ids (torch tensor of shape (batch_size, number of negative samples)): 
                    indexes of the noise words (negative pairs) for a batch
        """
        emb_in = self.embed_in(in_ids)
        pos_emb_out = self.embed_out(pos_out_ids)
        neg_emb_out = self.embed_out(neg_out_ids)

        # calculate loss for true pair
        # ----------------------------
        # step 1 is calculate the dot product between the input and output word embeddings
        pos_loss = torch.mul(pos_emb_out, emb_in)      # element-wise multiplication
        pos_loss = torch.sum(pos_loss, dim=1)           # sum the element-wise components
        
        # step 2 is to calculate the log sogmoid of dot product
        pos_loss = -F.logsigmoid(pos_loss)

        # calculate loss for negative pairs
        # ----------------------------------
        # step 1 is calculate the dot product between the input and output word embeddings
        neg_loss = torch.bmm(-neg_emb_out, emb_in.unsqueeze(2)).squeeze()   # matrix-matrix multiplication
        neg_loss = torch.sum(neg_loss, dim=1)                               # sum the element-wise components

        # step 2 is to calculate the log sogmoid of dot product
        neg_loss = -F.logsigmoid(neg_loss)

        return torch.mean(pos_loss + neg_loss)

```

```python
# intialize the model and optimizer
vocab_size = len(dataset.word2idx)
embed_dim = 300
model = SkipGramModel(vocab_size, embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
```

```python
# training the network
n_epochs = 5
n_neg_samples = 5
batch_size = 512

print("-" * 60)
print("Start of training")
print("-" * 60)

for epoch in range(n_epochs):
    losses = []
    start = time.time()

    for batch in dataset.get_batches(batch_size):
        # get the negative samples
        noise_word_ids = dataset.get_negative_samples(len(batch), n_neg_samples)

        # load tensor to GPU
        input_word_ids = batch[:, 0].to(device)
        target_word_ids = batch[:, 1].to(device)
        noise_word_ids = noise_word_ids.to(device)
        
        # forward pass
        loss = model.forward(input_word_ids, target_word_ids, noise_word_ids)

        # backward pass, optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    end = time.time()

    print(f"Epochs: {epoch + 1}/{n_epochs}\tAvg training loss: {np.mean(losses):.6f}\tEllapsed time: {(end - start):.0f} s")

print("-" * 60)
print("End of training")
print("-" * 60)
```

```python
# get the trained embeddings from the model
embeddings = model.embed_in.weight.to("cpu").data.numpy()

# number of words to be visualized
viz_words = 200

# projecting the embedding dimension from 300 to 2
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

# plot the projected embeddings
plt.figure(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color="blue")
    plt.annotate(dataset.idx2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
```


## GloVe

### Overview

Previously, there were two main directions for learning distributed word representations: 1) count-based methods such as Latent Semantic Analysis (LSA) 2) direct prediction-based methods such as Word2Vec. Count-based methods make efficient use of statistical information about the corpus, but they do not capture the meaning of the words like word2vec and perform poorly on analogy tasks such as `king - queen = man - woman`. On the other hand, direct prediction-based methods capture the meaning of the word semantically and syntactically using local context but fail to consider the global count statistics. This is where GloVe comes into the picture and overcomes the drawbacks of both approaches by combining them. The author proposed a global log bilinear regression model to learn embeddings based on the co-occurrence of words. Note that the GloVe does not use a neural network for learning word vectors.

### Co-occurrence matrix

The authors used a co-occurrence matrix with a context window of fixed size $m$ to learn the word embeddings. Let's try to generate this matrix for the below toy example with a context window of size 2:
- I like deep learning
- I like NLP
- I enjoy flying

{% include figure.html path="assets/img/word_embeddings/co_occurrence_matrix.jpeg" class="img-fluid rounded" zoomable=true %}

<div class="caption">
    Co-occurrence Matrix Example (<a href="https://stanford.io/3n4FH4H">image source</a>)
</div>

### Mathematics

Before we move ahead, let's get familiarized with some notations.
- $X$ denotes the word-word co-occurrence matrix
- $X_{ij}$ denotes the number of times word $j$ occurs in the context of word $i$
- $X_i$ = $\sum_{k}{X_{ik}}$ denotes the number of times any word $k$ appearing in context of word $i$ and $k$ represents the total number of distinct words that appear in context of word $i$)
- $P_{ij} = P(j \| i) = \frac{X_{ij}}{X_i}$ denotes the co-occurence probablity i.e. probability that word $j$ appears in the context of word $i$

The denominator term in the co-occurrence probability accounts for global statistics, which word2vec does not uses. The main idea behind the GloVe is to encode meaning using the ratios of co-occurrence probabilities. Let's understand the above by deriving the linear meaning components for the following words based on co-occurrence probability.

{% include figure.html path="assets/img/word_embeddings/co_occurrence_probs.jpeg" class="img-fluid rounded" zoomable=true %}

<div class="caption">
    Co-occurrence Probabilities Example (<a href="http://nlp.stanford.edu/pubs/glove.pdf">image source</a>)
</div>

The matrix shows the co-occurrence probabilities for the words from the concept of the thermodynamic phases of water (i.e., $ice$ and $steam$). The first two rows represent the co-occurrence probabilities for the words $ice$ and $steam$, whereas the last row represents their ratios. We can observe the following:
- ratio is not neural for closely related words such as $solid$ and $ice$ or $gas$ and $steam$
- ratio is neutral for words relevant to $ice$ and $steam$ both or not completely irrelevant to both

The ratio of co-occurrence proababilities is a good starting point for learning word embeddings. Let's start with the most general function $F$ parametrized by 3 word vectors ($w_i$, $w_j$ and $\tilde{w_k}$) given below.
 
$$
F(w_i, w_j, \tilde{w_k}) = \frac{P_{ik}}{P_{jk}}
$$
 
where $w, \tilde{w} \in \mathrm{R^d}$ and $\tilde{w}$ represent the separate context words.

How do we choose $F$?
 
There can be many possibilities for choosing $F$ but imposing some constraints allows us to restrict $F$ and select a unique choice. The goal is to learn word vectors (embeddings) that can be projected in the word vector space. These vector spaces are inherently linear, i.e., think of vectors as a line in $\mathrm{R^d}$ space, so the most intuitive way is to take vector differences which makes our function $F$ as follows:
 
$$
F(w_i - w_j, \tilde{w_k}) = \frac{P_{ik}}{P_{jk}}
$$
 
We see that the right-hand side of the above equation is a scalar. Choosing a complex function such as a neural network would introduce non-linearities since our primary goal is to capture the linear meaning components from word vector space. Here, we take dot product on the left-hand side to make it a scalar similar to the right-hand side.
 
$$
F((w_i - w_j)^T \tilde{w_k}) = \frac{P_{ik}}{P_{jk}}
$$

We also need to preserve symmetry for the distinction between a word and a context word which means that if $ice$ can be used as a context word for $water$, then $water$ can also be used as a context word for $ice$. In a simple, it can be expressed as $w \leftrightarrow \tilde{w}$. This is also evident from our co-occurrence matrix since $X \leftrightarrow X^T$. In order to restore the symmetry, we require that function $F$ is a homomorphism between groups $(\mathrm{R, +})$ and $(\mathrm{R, \times})$.

<div class="note">
    <em>
        Given two groups, $\small (G, ∗)$ and $\small (H, \cdot)$, a group homomorphism from $\small (G, ∗)$ to $\small (H, \cdot)$ is a function $\small h : G \rightarrow H$ such that for all $u$ and $v$ in $\small G$ it holds that $\small h(u * v) = h(u) \cdot h(v)$.
    </em>
</div>

$$
\begin{align*}
    F((w_i - w_j)^T \tilde{w_k}) 
        &= F(w_i^T \tilde{w_k} + (-w_j^T \tilde{w_k})) \\
        &= F(w_i^T \tilde{w_k}) \times F(-w_j^T \tilde{w_k}) \\
        &= F(w_i^T \tilde{w_k}) \times F(w_j^T \tilde{w_k})^{-1} \\
        &= \frac{F(w_i^T \tilde{w_k})}{F(w_j^T \tilde{w_k})}
\end{align*}
$$

So if we recall the $F$ in terms of co-occurrence probabilities, we get the following.

$$
F(w_i^T \tilde{w_k}) = P_{ik} = \frac{X_{ik}}{X_i}
$$

Since we are expressing $F$ in terms of probability which is a non-negative term, so we apply exponential to dot product $w_i^T \tilde{w_k}$ and then take logarithm on both sides.
 
$$
w_i^T \tilde{w_k} = log(P_{ik}) = log(X_{ik}) - log(X_i)
$$
 
On the right hand, the term $log(X_i)$ is independent of $k$ so it can be absorbed into a bias $b_i$ for $w_i$. Finally, we add bias $\tilde{b_k}$ for $\tilde{w_k}$ to restore the symmetry.
 
$$
w_i^T \tilde{w_k} + b_i + \tilde{b_k} = log(X_{ik})
$$
 
The above equation leads to our objective function, a weighted least squares regression model where we use the weighting function $f(X_{ij})$ for word-word co-occurrences.
 
$$
J = \sum_{i,j = 1}^{V}f(X_{ij}) (w_i^T \tilde{w_k} + b_i + \tilde{b_k} - logX_{ik})^2
$$
 
where $V$ is the size of the vocabulary.

Here, the weighting function is defined as follows:

$$
f(x) = \begin{cases}
        (x / x_{max})^{\alpha} & \text{if}\ x < x_{max} \\
        1 & \text{otherwise}
       \end{cases}
$$

where $x_{max}$ is the cutoff of the weighting function and $\alpha$ is power scaling similar to Word2Vec.


### GloVe Implementation

Here we will be using text corpus of cleaned wikipedia articles provided by Matt Mahoney.

```python
!wget https://s3.amazonaws.com/video.udacity-data.com/topher/2018/October/5bbe6499_text8/text8.zip
!unzip text8.zip
```

```python
%matplotlib inline
%config InlineBackend.figure_format = "retina"

import time
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

```python
# check if gpu is available since training is faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
class GloVeDataset(object):
    def __init__(self, corpus, min_count=5, window_size=5):
        """ Prepares the training data for the glove model.
            Params:
                corpus (string): corpus of words
                min_count (int): words with minimum occurrence to consider
                window_size (int): context window size for generating co-occurrence matrix
        """
        self.window_size = window_size
        self.min_count = min_count

        tokens = corpus.split(" ")
        word_counts = Counter(tokens)
        # only consider the words that occur more than 5 times in the corpus 
        word_counts = Counter({word:count for word, count in word_counts.items() if count >= min_count})
        
        self.word2idx = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # create the training corpus
        self.token_ids = [self.word2idx[word] for word in tokens if word in self.word2idx]

        # create the co-occurrence matrix for corpus
        self.create_cooccurrence_matrix()


    def create_cooccurrence_matrix(self):
        """ Creates the co-occurence matrix of center and context words based on the context window size.
        """
        cooccurrence_counts = defaultdict(Counter)
        for current_idx, word in enumerate(self.token_ids):
            # find the start and end of context window
            left_boundary = max(current_idx - self.window_size, 0)
            right_boundary = min(current_idx + self.window_size + 1, len(self.token_ids))

            # obtain the context words and center words based on context window
            context_word_ids = self.token_ids[left_boundary:current_idx] + self.token_ids[current_idx + 1:right_boundary]
            center_word_id = self.token_ids[current_idx]

            for idx, context_word_id in enumerate(context_word_ids):
                if current_idx != idx:
                    # add (1 / distance from center word) for this pair
                    cooccurrence_counts[center_word_id][context_word_id] += 1 / abs(current_idx - idx)
        
        # create tensors for input word ids, output word ids and their co-occurence count
        in_ids, out_ids, counts = [], [], []
        for center_word_id, counter in cooccurrence_counts.items():
            for context_word_id, count in counter.items():
                in_ids.append(center_word_id)
                out_ids.append(context_word_id)
                counts.append(count)

        self.in_ids = torch.tensor(in_ids, dtype=torch.long)
        self.out_ids = torch.tensor(out_ids, dtype=torch.long)
        self.cooccurrence_counts = torch.tensor(counts, dtype=torch.float)


    def get_batches(self, batch_size):
        """ Creates the batches for training the network.
            Params:
                batch_size (int): size of the batch
            Returns:
                batch (torch tensor of shape (batch_size, 3)): tensor of word pair ids and 
                    co-occurence counts for a given batch
        """
        random_ids = torch.tensor(np.random.choice(len(self.in_ids), len(self.in_ids), replace=False), dtype=torch.long)

        for i in range(0, len(random_ids), batch_size):
            batch_ids = random_ids[i: i+batch_size]
            yield self.in_ids[batch_ids], self.out_ids[batch_ids], self.cooccurrence_counts[batch_ids]
```

```python
# read the file and initialize the GloVeDataset
with open("text8", encoding="utf-8") as f:
    corpus = f.read()

dataset = GloVeDataset(corpus)
```

```python
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, x_max=100, alpha=0.75):
        """ GloVe model for learning word embeddings. Uses the approach of predicting 
            context words given the center word.
            Params:
                vocab_size (int): number of words in the vocabulary
                embed_dim (int): embeddings of dimension to be generated
                x_max (int): cutoff of the weighting function
                alpha (int): parameter of the weighting funtion
        """
        super(GloVeModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.x_max = x_max
        self.alpha = alpha

        # embedding layers for input (center) and output (context) words along with biases
        self.embed_in = nn.Embedding(vocab_size, embed_dim)
        self.embed_out = nn.Embedding(vocab_size, embed_dim)
        self.bias_in = nn.Embedding(vocab_size, 1)
        self.bias_out = nn.Embedding(vocab_size, 1)

        # initialize the embeddings with uniform dist and set bias to zero
        self.embed_in.weight.data.uniform_(-1, 1)
        self.embed_out.weight.data.uniform_(-1, 1)
        self.bias_in.weight.data.zero_()
        self.bias_out.weight.data.zero_()

    
    def forward(self, in_ids, out_ids, cooccurrence_counts):
        """ Trains the GloVe model and updates the weights based on the
            criterion.
            Params:
                in_ids (torch tensor of shape (batch_size,)): indexes of the input words for a batch
                out_ids (torch tensor of shape (batch_size,)): indexes of the output words for a batch
                cooccurrence_counts (torch tensor of shape (batch_size,)): co-occurence count of input 
                    and output words for a batch
        """
        emb_in = self.embed_in(in_ids)
        emb_out = self.embed_out(out_ids)
        b_in = self.bias_in(in_ids)
        b_out = self.bias_out(out_ids)

        # add 1 to counts i.e. cooccurrences in order to avoid log(0) case
        cooccurrence_counts += 1

        # count weight factor
        weight_factor = torch.pow(cooccurrence_counts / self.x_max, self.alpha)
        weight_factor[cooccurrence_counts > 1] = 1
        
        # calculate the distance between the input and output embeddings
        emb_prods = torch.sum(emb_in * emb_out, dim=1)
        log_cooccurrences = torch.log(cooccurrence_counts)
        distances = (emb_prods + b_in + b_out - log_cooccurrences) ** 2

        return torch.mean(weight_factor * distances)

```

```python
# intialize the model and optimizer
vocab_size = len(dataset.word2idx)
embed_dim = 300
model = GloVeModel(vocab_size, embed_dim).to(device)
optimizer = optim.Adagrad(model.parameters(), lr=0.05)
```

```python
# training the network
n_epochs = 5
batch_size = 512

print("-" * 60)
print("Start of training")
print("-" * 60)

for epoch in range(n_epochs):
    losses = []
    start = time.time()

    for input_word_ids, target_word_ids, cooccurrence_counts in dataset.get_batches(batch_size):
        # load tensor to GPU
        input_word_ids = input_word_ids.to(device)
        target_word_ids = target_word_ids.to(device)
        cooccurrence_counts = cooccurrence_counts.to(device)
        
        # forward pass
        loss = model.forward(input_word_ids, target_word_ids, cooccurrence_counts)

        # backward pass, optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    end = time.time()

    print(f"Epochs: {epoch + 1}/{n_epochs}\tAvg training loss: {np.mean(losses):.6f}\tEllapsed time: {(end - start):.0f} s")

print("-" * 60)
print("End of training")
print("-" * 60)
```

```python
# get the trained embeddings from the model
emb_in = model.embed_in.weight.to("cpu").data.numpy()
emb_out = model.embed_out.weight.to("cpu").data.numpy()
embeddings = emb_in + emb_out

# number of words to be visualized
viz_words = 200

# projecting the embedding dimension from 300 to 2
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

# plot the projected embeddings
plt.figure(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color="blue")
    plt.annotate(dataset.idx2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
```

<div class="citations">
    <d-cite key="Mikolov2013EfficientEO">
    <d-cite key="Mikolov2013DistributedRO">
    <d-cite key="mccormick2016Word2Vec">
    <d-cite key="mccormick2017Word2Vec">
    <d-cite key="cs224n_word2vec">
    <d-cite key="kaggle_word2vec">
    <d-cite key="Pennington2014GloVeGV">
    <d-cite key="group_homomorphism_wiki">
    <d-cite key="glove_stack_exchange">
    <d-cite key="gauthier_glove">
    <d-cite key="gavrilov_glove">
</div>
