# Distributed Tokenized Attention

**Authors:**
* @Lifann

## **Summary**

## **Motivation**
- Contemporary deep learning models in NLP and recommenders domains are widely built on transformer structure, where attention was used to represent the low-rank
  context space to score the tokens inside the context.
- The L-square cost in storage and computation is pain in the bone for the engineering of transformer models, especially when the context grows too long.
- In practice, I found that the long context can significantly improve the expressive power of the model.
- There are many researches on MP, DP, PP, TP strateges to split memory costs on multiple nodes, or apply more computing power on the attention. But the sparsity
  existing in many real world datasets have not been taken into consideration in model construction.
- In this RFC, I proposed a distributed tokenized attention algorithm, to achieve long context while keeping high batch size and throughput, with
  better latency on full-rank self-attention or cross-attention, based on the sparsity of the symbolic datasets. Furthur more I proposed an enquivalent substitution
  method of target attention, which is widely used in recommenders system industry, to remarkably reduce the computation, storage, and communication costs.

## Reference

## **Proposed Implementation**

#### Attention
- Self-attention described in [paper(attention is all you need)](https://arxiv.org/abs/1706.03762) can be described by a formula as:
<img width="262" alt="image" src="https://github.com/Lifann/TokenizedAttention/assets/67221898/8abf3559-3607-4ea9-beaa-270ff55ae0be">

- Given:
  * `B`: `batch size`,
  * `L`: `sequence lengths`
  * `D`: `embedding dimension`
  The self-attention procedure can be described as follows:
  1. `tokens`: `(B, L)` is batched sequence. `lengths`: `(B,)` denotes lengths of sequence in each sample.
     And there are `N` ranks and every rank hold one GPU.
  2. Every rank read a mini-batch of `tokens`: `(B, L)`, and its `lengths`.
  3. Every rank routes the `tokens` to dim-1, while recoding position of every token. The `rankwise-sizes`: `(N, B)`
     is recorded to deontes the number of tokens which will be send to rank-i, and was generated from batch-j, where
	 `(i, j)` is one of the element in `rankwise-sizes`. Position is an integer to mark where the token is.
	 `pos = (rank * B + b) * L + pos-in-L`, where b is the sample id in mini batch of local rank, and pos-in-L is the
	 position in sample of the token.
  4. Alltoall the `rankwise-tokens`, and its `positions`. Alltoall the `rankwise-sizes`.
  5. Lookup the embedding parameter and get activated embeddings in MP-style, where keeping the distributed
     rankwise-tokens stored in unique space.

This is the bulk of the RFC. Explain the design in enough detail for somebody familiar with PyTorch to understand, and for somebody familiar with the implementation to implement. 
Tndaoguhis should get into specifics and corner-cases, and include examples of how the feature is used, and how it will interact with other features. Any new terminology should be defined here.
Consider:
*   using examples and diagrams to help illustrate your ideas.
*   including code examples, if you're proposing an interface or system contract.
*   linking to project briefs or wireframes that are relevant.

## **Proposed API**

```python3
class TokenizedAttention(object):
  def __init__(self, batch_size, sequence_length, nranks, embedding_fn):
    pass

  def forward(self, tokens, lengths):
    pass
```
