# Distributed Tokenized Attention

**Authors:**
* @Lifann

## **Summary**

## **Motivation**
- Contemporary deep learning models in NLP and recommenders domains are widely built on transformer structure, where attention was used to represent the low-rank
  context space to score the tokens inside the context.
- The L-square cost in storage and computation is pain in the bone for the engineering of transformer models, especially when the context grows too long.
- In out practice, we found that the long context can significantly improve the expressive power of the model.
- So, here in the RFC, we proposed a distributed tokenized attention algorithm, to achieve long context while keeping high batch size and throughput, with
  better latency.

## Reference
- Self-attention described in [paper(attention is all you need)](https://arxiv.org/abs/1706.03762) can be described by a formula as:
<img width="262" alt="image" src="https://github.com/Lifann/TokenizedAttention/assets/67221898/8abf3559-3607-4ea9-beaa-270ff55ae0be">

## **Proposed Implementation**
This is the bulk of the RFC. Explain the design in enough detail for somebody familiar with PyTorch to understand, and for somebody familiar with the implementation to implement. 
This should get into specifics and corner-cases, and include examples of how the feature is used, and how it will interact with other features. Any new terminology should be defined here.
Consider:
*   using examples and diagrams to help illustrate your ideas.
*   including code examples, if you're proposing an interface or system contract.
*   linking to project briefs or wireframes that are relevant.

## **Proposed API Implementation**

```python3
class TokenizedAttention(object):
  def __init__(self, batch_size, sequence_length, nranks, embedding_fn):
    pass

  def forward(self, tokens, lengths):
    pass
```
