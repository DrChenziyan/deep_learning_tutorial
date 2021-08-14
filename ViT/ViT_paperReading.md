## Vision Transformer
1. [Paper](https://arxiv.org/pdf/2010.11929.pdf)
2. Introduced by Google

### Parameters summary (num_classes=100)
| Model | Patches |Layers(Repeats) | Hidden Size(D) | MLP size | Heads | Input size | Params |
| :--: | :--: | :--: | :--:| :--: | :--: | :--: | :--: | 
| ViT-Base | 16 | 12 | 768 | 3072 | 12 | 224 | 86M |
| ViT-Base | 32 | 12 | 768 | 3072 | 12 | 224 | 88M |
| ViT-Base | 32 | 12 | 768 | 3072 | 12 | 384 | 88M |
| ViT-Large | 16 | 24 | 1024 | 4096 | 16 | 224 | 304M |
| ViT-Large | 32 | 24 | 1024 | 4096 | 16 | 224 | 304M |
| ViT-Large | 16 | 24 | 1024 | 4096 | 16 | 384 | 306M |
| ViT-Large | 32 | 24 | 1024 | 4096 | 16 | 384 | 306M |
| ViT-Huge | 16 | 24 | 1280 | 5120 | 14 | 224 | 632M |


### Memroy access analysis
- Each token should do the $softmax(\frac{QK^{T}}{\sqrt{d_k}})$, thus leading to quadratic complexity with respect to the number of tokens.
- Complexity of MSA
    $$Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_k}})V$$
    - $QK^{T}$: (n, d) * (d, n)  --> (n, n) $\qquad\qquad\mathcal{O}=n^2d$
    - $softmax$: $\qquad\qquad\mathcal{O}=n^2$
    - $(..)\times V$:(n, n) * (n, d)  --> (n, d) $\qquad\qquad\mathcal{O}=n^2d$
    - 