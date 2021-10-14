## DynamicVIT
1. [Paper](https://arxiv.org/pdf/2106.02034.pdf)
2. Introduced by Tsinghua and UCLA
3. code: https://github.com/raoyongming/DynamicViT

## Paper Reading
### Abstract
1. Attention is sparse ----- how to understand the sparsification?
2. How the network remove the redundant tokens dynamically? -- **prune the less important tokens**
3. Which tokens are needed and which are redundant, how to judge?
4. **Attention masking** to differentite tokens importance -- how?
5. reducing greatly FLOPs with little decrease of accuracy


### Contents
1. Base framework --- ViT as backbone
2. Token sparsification  --- performed hierarchically through the whole network at certain locations
3. end-to-end manner --- attention masking

#### **Token sparsification**
By conducting a binary decision mask to determine whether keep or drop the tokens

total masks : number_embeded

initialize all masks to 1 and update progressively

for each token $x$:
firstly do local
$$ z^{local} = MLP(x)\in \mathbb{R}^{N \times C'} $$
then do global
$$ z^{global} = Agg(MLP(x), D)\in \mathbb{R}^{C' } $$
where
$$Agg(u, \widehat{D}) = \frac{\sum^N_{i=1}\widehat{D}_iu_i}{\sum^{N}_{i=1}\widehat{D_i}},\quad u\in\mathbb{R}^{N \times C'}$$

concat $z^{local}$ and $z^{global}$ and do the softmax

*Explaination:*
   1. The local feature encodes the information of a certain token while the global feature contains the context of the whole image, then using mlp and softmax to choose which to keep or drop.
   2. herein, softmax applied for binary classification which used to update mask $D$ .
   3. but it may not be differentiable

#### **End-to-end Optimization**
To implement pruning sparsification, using [`Gumbel-Softmax`](https://arxiv.org/pdf/1611.01144.pdf) to make it possible for end-to-end training.
* Gumbel sampling
  * Gumbel distribution(极值分布): $F(x) = e^{-e^{-x}}$
  * Gumbel sampling: $G_i = -log(-log(U_i))$
  * 使用Gumbel 采样的噪声可以最接近于原始的概率分布，原始分布可以通过np.random.choice来估计，但是无法进行反向传播
  * 因此通过Gumbel-softmax就可以模拟真实的类别分布情况(tokens importance)，设置在训练过程中tempature逐渐降低来接近真实情况，因此来对tokens做keep or drop
  * 

* Tokens pruning
  * knowledge distillation: training a teacher model with a high `Temperature` to get a soft predictions(*dark knowledge*) and distillation loss, and followed by a student model training with the ground truth labels in addition to soft predictions.
    * $$\mathcal{L}_{KD} = \alpha * H(y,\sigma(\mathcal{z_s})) + \beta * H(\sigma(\mathcal{z_t}; \rho),\sigma(\mathcal{z_s}; \rho))  =  \alpha * H(y,\sigma(\mathcal{z_s})) + \beta * [KL(\sigma(\mathcal{z_t}; \rho), \sigma(\mathcal{z_s}; \rho)) + H(\sigma(\mathcal{z_t}))]$$
    * where H -- loss function, y -- ground truth,  $\sigma$ -- softmax,    $\rho$ -- temperature para,     $\alpha$ and $\beta$ -- hyper parameters,  $\mathcal{z_t}$ and $\mathcal{z_s}$ -- teacher and student logits respectively,  KL -- Kullback-Leibler Divergence/relative entropy, which used for calculating the differences of two distribution in one events, herein, $\mathcal{z_t}$ and $\mathcal{z_s}$/
  * KL divergence
  * MSE loss
  * $$\mathcal{L} = \mathcal{L}_{cls} + \lambda_{distill}\mathcal{L_{distill}} + \lambda_{KL}\mathcal{L}_{KL} + \lambda_{MSE}\mathcal{L}_{MSE} $$


### Summary
    1. Accuracy
    2. Model scaling: like EfficientNet, DynamicViT 
    3. Highlights:
       1. pruning tokens: KD
       2. tokens sparsification: Gumbel-softmax and attention mask

