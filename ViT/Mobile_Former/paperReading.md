## Conformer
1. [Paper](https://arxiv.org/pdf/2108.05895.pdf)
2. Introduced by Microsoft and USTC(中科大)
3. code: 
   
## paper Reading
### 1. Abstract
*  Mobile-Former was designed as a two-way bridge in MobileNet and Transformer
*  This structure is likely to Conformer, combining the CNN and ViT
*  Advantages: lower computational complexity but higher representation power. Combining the advantages of MobileNetV3 and Transformer

### 2. Stratage of Mobile-Former
 * Each Mobile-Former block contains a sub-mobile block, a sub-former block, mobile2former block and former2mobile block.
 * **Mobile Block** 1x1Conv --> Dynamic_Relu --> 3x3 dw_Conv --> Dynamic_Relu --> 1x1Conv (expandConv -> pwConv -> projConv `invertedResidualBlock`)
 * **Former Block** NormLayer --> MHA --> NormLayer --> MLP (`standard Transformer Block`)
 * **Mobile2Fromer Block**:
    - input: $x_i$ [B, C, HW] and $z_i$ [B, num_tokens, embeded_dim]
    - $z_i$ [B, num_tokens, embedd_dim] --> [B, num_tokens, C] by $W^Q$(need to learn)
    - do the attention $z_i \times x_i$ (QK) [B, num_tokens, C] x [B, C, HW] --> [B, num_tokens, HW]
    - then  do the `softmax` and times $V$ [B, num_tokens, HW] x [B, HW, C] --> [B, num_tokens, C] --> [B, num_tokens, embeded_dim]
    - $$z^{out} = z + [Attention(z_hW^Q_h, x_h, x_h)]_{h=1:H} W^O$$
 * **Former2Mobile Block**:
   * input: $x_i$ [B, C, HW] and $z_i$ [B, num_tokens, embeded_dim]
   * $z_i$ [B, num_tokens, embedd_dim] --> [B, num_tokens, C] by $W^K$(need to learn)
   * do the attention $x_i \times z_i$ [B, HW, C] x [B, C, num_tokens] --> [B, HW num_tokens]
   * thne do the `softmax` and times $V$ [B, HW num_tokens] x [B, num_tokens, C] --> [B, HW, C]
   * $$x^{out} = x + [Attention(x_h, z_iW^k_h, z_iW^k_h)]_{h=1:H} W^O$$

## Code details
* bottleneck_lite: '''Proposed in Yunsheng Li, Yinpeng Chen et al., MicroNet, arXiv preprint arXiv: 2108.05894v1'''\
    Sequential (conv2d --> relu6 --> conv2d --> batchNorm)

* Dynamic Relu: '''Yinpeng Chen, Xiyang Dai et al., Dynamtic ReLU, arXiv preprint axXiv: 2003.10027v2'''\
     theta(x) is from $z$, so that re-writing the DyRELU class 

* For `Former`, the expand_ratio is 2 instead of 4

* 