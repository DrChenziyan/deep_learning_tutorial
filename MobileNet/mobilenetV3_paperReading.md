## MobileNet V3
1. [Paper](https://arxiv.org/pdf/1905.02244.pdf)
2. Introduced by Google
3. Cited by the latter network such as EfficientNet_V1 and EfficientNet_V2

## paper reading
1. This paper first used NAS(natural architecture searching) to imporve the model performance.
    - [NetAdapt algorithm](https://arxiv.org/pdf/1804.03230.pdf)
    - It tried to fine-tune the number of filters for every conv in a sequential manner and picks the model with the highest accuracy.
    - How to apply it:
        * Start with a seed network architecture
        * Generate a set of new proposals. 
        * Use the pre-trained model to fine-tune the proposal and get the accuracy for **T** steps.
        * Iterate to target latency.
        $$\underset{Net}{\rm maximize} \: Acc(Net)$$
        $${\rm subject\:to} \: Res_j(Net)\leq Bud_j, j=1,\ldots,m$$
        where $Res_j(.)$ evaluates the direct metric for resource consumption of the $j_{th}$, $Bud_j$ is the budget of the $j_{th}$ resource and the constraint on the optimization.
        * `Sudo code could be seen in the `  [NetAdapt](https://arxiv.org/pdf/1804.03230.pdf)`.

2. MobileNet_v3 update the model blocks 
    
    - In mobileNet_V1: Depthwise separable convolutions was firstly introduced as a efficient way for computing and higher performance.
    
    - In mobileNet_V2: Inverterd residual structure was introduced to make more efficient layer structures.
        * (1_1expansion_conv --> dw_conv --> 1x1_projection_conv)
        * 'This structure maintains a compact representation at the input and the output while expanding to a higher-dimensional feature space internally to increase the expressiveness of nonlinear perchannel transformations.' (from paper)
    
    - In mobileNet_V3: Sequeeze-and-Exciation block was introduced for channels attention.
        * New nonlinerities (h-swish modified from swish)
         $$ sigmoid(x) / \sigma(x) = \frac{1}{1 + e^{-x}}$$
         $$ ReLu6(x) = min(max(0,x),6)$$
         $$ SiLu/swish(x) = x * \sigma(x)$$

         The following formulas are the newly introduced.
         $$ h(ard)-sigmoid(x) = \frac{ReLU6(x+3)}{6}$$ 
         $$ h-swish(x) = x\frac{ReLU6(x+3)}{6}$$ 
        ReLU6 is available on virtually all software and hardware frameworks. And the h-sigmoid and h-swish could be easy for derivaition and quantization.
        * Squeeze-and-Exciation blocks: EfficientNet also applied this.
        * Redesign the expensive layers: Use less filters than V1 and V2 and more efficient last stage to reduce the latency.