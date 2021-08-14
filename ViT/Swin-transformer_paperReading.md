## Swin-Transformer
1. [Paper](https://arxiv.org/pdf/2103.14030.pdf)
2. Introduced by Microsoft Research Asia
3. code: https://github.com/microsoft/Swin-Transformer.
   
## Paper Reading
### 1. Weakness of ViT
 - Visual elements vary substantially in scale. For example, Near and Big, Far and Small in Space will bring the visual difference which should not be treated as the same tokens.
 - Much higher resolution in images will bring much more computional complexity.
### 2. Strategy of Swin-ViT
- Hierarchical feature maps -- from small size patches and merging neighboring patches in deeper layers.
    ```html
     With these hierarchical feature maps, the Swin Transformer model can conveniently leverage advanced techniques for dense prediction such as feature pyramid networks (FPN) or U-Net. The linear computational complexity is achieved by computing self-attention locally within non-overlapping windows that partition an image.
    ```
- Shift window for attention -- like the kernel moving in CNN, shift window provides the powerful connections among patches. Also, sharing 
- All query patches within a window share the same key set, which could facilitate memory access in hardware and have loer latency.

### 3. Model Architecture
    (B, 3, H, W)        -- input size
            |           -- 4 x 4 patch_size
    (B, H/4, W/4, 48)   -- Patch Partition
            |
            |           -- Linear embedding + Swin-Transformer Block * 2
            |
    (B, H/4, W/4, C)
            |
            |           -- Linear embedding + Swin-Transformer Block * 2
            |
    (B, H/8, W/8, 2C)
            |
            |           -- Linear embedding + Swin-Transformer Block * 6
            |
    (B, H/16, W/16, 4C)
            |
            |           -- Linear embedding + Swin-Transformer Block * 2
            |
    (B, H/32, W/32, 8C)
            |
            |
        Classification

### 4. Details in code
  - Relative position encoding
    1. Create a mshgrid network with dimension of (2, Wh, Ww), where `W` means the window size, and then flatten the newotrk (2, Wh*Ww)
    2. Generate relative coords by adding a dimension and do the minus.
    3. Shift the coords and start from 0 by adding the `window_size` to each coords.
    4. Differentate horizonal and vertical coords by $vertical_coords *= 2*window_size[1]-1$ and plus them.
  - Shifted window MSA
    1.  



