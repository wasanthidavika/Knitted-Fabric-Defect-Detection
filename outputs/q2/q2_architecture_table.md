| idx | from | type | x n | params | output (C x H x W) | section |
|---|---|---|---|---:|---|---|
| 0 | -1 | Conv | 1 | 928 | 32x320x320 | backbone |
| 1 | -1 | Conv | 1 | 18,560 | 64x160x160 | backbone |
| 2 | -1 | C2f | 1 | 29,056 | 64x160x160 | backbone |
| 3 | -1 | Conv | 1 | 73,984 | 128x80x80 | backbone |
| 4 | -1 | C2f | 2 | 197,632 | 128x80x80 | backbone |
| 5 | -1 | Conv | 1 | 295,424 | 256x40x40 | backbone |
| 6 | -1 | C2f | 2 | 788,480 | 256x40x40 | backbone |
| 7 | -1 | Conv | 1 | 1,180,672 | 512x20x20 | backbone |
| 8 | -1 | C2f | 1 | 1,838,080 | 512x20x20 | backbone |
| 9 | -1 | SPPF | 1 | 656,896 | 512x20x20 | backbone |
| 10 | -1 | Upsample | 1 | 0 | 512x40x40 | neck |
| 11 | [-1, 6] | Concat | 1 | 0 | 768x40x40 | neck |
| 12 | -1 | C2f | 1 | 591,360 | 256x40x40 | neck |
| 13 | -1 | Upsample | 1 | 0 | 256x80x80 | neck |
| 14 | [-1, 4] | Concat | 1 | 0 | 384x80x80 | neck |
| 15 | -1 | C2f | 1 | 148,224 | 128x80x80 | neck |
| 16 | -1 | Conv | 1 | 147,712 | 128x40x40 | neck |
| 17 | [-1, 12] | Concat | 1 | 0 | 384x40x40 | neck |
| 18 | -1 | C2f | 1 | 493,056 | 256x40x40 | neck |
| 19 | -1 | Conv | 1 | 590,336 | 256x20x20 | neck |
| 20 | [-1, 9] | Concat | 1 | 0 | 768x20x20 | neck |
| 21 | -1 | C2f | 1 | 1,969,152 | 512x20x20 | neck |
| 22 | [15, 18, 21] | Detect | 1 | 2,118,757 | [11x8400; dict] | detect head |

Total parameters (nc=7): **11,138,309**; COCO reference (nc=80): 11,166,560; GFLOPs @ 640: 28.6549504
