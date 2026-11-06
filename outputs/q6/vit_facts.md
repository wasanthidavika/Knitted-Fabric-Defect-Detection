| model | input | patch | patch_tokens | sequence_length_with_CLS | attention_entries_per_head_per_layer | median_defect_short_side_px | share_defects_smaller_than_one_patch_% | params_with_7_class_head |
|---|---|---|---|---|---|---|---|---|
| ViT-B/16 @384 | 384x384 | 16 | 576 | 577 | 332929 | 8.7 | 80.6 | 86095879 |
| ViT-B/16 @224 | 224x224 | 16 | 196 | 197 | 38809 | 5.1 | 92.9 | 85804039 |
| CLIP ViT-B/32 @224 | 224x224 | 32 | 49 | 50 | 2500 | 5.1 | 98.1 | None |