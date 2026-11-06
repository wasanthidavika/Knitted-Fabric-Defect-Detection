# Q6 - Generative AI / foundation models (report skeleton, ~4 pages)

## 6.a Applying foundation models

### 1. What was applied, and why it is technically meaningful here
- Task: image-level multi-label defect presence (7 classes + "defective or not"), derived from the boxes. YOLO and the SVM are converted to the same form (max detection confidence per class), so all four models are scored on the SAME test images with the SAME metric: image-level macro-F1, per-class thresholds tuned on validation. mAP@0.5 stays the primary metric for the detectors (Q3) but cannot be computed for ViT/CLIP (no boxes): say so.
- ViT-B/16: ImageNet-21k pretrained, fine-tuned (transfer learning). CLIP ViT-B/32: contrastive image-text foundation model used zero-shot with text prompts.
- A generative LLM/VLM was not used for the main task because [reason: boxes/pixel-level defects of ~10-20 px are below what current general VLMs reliably localise; verify on a few validation images if you want to claim this]. State it as your judgement, with the evidence you have.

### 2. Transformer treatment anchored to your data (the lecturer's point)
Insert `outputs/q6/vit_facts.md`. Explain in your own words, with YOUR numbers:
- Patch embedding: each 16x16x3 patch (768 numbers) is linearly projected to 768 dimensions; at 384x384 that is 24 x 24 = **576 patch tokens, sequence length 577 with the [CLS] token** (224: 196 tokens, 197 with [CLS]; CLIP ViT-B/32 at 224: 49 tokens, 50 with [CLS]).
- Positional embeddings (learned) restore position; 12 encoder layers, each: multi-head self-attention (12 heads of 64 dimensions) + MLP (768 -> 3072 -> 768), LayerNorm, residual connections; [CLS] output -> linear head (7 logits, sigmoid).
- Self-attention compares every token with every other: 577^2 = 332,929 attention entries per head per layer (224: 38,809): cost grows quadratically with tokens, which is why higher resolution is expensive.
- Link to your data: [X]% of defects are smaller than ONE 16-px patch at 384 (from the table), so a token mixes defect and background; global context helps when defects co-occur or when fabric texture varies, but tiny defects are hard for patch-based models. Resolution ablation (V01 at 224: 197 tokens) tests this.
- Why fine-tune rather than train from scratch: 2,360 images are too few for a ViT; pretraining supplies general visual features.

### 3. ViT experiments (all logged; selection on validation only)
| Run | Change | Val macro-AP | Val macro-F1 | Best epoch | Time (min) | Learned |
|---|---|---|---|---|---|---|
| V00 | ViT-B/16 @384, full fine-tune, lr 3e-5 | | | | | |
| V01 | 224 input (197 tokens) | | | | | |
| V02 | frozen backbone (linear probe) | | | | | |
| V03 | lr 1e-5 | | | | | |
(add --pos-weight if rare classes fail). Epoch chosen by validation macro-AP; seeds 42 (and 43/44 for the chosen config if results vary).

### 4. CLIP zero-shot
- Model/version: [openai/clip-vit-base-patch32, transformers x.y], run locally, no paid service.
- Method: tiles (224 px, stride 112; 20 tiles for a 640x480 image) or whole image, text prompts (class descriptions x templates, ensembled, plus a "normal fabric" prompt), softmax over 8 prompts, aggregate over tiles (max or top-3 mean).
- Prompts: paste `configs/q6_prompts.yaml`. Selection: table of ALL 12 combinations from `clip_selection.csv` (val macro-AP), chosen on validation only; no test result influenced any prompt.
- Honest expectations: CLIP sees 224 px tiles and one ViT-B/32 patch is 32 px, larger than most defects: say what the results show.

### 5. Documentation of configuration, services, limitations, costs
Model names/tags and library versions (`pip freeze`), hyper-parameters, hardware, one-time weight download from the hub (internet), licences [verified], no API fees. Limitations: image-level only (no localisation), zero-shot prompts depend on how well the descriptions match real defects, small test classes (CI widths), class imbalance.

## 6.b Comparative analysis and innovation

### 6. Evidence table (from outputs/q6/q6_comparison.md and q6_final.json)
| Model | Test macro-F1 [95% CI] | Macro-AP | Defect recall @ val operating point | False-alarm rate | Latency (ms/image, hardware) | Training time | Labelled data needed | Cost / prediction | Privacy | Third-party dependence |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO final | | | | | | | | | | |
| SVM baseline | | | | | | | | | | |
| ViT-B/16 | | | | | | | | | | |
| CLIP zero-shot | | | | | | | | | | |
Paired bootstrap (`q6_paired_bootstrap.csv`): for each model, macro-F1 difference vs the reference with 95% CI and P(better). A CI containing 0 means NO demonstrated difference.
Cost per prediction: no API was used (marginal fee 0); if you quote a compute cost, state the assumed GPU price and how latency was converted (`--gpu-usd-per-hour`), or say it was not computed. Latency scopes differ between models (see each JSON): state them.

### 7. Conclusion and deployment conditions (must follow the numbers)
- State which model is best on the primary metric AND whether the difference is demonstrated (CI). If the zero-shot model is near chance or the SVM wins, say so and explain with evidence (defect size, patch size, data volume).
- When each is appropriate: YOLO [boxes needed, labelled data exist, real-time, local]; SVM [no GPU, tiny data, features explainable]; ViT [image-level triage is enough, labelled data exist]; CLIP [no labels yet, new defect types prototyped quickly, low-stakes triage]. Adjust to your findings.
- What would change the conclusion: [more data, higher resolution, other prompts, larger CLIP].

### 8. Opportunities beyond the ANN (at least one, realistic)
1. **Label auditing (implemented).** CLIP crop embeddings + class-prior-balanced nearest-neighbour vote on train/val boxes: [N flagged of M; your manual review of 50 flagged + 50 random: precision of flags]; consequence: [label noise limits the ANN; error analysis cases may be annotation problems]. Labels were not changed.
2. **Generated inspection notes (not implemented).** A generative VLM could turn "Hole, score 0.83, box at ..." plus the crop into a plain-language note or shift report. Needs: an instruction-tuned VLM, a prompt template, a human review step; risks: hallucinated detail, cost/latency, privacy if hosted. Evaluate by [inspector rating, factual agreement with detector output].
3. **Retrieval of similar past defects.** CLIP embeddings let an inspector search historical defect crops similar to the current one.
4. **Synthetic data for rare classes** (Needle Mark, Loop Mark): possible with image generation, but synthetic defects must be validated against real ones before use; not attempted.
Say clearly which are implemented and which are proposals.
