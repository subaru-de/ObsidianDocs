# 1 Introduction
The Predominant focus:
- refining alignment between modalities
- aligning with human intent via a MM Pre-Training (PT) + MM Instruction-Tuning (IT) pipeline

a research fervor on MM-LLMs has been sparked.

- Initial research primarily focuses on MM content comprehension and text generation.
- In pursuit of MM-LLMs capable of both MM input and output, some studies additionally explore the generation of specific modalities.
- Recent research endeavors have focused on mimicking human-like any-to-any modality conversion, shedding light on the path to artificial general intelligence. Some efforts aim to amalgamate LLMs with external tools to reach an approaching ‘any-to-any’ MM comprehension and generation.

![[Pasted image 20240312191330.png]]
# 2 Model Architecture
Five components comprising model architecture
- Modality Encoder
- Input Projector
- LLM Backbone
- Outpupt Projector
- Modality Generator

MM-LLMs that emphasize MM understanding only include the first three components.
During training, Modality Encoder, LLM Backbone, and Modality Generator are generally maintained in a frozen state.

The primary optimization emphasis is on Input and Output Projectors. Given that Projectors are lightweight components, the proportion of trainable parameters in MM-LLMs is notably small compared to the total parameter count (typically around 2%).

![[Pasted image 20240313205241.png]]
## 2.1 Modality Encoder
### Visual Modality
- NFNet-F6: normalizer-free ResNet
- ViT
- CLIP ViT
- Eva-CLIP ViT
### Audio Modality
- C-Former: CIF alignment mechanism
- HuBERT
- BEATs
- Whisper
### 3D Point Cloud Modality
- ULIP-2

ImageBind, a unified edcoder covering six modalities, including image/video, text, audio, heat map, inertial measurement units, and depth.
## 2.2 Input Projector
- Linear Projector
- MLP (Multi-Layer Proceptron)
- Cross-Attention
- Q-Former
- P-Former
## 2.3 LLM Backbone

## 2.4 Output Projector
## 2.5 Modality Generator
# 3 Training Pipeline
## 3.1 MM PT
## 3.2 MM IT
# 4 SOTA MM-LLMs
# 5 Benchmarks and Performance
## Training Recipes
1. Firstly, higher image resolution can incorporate more visual details for the model, benefiting tasks that require fine-grained details. However, higher resolutions lead to longer token sequences, incurring additional training and inference costs.
   - MiniGPT-v2
   - Monkey
   - Docpedia
2. The incorporation of high-quality SFT data can significantly improve performance in specific tasks.
3. Performing PEFT on the LLM Backbone promotes deep embedding alignment, crucial for ICL.
4. Interleaved Image-Text data proves beneficial, whereas Image-Text pairs alone are sub-optimal.
5. Re-blending text-only instruction data with image-text data during SFT not only addresses the degradation of text-only tasks but also enhances VL task accuracy.
# 6 Future Directions
## More Powerful Models
- Expanding Modalities
- Diversifying LLMs
- Improving MM IT Dataset Quality
- Strengthening MM Generation Capabilities
## More Challenging Benchmarks
## Mobile/Lightweight Deployment
## Embodied Intelligence
## Continual IT
# 7 Conclusion