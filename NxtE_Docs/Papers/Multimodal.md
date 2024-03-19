# CLIP
擅长图像文本匹配、图像文本检索
Image Text Contrastive(ITC) Loss
# ViLT
移除了预训练的目标检测器，换成了可学习的 Patch Embedding Layer
性能不够高，推理时间快，训练时间很慢
# ALBEF
- ITC(Image-Text Contrastive) Loss: (I, T)
- ITM(Image-Text Matching) Loss: (I, T)
- MLM(Masked Language Modeling): (I, T') 多次前向 两次forward
  带掩码的文本与图片输入 ALBEF 模型，借助图像信息恢复原始文本

Two contributions:
1. Introduce ITC Loss
2. self-training method which learns from Pseudo Label produced by a momentum model ([[Moco]])

data augmentation
Semantic Preserving：只要是语义匹配的图像文本对，就应该被当成一对

图文检索

Image -> patch -> patch embedding layer -> vision transformer

对比学习：负样本需要 constraint，选择最接近正样本的负样本
hard negatives: 依赖 ITC 生成，一个图片与同一个 batch 里的所有文本计算 cos similarity，选择除自己之外相似度最高的文本。

Noisy web data: Weakly-correlated
self-training [[Noisy Student]], [[DINO]]
EMA(exponential-moving-average)