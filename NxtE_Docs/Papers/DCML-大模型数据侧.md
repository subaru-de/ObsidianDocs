# LAION-5B: An open large-scale dataset for training next generation image-text models

LAION-5B: An open large-scale dataset for training next generation image-text models
[[2210.08402] LAION-5B: An open large-scale dataset for training next generation image-text models](https://arxiv.org/abs/2210.08402)
CLIP models for the entire dataset. We removed all English image-text pairs with cosine similarity below 0.28

What is ==nearest neighbor indices==
==distribution shifts==
## 1 Introduction
CLIP 和 BASIC 分别的预训练数据集分别包含 4 亿对和 66 亿对图像-文本数据集，但都没有公开。

![[Pasted image 20240319155931.png]]
## 2 Related Work 相关工作
CLIP 对比学习
After CLIP’s initial success, ==ALIGN== and ==BASIC== improved contrastive multimodal learning by **increasing the training set size and the batch size used for training**. ==LiT== also **increased training scale** and experimented with a combination of pre-trained image representations and contrastive fine-tuning to connect frozen image representations to text. ==Flamingo== introduced the first large vision-language model with **in-context learning**.
## 3 Collection Methodology
### 3.1 Dataset Assembly Pipeline
1. Feed in Common Crawl
2. Web page Filtering
   Parse the HTML IMG (image) tags from Common Crawl’s WAT metadata files, focus on images with an _alt-text_.
   After extracting the alt-text, we perform language detection using CLD3 with three possible outputs: English, another language, or no detected language.
   We stored the resulting data in a PostgreSQL server for processing in the next stages of the pipeline. We maintained about 500M image URLs in the server at all times.
3. Download Image-Text Pairs
   We downloaded the raw images from the parsed URLs with asynchronous requests using the Trio and Asks Python libraries.
4. Content Filtering
   删除了所有余弦相似度低于0.28的英语图像-文本对，以及所有其他相似度低于0.26的对
5. Store Data
CLD3: language detect
### 3.2 Safety During Collection
We computed CLIP embeddings to filter out such samples.
## 4 Dataset Composition
We release LAION-5B as the following three subsets:
- 2.32 billion English image-text pairs. (LAION-2B-en or LAION-2B)
- 2.26 billion image-text pairs from over 100 other languages.
- 1.27 billion samples where a language could not be clearly detected.

We provide metadata files in the Apache Parquet format that consist of the following attributes for each image-text pair:
- A 64-bit integer identifier
- The URL of the image.
- The text string.
- Height and width of the image.
- Cosine similarity between the text and image embeddings.
- The output from our NSFW and watermark detectors (one score between 0 and 1 each).
3% of images were detected as NSFW, which can be filtered out by a user with the NSFW tag.  
## 5 Experiments Validating LAION-5B

# 1. DATACOMP: In search of the next generation of multimodal datasets 
[[2304.14108] DataComp: In search of the next generation of multimodal datasets](https://arxiv.org/abs/2304.14108)
多模态数据筛选 Benchmark，同时给出了一些多模态数据筛选的 Baseline
Benchmark的网页：
[https://www.datacomp.ai/](https://www.datacomp.ai/)
其实就是筛选数据集 train 一个 CLIP，比较性能
Baseline:  
1. No Filtering
2. Random subsets
3. Basic Filtering (Caption Length or Image Size)
4. CLIP score and LAION filtering
5. Text-based filtering( select examples that contain text overlapping with ImageNet class names)
6. Image-based filtering(visual content overlaps with ImageNet classes.)
CLIP score filtering excels on most tasks 

# 2. Improving multimodal datasets with image captioning 
[[2307.10350] Improving Multimodal Datasets with Image Captioning](https://arxiv.org/abs/2307.10350)
![](https://lh7-us.googleusercontent.com/aqwzcg18oBHb2fxuDwiVdaFAKiI_8OHadBn3FNAZOW2-wpxxCv8-bHqgXgLUA5CgkyUv_1_WtsxN8K5UJv822MVICKLQ0A2Uhyjhf0m8CXtqgLIjLFDsLFEt-YfhksJP6ACWD5m4W7Gk7OGRS7xxgs4)
1. synthetic captions in improving caption quality for multimodal training, as well as certain capabilities of the resulting model (e.g., retrieval). 需要不仅提升数据的质量，比如用BLIP2重新生成caption，还需要raw data来提升数据的diversity。合成caption对于retrieval task非常有用。
2. Notably, we find that fine-tuning general purpose models towards the task of image captioning actually makes them less effective at producing good captions for CLIP training. 
3. Our experiments with various candidate pool sizes, ranging from 12.8M to 1.28B image-text pairs, show that including generated captions in the training data can be highly effective at small and medium scales. However, with larger data quantities, the diversity gap between model-generated and web-scraped text begin to hinder performance gains, and it becomes increasingly harder to obtain state-of-the-art ImageNet accuracy by just improving text supervision alone.

# 3. SIEVE: MULTIMODAL DATASET PRUNING USING IMAGE CAPTIONING MODELS 
[[2310.02110] SIEVE: Multimodal Dataset Pruning Using Image Captioning Models](https://arxiv.org/abs/2310.02110)
文章介绍了一种名为SIEVE的数据清洗方法,用于去除噪声图像-文本对数据。其主要思路是:
1. 首先在一个小型但是质量很高(图像和文本对齐良好)的图文对数据集上预训练一个图像字幕生成模型。这个数据集虽然小,但是其多样性和质量可以保证训练出一个优质的字幕生成模型。
2. 利用这个预训练的字幕生成模型,对待清洗的大规模噪声图像-文本对数据集中的每个图像生成一个"合成字幕"。
3. 将合成字幕与原始数据集中图像对应的真实文本进行比较,评估它们的对齐程度或相似性。如果合成字幕与真实文本差异很大,说明原始的图文对可能是噪声数据,对齐质量较差。
4. 根据上述对齐度评估,对原始数据集进行"修剪"(pruning),即去除那些合成字幕与真实文本差异大的噪声图文对,得到一个质量更高的子集。

# 4. ShareGPT4V: Improving Large Multi-Modal Models with Better Captions 
[[2311.12793] ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](https://arxiv.org/abs/2311.12793)
利用GPT-Vision生成高质量 Caption，最终生成更好的多模态模型

# 5. ALLAVA: HARNESSING GPT4V-SYNTHESIZED DATA FOR A LITE VISION-LANGUAGE MODEL 
[[2402.11684] ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model](https://arxiv.org/abs/2402.11684)
利用 GPT-Vision 生成高质量 Caption，以及 Q&A
# 7. Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters 
[[2403.02677] Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters](https://arxiv.org/abs/2403.02677)
定义了比 CLIP 筛选效果更好的 4 个筛选数据 Metrics
Image-Text Matching (ITM)
The fine-tuned MLM data filter can explicitly generate the ITM score on a scale of 100

Object Detail Fulfillment (ODF) 
the ODF metric focuses on evaluating whether the image caption provides detailed descriptions of objects that align with the image. 

CaptionTextQuality(CTQ) 
the CTQ metric focuses on evaluating the text quality of image caption based on the grammatical correctness, diversity of vocabulary (e.g., the range and uniqueness of words), fluency (e.g., smoothness and natural flow of sentences), readability, length, and structure. 

Semantic Understanding (SU) 
the SU metric focuses on determining if the image caption provides additional semantic information that is not readily apparent just from the image itself. Such auxiliary semantic information can be 1) the professions of persons in the image; 2) the locations, addresses, festivals, country names, city names; 3) the names or entities of buildings, people, bird species, animal breeds, car models, engines in the image; 4) the social relationships between the people in the image, i.e., lovers, parent, or child. 

# 8. Multimodal C4: AnOpen, Billion-scale Corpus of Images Interleaved with Text
[2304.06939.pdf (arxiv.org)](https://arxiv.org/pdf/2304.06939.pdf)
将纯文本的C4扩展为图像/文本C4
一般的数据集是由图像-文本对构成，但是图像文本之间的关系不止于此。
收集C4中文本对应的网页中的图像。
将图像与文本的匹配看作一个二部匹配，使用CLIP-ViT计算相似度。
特点是处理的数据对象是文本图像交错序列而非图像文本对及其匹配文本与图像的方式

# 9. OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents
[arxiv.org/pdf/2306.16527.pdf](https://arxiv.org/pdf/2306.16527.pdf)
构建了一个图像文本数据集，爬取网页然后清洗，没有什么额外的
算是一个比较标准的处理数据流程，相当于是给出了LLM-TAP里面数据处理过程的一个实例。

# 10. Variance Alignment Score: A Simple But Tough-to-Beat Data Selection Method for Multimodal Contrastive Learning
[2402.02055.pdf (arxiv.org)](https://arxiv.org/pdf/2402.02055.pdf)
认为此前的工作**failed to consider the selection strategy from data distribution perspective**.This disadvan tage becomes more significant with limited computational resources, as high-quality data are not always the most in formative or representative.

Specifically, we propose Variance Alignment Score (VAS) which aims to find the most informative subset of the train ing data whose (cross-)covariance between the image-text pairs most aligns with the (cross-)covariance of a reference distribution. 其中reference distribution可以是data pool中的，也可以考虑其余task-dependent dataset.

![](https://lh7-us.googleusercontent.com/cVSUZE7lC4epGk-j1lFbF-IxunWUJCjLQrM58zJLhcxmCbaXz4dthH5xe0PAlPWuwzcsXLPW7cfCwD51oLi7-FeZ6-AbTRsaDA35ELg2wTwTBkVDozHtCySKiD9rpu5JffN7uOv8g4VmV5nkEbwBNq8)

这个比较新，是24年二月份挂出来的，感觉比较重要
它里面性能最好的是VAS加上使用CLIP score做初步剪枝，可以考虑加上caption生成和不同的meric
这篇文章中提到的一些其他方法：

- using forward and reverse message passing over a constructed dataset graph.
  Maharana, A., Yadav, P., and Bansal, M. D2 pruning: Mes sage passing for balancing diversity and difficulty in data pruning. arXiv preprint arXiv:2310.07931, 2023.
- Another relevant strategy is using the external image-set as data prior proposed
  Gadre, S. Y., Ilharco, G., Fang, A., Hayase, J., Smyrnis, G., Nguyen, T., Marten, R., Wortsman, M., Ghosh, D., Zhang, J., et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023.
# 11. Is Cosine-Similarity of Embeddings Really About Similarity?
[2403.05440.pdf (arxiv.org)](https://arxiv.org/pdf/2403.05440.pdf)
这个文章感觉没啥用，偶然看到的，就是说用余弦相似度有些情况下可能并不能有效度量相似性。但是他的理论和实验都是在矩阵分解这个问题上进行的，所以感觉意义不大。
# 12. CiT: Curation in Training for Effective Vision-Language Data
[2301.02241.pdf (arxiv.org)](https://arxiv.org/pdf/2301.02241.pdf)
使用先验，一边训练一遍构建数据集，只注重数据质量，感觉不重要
# 13. Vision Instruction Tuning
[https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
这是 LLaVA 模型的论文，生成数据的方法是喂给 GPT Image 的 Caption 和 Bounding Box，先提供几个例子作为 seed，之后用上面的 feature 去 query 大模型给出回答。

感觉这篇文章可以拿来当入门级论文看看，里面的方法写得很清楚。并且也有开源的数据集和代码，可以参考。

Future Work:
1. 可以多定义Metrics，Metrics之间可能有耦合的关系存在。可以通过实验选出重要的Metrics
2. 训练模型做更好的Evaluation。比如图像分割，对于每个物体识别，要求生成的Caption包含物体，也就是Object Detail Fulfillment

Questions:  
DATACOMP是个好数据集，针对数据集从零开始训练CLIP模型来检验成果。但是我认为可以在其他模型进行测试，比如其他的大语言模型

## Abstract
仅语言的 GPT-4 如何生成多模态的指令微调数据集
## 1 Introduction
Contributions:
- Multimodal instruction-following data
- Large multimodal models
- Multimodal instruction-following benchmark
- Open-source

## 2 Related works
==the teacher-student distillation ideas==

Flamingo can be viewed as the GPT-3 moment in the multimodal domain, due to its strong
performance on zero-shot task transfer and in-context-learning.

instruction tuning and ==prompt tuning==

==human crowd-scouring==

## 3 GPT-assisted Visual Instruction Data Generation
For an image $X_v$ and its associated caption $X_c$ , it is natural to create a set of questions $X_q$ with the
intent to instruct the assistant to describe the image content. We prompt GPT-4 to curate such a list of questions (see details in Appendix).
Human: $X_q\ X_v$ 
Assistant: $X_c$ 

## 4
### 4.1 architecture
第二行错词