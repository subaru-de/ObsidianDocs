# [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

However, existing techniques often introduce inference latency (Houlsby et al., [2019](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib22); Rebuffi et al., [2017](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib47)) by extending model depth or reduce the model’s usable sequence length (Li & Liang, [2021](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib29); Lester et al., [2021](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib27); Hambardzumyan et al., [2020](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib20); Liu et al., [2021](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib34)) ([Section 3](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#S3 "3 Aren’t Existing Solutions Good Enough? ‣ LoRA: Low-Rank Adaptation of Large Language Models")).


![[Pasted image 20240302190816.png]]

hardware barrier to entry

Can be combined with many prior methods, such as prefix-tuning.

We follow the conventions set out by (Vaswani et al., [2017](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib52); Brown et al., [2020](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib7)) and use Adam (Loshchilov & Hutter, [2019](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib37); Kingma & Ba, [2017](https://ar5iv.labs.arxiv.org/html/2106.09685?_immersive_translate_auto_translate=1#bib.bib25)) for model optimization.

A neural network contains many dense layers which perform matrix multiplication.

Thus, if the pre-trained model is large (such as GPT-3 with |Φ0|≈175 Billion), storing and deploying many independent instances of fine-tuned models can be challenging, if at all feasible.
