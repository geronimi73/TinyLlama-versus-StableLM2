# Fine-tune TinyLlama and Stable LM 2

Code for Medium story [TinyLlama 1.1B and Stable LM 2 1.6B: Another shallow dive into Performance and Fine-Tuning of Small Models](https://medium.com/@geronimo7/tinyllama-1-1b-and-stable-lm-2-1-6b-cc0051d79be9)

The code provided here is not a training framework or standalone trainer but should only illustrate the process of fine-tuning these model using the Hugging Face ecosystem.

## Notebooks

- `nb_prepare-OA2-dataset.ipynb`: Extract the top ranked conversation from the Open Assistant 2 dataset.
- `nb_finetune_StableLM2_OA2.ipynb`: Fine-tune Stable LM 2 1.6B on the Open Assistant 2 dataset.
- `nb_finetune_TinyLlama_OA2.ipynb`: Fine-tune TinyLlama 1.1B on the Open Assistant 2 dataset.
 
## `finetune_*.py`: Train on multiple GPUs with 🤗accelerate


