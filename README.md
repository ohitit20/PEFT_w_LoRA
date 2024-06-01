# Parameter-Efficient Fine-Tuning with LoRA

This repository contains two separate implementations of Parameter-Efficient Fine-Tuning using Low-Rank Adaptation (LoRA), implemented by Oğuz Kağan Hitit at Koç University in 2024. LoRA allows for efficient fine-tuning of large models by adapting only a small subset of their parameters.

## Running the Code
* To use the PEFT with LoRA that uses `peft` library, refer to `PEFT_run.ipynb`
* To use the custom PEFT with LoRA that does not use `peft` library, refer to the rest of the project. 
* To run the custom implementation in Google Colab, refer to the `run_in_colab.ipynb` notebook, which provides step-by-step instructions and necessary configurations.

## About the Implementation
LoRA's approach reduces computational overhead and preserves the original model's structure by focusing on key parameters, making it ideal for applications with limited computational resources.

