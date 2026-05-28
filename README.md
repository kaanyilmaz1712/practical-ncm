# practical-ncm

Python Implementation of Bachelor Thesis **"An Ablative Outlook on Practical Neural Causal Models Beyond Simplified Settings"**

Here we provide different approaches for various settings containing **binary, discrete and continuous variables** as well as complex causal structures.

## Quick Start

The notebook basics.ipynb provides a general overview on the core fundamentals of **SCMs and NCMs**, along with examples of how to use important functions mentioned in the thesis.

To run experiments navigate to the corresponding subfolder. For example, to run the binary experiments:

```bash
cd binary
python run.py
```

## Prerequisits

To be able to run the project, you need to install the latest machine learning libaries:

```bash
pip install numpy pandas torch matplotlib
```

## Structure

- `basics.ipynb` – Overview and examples  
- `binary/` – Binary NCM implementation 
- `discrete/` – Discrete extension
- `continuous/` – Continuous extension
