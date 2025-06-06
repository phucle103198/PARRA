# Assess Consistency FreeJAR data using Zero-Shot NLI

Welcome to the **Assess Consistency of FreeJAR Data using Zero-Shot NLI** repository! This project focuses on evaluating the consistency of FreeJAR data using a pre-trained Zero-Shot NLI model. We conduct a thorough analysis of the data, examining various aspects from the model's perspective. This includes a detailed look at descriptive data, individual hedonic classes, tester performance, and product-level evaluations.

<p align="center">
  <img src="./img/logo.png" width="300" />
</p>

---

## 🚀 How to Run

You have **two options** to run the notebook:

### ✅ Option 1: Use Google Colab (Recommended)

The easiest way is to run the notebook directly in Google Colab — no setup required.

Just click the badge below to open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gMZvE4MxNFR-SqTV44MNJXmYU2pGq8pe#scrollTo=ufhr8bfOKMVn)

> ✅ In this version, the model is automatically loaded from HuggingFace, so **you do not need to download or place it manually**.

---

### 💻 Option 2: Run Locally

If you prefer to run the notebook on your own machine (e.g., using VSCode or Jupyter):

1. **Clone the repository**:
   ```bash
   git clone https://github.com/phucle103198/PARRA.git
   cd PARRA
   ```

2. **(Optional)** Download the fine-tuned model from Google Drive:  
   [Download Model](https://drive.google.com/drive/folders/1c3t1HA1pPGGYvoz6qmQwpY7bWOCmQ10y)

   Place it in a folder named `model/`:
   ```
    PARRA/
      ├── data/
      ├── model/
      │   └── fine_tuned_deberta_nli_018/   # You need to download model or using another model from Huggingface
      ├── notebook/
      └── README.md
   ```

   > Alternatively, you can let the notebook load the model directly from HuggingFace.

3. **Create and activate a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate      # On Windows: .\env\Scripts\activate
   ```

4. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

   Or open the `.ipynb` file in VSCode with the Python and Jupyter extensions enabled.

---

## 📌 Features

- **Consistency Evaluation**: Assess both individual tester consistency and product evaluation consistency.
- **Model Confidence**: Analyze the model’s confidence in the labels selected by the testers.
- **Panel Performance**: Understand how different testers evaluate products and identify inconsistencies.


## Troubleshooting

If you encounter any issues, please check the following:
- Ensure that all dependencies are correctly installed by running `pip install -r requirements.txt` again.
- Verify that the model files are placed in the correct folder (`model/`).


## 📬 Contact

If you have any questions or suggestions, feel free to reach out to me via email: [phuc.letuan@hust.edu.vn](mailto:phuc.letuan@hust.edu.vn)
---

Happy analyzing! 🎉
