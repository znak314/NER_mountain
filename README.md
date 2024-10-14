# Task 1. NER model for mountain names identififcation

## Overview of the solution
The list of mountain names was extracted from the [List of Mountains in the World](https://www.kaggle.com/datasets/codefantasy/list-of-mountains-in-the-world) dataset.
A certain part of them was selected. For each of the names a sentence with this name was generated by using chatGPT. To build the model, I used the pre-trained [BERT](https://huggingface.co/docs/transformers/model_doc/bert) model and trained it on the created dataset. The balance between model size, accuracy, and training time was studied

## Project structure
* `Dataset creation` - a folder with all stages of dataset creation
* `model_creating.py` -  python script for model training. 
* `model_inference.py` -  python script for performing mountain names recognition
* `demo_notebook.ipynb` - notebook with a demonstration of the model's work 
* `Boost ideas.pdf` - pdf with a description of possible improvements
* `test_task_NER_dataset.csv` - created dataset, that will be used in training and evaluation of the model


## Model training

After training the Named Entity Recognition (NER) model using the `simpletransformers` library and a BERT-based architecture, we evaluated the model on a held-out test dataset (15% of the total data). Below are the key performance metrics of the trained model:

| Metric        | Value   |
|---------------|---------|
| **Eval_loss** | `0.037` |
| **Precision** | `0.941` |
| **Recall**    | `0.99`  |
| **F1-Score**  | `0.965` |

### Sample Output

The model was trained to identify  entities, such as beginning of the name of the mountain  `B-MOUNT`  and inner part of the name `I-MOUNT` and other entities - `O`. Post-processing script transform the answer into human-readable format.

`Example input:`
Mont Blanc dominates the Alps as its tallest peak.

| Word         | Predicted Label | Actual Label |
|--------------|-----------------|--------------|
| "Mont"    | `B-MOUNT`       | `B-MOUNT`      |
| "Blanc"   | `I-MOUNT`       | `I-MOUNT`      |
| "dominates"      | `O`             | `O`          |
| "the"       | `O`             | `O`          |
| "Alps"       | `B-MOUNT`       | `B-MOUNT`    |
| "as"       | `O`             | `O`          |
| "its"       | `O`             | `O`          |
| "tallest"       | `O`             | `O`          |
| "peak"       | `O`             | `O`          |

`Output after post-processing:`
**Mont Blanc** (Mountains) dominates the **Alps** (Mountains) as its tallest peak.

### Possible issues
Sometimes, when we receive a mountain name consisting of two words, we get two consecutive `B-MOUNTs` instead of the labels `B-MOUNT` and `I-MOUNT`. This reduces the overall precision, but such situations can be additionally handled in post-processing script.

## How to set-up a project?
### 1. **Clone the repository**
   Clone this repository to your local machine using:

   ```bash
   git clone https://github.com/znak314/NER_mountain.git
   ```
### 2. **Install all necessary libraries**
   Install all dependencies by using:

   ```bash
   pip install -r requirements.txt
   ```
### 3. **Download a model**
   Download a model folder from [google drive](https://drive.google.com/drive/folders/1O9SIv2yvpnQjdpcuzX2833dm_pBa-874?dmr=1&ec=wgc-drive-globalnav-goto) and place it with the same directory as `model_inference.py`. Your file structure should look like this:
   ```
   NER_mountain/
│
├── .idea/ 
├── model/
├── ...
└── model_inference.py  
 ```
Now you can easily use a model by: 
```
python model_inference.py 
```
