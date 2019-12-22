#  Convolutional Neural Network (CNN)

## Instructions for training

To train a model using our dataset and our proposed model architecture do the following steps:

1. Download the provided dataset and unzip it

1. Copy the folder [dataset_LanePose](https://github.com/duckietown-ethz/proj-lfi-ml/data_sets/) to your home directory

1. Clone this repo and navigate in the terminal to proj-lfi-ml/cnn/

1. Run: `pyhton train_model.py [learning rate] [batch size] [epochs] [cuda device] [num_workers] [filename_notes]`  - for example: `pyhton train_model.py 0.05 16 200 cuda:0 4 adam_Huberloss`
    
1. The model is being trained now and you can see the progress of training via the prints in the terminal

1. When training is finished you will find the model, configuration data and data recorded concerning the training in the folder proj-lfi-ml/cnn/savedModels/

1. To evaluate the model run: `python evaluate_model.py folder` to evaluate all models which are saved in the folder savedModels/ or use `python evaluate_model.py model_name1 model_name2` which can be a list of the models (delimiter is spacing) you want to evaluate. Make sure to evaluate on a recording that you have not used before to train your models.
