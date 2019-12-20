#  Convolutional Neural Network (CNN)

## Instructions for training

To train a model using our dataset and our proposed model architecture do the following steps:

  **1.) Download the provided dataset and unzip it**

  **2.) Copy the folder data_LanePose to your home directory**

  **3.) Clone this repo and navigate in the terminal to proj-lfi-ml/cnn/*

  **4.) Run: pyhton train_model.py [learning rate] [batch size] [epochs] [cuda device] [num_workers] [filename_notes]
        for example: pyhton train_model.py 0.05 16 200 cuda:0 4 adam_Huberloss**
    
  **5.) The model is being trained now and you can see the progress of training via the prints in the terminal**

  **6.) When training is finished you will find the model, configuration data and data recorded concerning the training in the folder proj
        lfi-ml/cnn/savedModels/**
