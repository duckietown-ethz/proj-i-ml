1.) copy data_LanePose to your home directory and unzip

2.) open a terminal and run: pyhton train_model.py [lr] [bs] [epoch] [cuda device] [num_workers] [filename_notes]
    for example: pyhton train_model.py 0.05 16 200 cuda:0 4 adam_Huberloss
    
3.) you will find the model and all other outputs in /savedModels/
