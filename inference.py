import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train import UNET
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

def major_voting(predictions):
    """From several masks, prepared by prediction of all CV models,
       by major voting creating final image in shape of 2D array
       is prepared

    Args:
        predictions (Numpy ND array): N x 2D array, according to the prediction
                                      from N models  

    Returns:
        Numpy 2D array: Final predicted mask
    """

    i = 0

    for prediction in predictions:

        if i == 0:
            final_predictions = prediction.argmax(2)
        else:
            final_predictions = np.dstack((final_predictions, prediction.argmax(2)))
        i = i+1

    final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=final_predictions)
    return final_predictions

def predict_ensemble(image, models):
    """Predicts mask by all the models, afterwards final prediction
       is created by using major voting fn.

    Args:
        image (pytorch tensor): image to score
        models (list): list of models

    Returns:
        Numpy ND array: Prediction of mask by all the models
    """

    predictions = []
    for model in models:             
        model.eval()
        prediction = model(image[None, ...])[0].permute(1,2,0).cpu().detach().numpy()
        predictions.append(prediction)
        del model
    return predictions
    

def load_models(path_models):
    """Loads models

    Args:
        path_models (string): folder path with stored models

    Returns:
        list: list with loaded models
    """
    models = []
    for model_path in os.listdir(path_models):
        model = torch.load(f"{path_models}\\{model_path}")              
        models.append(model)
        del model
    return models
   


def make_inference(path_files, models):
    """Goes through all images in path_files folder
       and predict the mask (classes) for them and consequently 
       computes percentages of areas that are covered with cloud, urban, non urban.


    Args:
        path_files (string): path of images that will be scored
        models (list): list with stored models

    Returns:
        list: List of 4 lists with time series of Urban pop, non urban pop, cloud pop
              and time
    """

    list_of_files = os.listdir(path_files)  # list all files in given folder

    nonurban_pop = []
    urban_pop = []
    cloud_pop = []
    time = []

    for filename in list_of_files: # go throught all files in folder

        name = path_files + filename
        image = np.load(name)["arr_0"]

        crop_pix = 336
        red = image[:crop_pix,:crop_pix,3]
        green = image[:crop_pix,:crop_pix,2]
        blue = image[:crop_pix,:crop_pix,1]
        nir = image[:crop_pix,:crop_pix,4]
    
        image = np.stack((red, green, blue, nir), axis = 2)
    
        image = image/255
        image = image.transpose(2,0,1)
        image = torch.tensor(image).cuda()
        image = image.float()

        predictions = predict_ensemble(image, models)
        prediction = major_voting(predictions)

        print(f'Percentage of nonurban:{(prediction==1).sum()/(336*336)}')
        print(f'Percentage of urban:{(prediction==0).sum()/(336*336)}')
        print(f'Percentage of clouds:{(prediction==2).sum()/(336*336)}')
    
        nonurban_pop.append((prediction==1).sum()/(336*336)) 
        urban_pop.append((prediction==0).sum()/(336*336)) 
        cloud_pop.append((prediction==2).sum()/(336*336)) 
        time.append(filename.split('.')[0][3:])
    
        print(filename)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,14))
        ax1.imshow(nir)
        ax2.imshow(prediction, vmax = 2)
        plt.show()
    
    return [nonurban_pop, urban_pop, cloud_pop, time]


def plot_index(nonurban_pop, urban_pop, cloud_pop, time):
    """Plots evolution of urban coverage.
       Excludes snapshots where cloudes are observed.

    Args:
        nonurban_pop (list): time series of nonurban area covered in images
        urban_pop (list): time series of urban area covered in images
        cloud_pop (list): time series of cloud area covered in images
        time (list): time series of times
    """

    ax1 = sns.set_style(style=None, rc=None )
    plt.figure(figsize = (15,5))

    sns.lineplot(x= [x[0] for x in zip(time, cloud_pop) if x[1]<0.005], 
                y=[x[0] for x in zip(urban_pop, cloud_pop) if x[1]<0.005], 
                marker='o', sort = False, ax=ax1)
    plt.xticks(rotation=45)
    plt.ylabel("Percentage of Area covered by urban")
    plt.show()


def plot_results(path_of_models, folders_to_infer):
    """Call make inference function for all input paths

    Args:
        path_of_models (string): path where models are stored
        folders_to_infer (list): list with the folders where we want to make inference
    """


    models = load_models(path_of_models)
    
    for path in folders_to_infer:

        print('Making inference to folder:')
        print(path)
        x = make_inference(path, models)
        plot_index(x[0], x[1], x[2], x[3])

        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    