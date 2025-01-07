import shutil
import random
import os

def get_dataset_from_file(path:str, training:int=80, seed:int=42):
    
    # Set random seed
    random.seed(seed)
    
    # Get the name of all files in images folder inside path
    files = os.listdir(path + '/images')
    
    # Randomize the order of the files
    random.shuffle(files)
    
    # Select training data and validation data
    training_files = files[:int(len(files) * training / 100)]
    validation_files = files[int(len(files) * training / 100):]
    
    # Create dataset path
    training_path = "../dataset/training"
    validation_path = "../dataset/validation"
    
    # Move the files to the corresponding folders
    for file in training_files:
        
        # Move the images with the same name to training path
        shutil.move(path + '/images/' + file, training_path + '/images/' + file)
        
        # Change the extension of the file to .txt
        file = file.replace('.jpg', '.txt')
        
        # Move the labels with the same name to training path
        shutil.move(path + '/labels/' + file, training_path + '/labels/' + file)
        
    for file in validation_files:
        
        # Move the images with the same name to validation path
        shutil.move(path + '/images/' + file, validation_path + '/images/' + file)
        
        # Change the extension of the file to .txt
        file = file.replace('.jpg', '.txt')
        
        # Move the labels with the same name to validation path
        shutil.move(path + '/labels/' + file, validation_path + '/labels/' + file)
        
def clear():
    
    if os.path.exists("../dataset/training"):
        shutil.rmtree("../dataset/training")
    if os.path.exists("../dataset/validation"):
        shutil.rmtree("../dataset/validation")
    
    os.mkdir("../dataset/training")
    os.mkdir("../dataset/training/images")
    os.mkdir("../dataset/training/labels")
    os.mkdir("../dataset/validation")
    os.mkdir("../dataset/validation/images")
    os.mkdir("../dataset/validation/labels")