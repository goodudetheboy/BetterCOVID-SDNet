import pandas as pd
import shutil
from pathlib import Path
import os


def preprocess_data_by_severity(label_dir: str, input_dir: str, output_dir: str) -> str:
    """
        label_dir: the directory of label for the severity of COVID case
        data_original_dir: the directory saving positive and negative data

        This function will make another directory called Processed where we will save folders for multi-class classification task
        It also returns the directory to access Process
    """
    severity = pd.read_csv(label_dir) #read label file
    #make two columns to list
    names_list = severity["Name"].values.tolist() 
    severity_list = severity["Severity"].values.tolist()

    #make Processed directory and 4 classes directories of images
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    moderate_dir = os.path.join(output_dir, "Moderate")
    mild_dir = os.path.join(output_dir, "Mild")
    severe_dir = os.path.join(output_dir, "Severe")
    normal_dir = os.path.join(output_dir, "Normal_PCR")
    neg_output_dir = os.path.join(output_dir, "Neg")

    Path(moderate_dir).mkdir(parents=True, exist_ok=True)
    Path(mild_dir).mkdir(parents=True, exist_ok=True)
    Path(severe_dir).mkdir(parents=True, exist_ok=True)
    Path(normal_dir).mkdir(parents=True, exist_ok=True)
    Path(neg_output_dir).mkdir(parents=True, exist_ok=True)

    positive_dir = os.path.join(input_dir, "P")

    #add image to corresponding class directory using label from csv file
    for i in range(len(names_list)):
        img_dir = os.path.join(positive_dir, f"{names_list[i]}.jpg")
        if severity_list[i] == "MODERATE":
            shutil.copy(img_dir, moderate_dir)
        elif severity_list[i] == "MILD":
            shutil.copy(img_dir, mild_dir)
        elif severity_list[i] == "SEVERE":
            if os.path.exists(img_dir):
                shutil.copy(img_dir, severe_dir)
        else:
            shutil.copy(img_dir, normal_dir)
    
    #make class directory for negative class
    negative_dir = os.path.join(input_dir, "N")
    for file in os.listdir(negative_dir):
        shutil.copy(os.path.join(negative_dir, file), neg_output_dir)

    return output_dir

dire = preprocess_data_by_severity("./severity.csv","./cleaned-data", './cleaned-data-processed')

print(f"Finished splitting images by severity, output folder in {dire}")
