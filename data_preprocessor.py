import pandas as pd
import shutil

# import sys
# from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument("-i", "--input", dest="filename",
#                     help="write report to FILE", metavar="FILE")
# parser.add_argument("-l", "--label", dest="filename",
#                     help="write report to FILE", metavar="FILE")
# parser.add_argument("-q", "--quiet",
#                     action="store_false", dest="verbose", default=True,
#                     help="don't print status messages to stdout")

# args = parser.parse_args()

# def main(args):
#    print("something")

# if __name__ == "__main__":
#    main(args)

def preprocess_data_by_severity(label_dir:str, data_original_dir:str) -> str:
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
    
    import os
    #make Processed directory and 4 classes directories of images
    os.mkdir(os.path.join(data_original_dir, "Processed"))
    dire = "./Data/Processed"
    os.mkdir(os.path.join(dire, "Moderate"))
    os.mkdir(os.path.join(dire, "Mild"))
    os.mkdir(os.path.join(dire, "Severe"))
    os.mkdir(os.path.join(dire, "Normal_PCR"))
    
    #add image to corresponding class directory using label from csv file
    for i in range(len(names_list)):
        if severity_list[i] == "MODERATE":
            shutil.copy(f"./Data/P/{names_list[i]}.jpg","./Data/Processed/Moderate/")
        elif severity_list[i] == "MILD":
            shutil.copy(f"./Data/P/{names_list[i]}.jpg","./Data/Processed/Mild/")
        elif severity_list[i] == "SEVERE":
            if os.path.exists(f"./Data/P/{names_list[i]}.jpg"):
                shutil.copy(f"./Data/P/{names_list[i]}.jpg","./Data/Processed/Severe/")
        else:
            shutil.copy(f"./Data/P/{names_list[i]}.jpg","./Data/Processed/Normal_PCR/")
    
    #make class directory for negative class
    shutil.copytree("./Data/N/","./Data/Processed/Neg/")
    return dire

dire = preprocess_data_by_severity("./severity.csv","./Data")

print(f"Finished splitting images by severity, output folder in {dire}")
