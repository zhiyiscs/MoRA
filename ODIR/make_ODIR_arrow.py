import pandas as pd
import numpy as np
import argparse
import json
import pyarrow as pa
import random
import os
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./datasets', type=str, help='Root of datasets')
args = parser.parse_args()

def make_arrow(root):

        
    # Replace the path with the actual path to your Excel file
    file_path = args.root + "/" + 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'

    # Reading the Excel file
    df_reports = pd.read_excel(file_path, engine='openpyxl')

    # Assuming 'cleaned_df' is your DataFrame after removing NaN values
    num_samples = df_reports.shape[0]
    print("num_samples:",num_samples)
    # Create a random array of floats between 0 and 1
    random_floats = np.random.rand(num_samples)
    # Assign 'train' for 80%, 'val' for 10%, and 'test' for the remaining 10%
    split_values = np.where(random_floats < 0.8, 'train', np.where(random_floats < 0.9, 'val', 'test'))
    # Add the 'split' column to the DataFrame
    df_reports['split'] = split_values
    
    split_types = ['train','val','test']

    
    for split_type in split_types:
        data_list = []
        df = df_reports[df_reports['split'] == split_type]
        for index, sample in df.iterrows():
            left_image_path = root + "/" + sample['Left-Fundus']
            right_image_path = root + "/" + sample['Right-Fundus']
            with open(left_image_path, "rb") as fp:
                    left_binary = fp.read()
            with open(right_image_path, "rb") as fp:
                    right_binary = fp.read()

            label = [int(sample["N"]),int(sample["D"]),int(sample["G"]),int(sample["C"]),int(sample["A"]),int(sample["H"]),int(sample["M"]),int(sample["O"])]


            # Convert NaN to empty strings
            Left_text = str(sample['Left-Diagnostic Keywords']) if pd.notnull(sample['Left-Diagnostic Keywords']) else ''
            Right_text = str(sample['Right-Diagnostic Keywords']) if pd.notnull(sample['Right-Diagnostic Keywords']) else ''
            # Now concatenate findings and impression
            text = ["Left_text:" + Left_text + "Right_text:" + Right_text]

            split = sample['split']

            data = (left_binary, right_binary, text, label, split)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "left_image",
                "right_image",
                "text",
                "label",
                "split",
            ],
        )

        print(dataframe.shape)

        table = pa.Table.from_pandas(dataframe)

        with pa.OSFile(f"{root}/ODIR_{split_type}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)  


make_arrow(f'{args.root}')