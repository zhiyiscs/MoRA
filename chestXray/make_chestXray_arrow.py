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
    CLASS = ['Lung', 'Opacity', 'Cardiomegaly', 'Calcinosis', 'Pulmonary Atelectasis', 'Calcified Granuloma',
        'Thoracic Vertebrae', 'Cicatrix', 'Spine', 'Markings', 'Pleural Effusion', 'Aorta', 'Diaphragm',
        'Density', 'Atherosclerosis', 'Deformity', 'Airspace Disease', 'Catheters, Indwelling', 'Scoliosis', 'Nodule']
    CLASS_DICT = {}
    for idx, genre in enumerate(CLASS):
        CLASS_DICT[genre] = idx   
        
    df_reports = pd.read_csv(root + 'indiana_reports.csv')
    df_projections = pd.read_csv(root + 'indiana_projections.csv')
    frontal_projections = df_projections[df_projections['projection'] == 'Frontal']
    combined_df = pd.merge(frontal_projections, df_reports, on='uid')
    cleaned_df = combined_df.dropna(subset=['findings', 'impression'], how='all')
    
    # Assuming 'cleaned_df' is your DataFrame after removing NaN values
    num_samples = cleaned_df.shape[0]
    # Create a random array of floats between 0 and 1
    random_floats = np.random.rand(num_samples)
    # Assign 'train' for 80%, 'val' for 10%, and 'test' for the remaining 10%
    split_values = np.where(random_floats < 0.8, 'train', np.where(random_floats < 0.9, 'val', 'test'))
    # Add the 'split' column to the DataFrame
    cleaned_df['split'] = split_values
    
    split_types = ['train','val','test']
    
    
    for split_type in split_types:
        data_list = []
        df = cleaned_df[cleaned_df['split'] == split_type]
        for index, sample in df.iterrows():
            image_path = root + "images/images_normalized/" + sample['filename']
            with open(image_path, "rb") as fp:
                    binary = fp.read()

            problems = sample['Problems'].split(";")
            if problems[0] == "normal":
                label = [0]*20
            else:
                label = [1 if p in problems else 0 for p in CLASS]

            # Convert NaN to empty strings
            findings = str(sample['findings']) if pd.notnull(sample['findings']) else ''
            impression = str(sample['impression']) if pd.notnull(sample['impression']) else ''
            # Now concatenate findings and impression
            text = [findings + impression]

            split = sample['split']

            data = (binary, text, label, split)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "text",
                "label",
                "split",
            ],
        )

        print(dataframe.shape)

        table = pa.Table.from_pandas(dataframe)

        with pa.OSFile(f"{root}/chestXray_{split_type}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)  


make_arrow(f'{args.root}')