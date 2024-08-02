import os  
import cv2 
import time 
import json 
import pandas as pd 
import numpy as np 
from tqdm import tqdm  
import shutil


def create_dataframe(data_info): 

    root_dir = data_info['root_dir'] 
    raw_data = data_info['raw_data']
    label_data = data_info['label_data']
    design_id = data_info['design_id']
    raw_data_type = data_info['raw_data_type']
    label_type = data_info['label_type']
    target_root = data_info['target_root']

    scenarios = [] 
    curr_scenario = [] 
    image_fnames = [] 
    image_fpaths = []  
    label_fnames = [] 
    label_fpaths = [] 
    trg_image_fpaths = [] 
    trg_label_fpaths = [] 
    trg_seg_label_fpaths = [] 


    for scenario in os.listdir(os.path.join(root_dir, raw_data)): 
        scenarios.append(scenario) 

        print(f'start {scenario}')
        # find image files and label files 
        for image in tqdm(os.listdir(os.path.join(root_dir, raw_data, scenario, design_id, raw_data_type))): 

            curr_scenario.append(scenario) 

            image_fnames.append(image)
            image_fpaths.append(os.path.join(root_dir, raw_data, scenario, design_id, raw_data_type, image)) 

            if raw_data_type == 'Image_RGB': 
                img_format = '.png'
            else: 
                raise ValueError('unknown raw data type')

            if label_type == 'outputJson/PolyLine': 
                label_format = '.json'
            else: 
                raise ValueError('unknown label type')

            label_fname = image.replace(img_format, label_format)
            label_fnames.append(label_fname)
            label_fpaths.append(os.path.join(root_dir, label_data, scenario, design_id, label_type, label_fname))    

            trg_image_fpaths.append(os.path.join(target_root, 'images', scenario, image)) 
            trg_label_fpaths.append(os.path.join(target_root, 'polylines', scenario, label_fname)) 
            trg_seg_label_fpaths.append(os.path.join(target_root, 'seg_labels', scenario, image.replace(img_format, '.png')))

        
    df = pd.DataFrame({
        'scenario': curr_scenario, 
        'image_fname': image_fnames, 
        'label_fname': label_fnames, 
        'image_path': image_fpaths, 
        'label_path': label_fpaths, 
        'trg_image_path': trg_image_fpaths,
        'trg_label_path': trg_label_fpaths,
        'trg_seg_label_path': trg_seg_label_fpaths
    }) 

    return df 


def make_seg_label(label_path): 

    try: 
        with open(label_path, 'r') as f: 
            label = json.load(f)

        image_info = label['image'] 
        annotations = label['annotations'] 

        w, h = image_info['width'], image_info['height'] 

        mask = np.zeros((h, w), dtype=np.uint8) 

        for ann in annotations:
            line = ann['points'] 
            points = []
            for point in line:
                x, y = point.values() 
                if x <= 0 or y <= 0 or x >= w or y >= h:  
                    continue

                points.append([int(x), int(y)]) 
            points = np.array(points, np.int32) 
            cv2.polylines(mask, [points], isClosed=False, color=(1, 1, 1), thickness=12) 

    except:
        print(f'error {label_path}') 
        mask = None 

    return mask


def generate_dss_dataset(df, target_root):
    
    result_df = df.copy()

    # 1. make directories 
    for scenario in df['scenario'].unique(): 
        os.makedirs(os.path.join(target_root, 'images', scenario), exist_ok=True) 
        os.makedirs(os.path.join(target_root, 'polylines', scenario), exist_ok=True) 
        os.makedirs(os.path.join(target_root, 'seg_labels', scenario), exist_ok=True) 


    for row in tqdm(result_df.itertuples()): 
        index = row.Index
        image_path = row.image_path 
        label_path = row.label_path 
        trg_image_path = row.trg_image_path 
        trg_label_path = row.trg_label_path 
        trg_seg_label_path = row.trg_seg_label_path 

        # 2. make seg label
        seg_label = make_seg_label(label_path) 

        if seg_label is not None: 
            cv2.imwrite(trg_seg_label_path, seg_label) 
            result_df.loc[index, 'state'] = True
        else: 
            result_df.loc[index, 'state'] = False 
             

        try: 
            # 3. copy image files 
            shutil.copy(image_path, os.path.dirname(trg_image_path)) 

            # 4. copy label files
            shutil.copy(label_path, os.path.dirname(trg_label_path)) 

        except: 
            print(f'error {image_path}') 
            result_df.loc[index, 'state'] = False

    return  result_df 
        

def make_txt(result_df, target_root): 

    df = result_df[result_df['state'] == True]

    for scenario in df['scenario'].unique(): 
        scenario_df = df[df['scenario'] == scenario]
        for image_path in scenario_df['trg_image_path']: 
            with open(os.path.join(target_root, f'{scenario}.txt'), 'a') as f: 
                f.write(image_path + '\n')


def print_result(result_df): 

    total = len(result_df) 
    labeled_count = result_df[result_df['state'] == True]['state'].count() 

    print('='*50, 'Result', '='*50)
    print('Total: ', total)
    print(f"Success: {labeled_count}  (ratio: {labeled_count / total * 100} %)") 
    print()  

    # ratio of scenario 
    for scenario in result_df['scenario'].unique(): 
        scenario_count = len(result_df[result_df['scenario'] == scenario]) 
        print(f"{scenario}: {scenario_count}  (ratio: {scenario_count / total * 100} %)") 

    print('='*110) 
 


if __name__ == '__main__': 

    data_info = {
        'root_dir': '/root/DssDataset', 
        'raw_data': 'rawData/Car', 
        'label_data': 'labelingData/Car', 
        'design_id': 'Design0001', 
        'raw_data_type': 'Image_RGB', 
        'label_type': 'outputJson/PolyLine', 
        'target_root': '/root/lanedet-carla/data/DSS', 
    }

    df = create_dataframe(data_info) 
    result_df = generate_dss_dataset(df, data_info['target_root']) 
    make_txt(result_df, data_info['target_root'])
    print_result(result_df)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_df.to_csv(os.path.join(data_info['target_root'], f'df_result_{timestamp}.csv'), index=False)
