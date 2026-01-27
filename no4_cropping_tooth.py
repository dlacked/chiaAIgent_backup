import os
import json
import numpy as np
import cv2
import pandas as pd

class CropTeeth():
    def __init__(self, json_path, image_path, output_image_path):
        '''
        json_path: 원본 json 경로
        img: 원본 이미지 그 자체
        output_image_path: 크롭된 이미지 저장 경로
        '''
        self.json_path = json_path
        self.img = cv2.imread(f'{image_path}.png')
        self.output_image_path = output_image_path

    def get_tooth_segmentation(self, teeth):
        teeth_num = teeth['teeth_num']
        teeth_sin = teeth['sin']
        teeth_cos = teeth['cos']
        is_cavity = teeth['decayed']
        coords = np.array(teeth['segmentation']).reshape(-1, 2)

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        return teeth_num, teeth_sin, teeth_cos, is_cavity, int(x_min), int(y_min), int(x_max), int(y_max)

    def crop_tooth(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for teeth in data['tooth']:
            teeth_num, teeth_sin, teeth_cos, is_cavity, x1, y1, x2, y2 = self.get_tooth_segmentation(teeth)

            # pad = 10
            
            # h, w, _ = self.img.shape
            # x1, y1 = max(0, x1-pad), max(0, y1-pad)
            # x2, y2 = min(w, x2+pad), min(h, y2+pad)

            # cropped = self.img[y1:y2, x1:x2]
            # cropped = cv2.resize(cropped, (224,224))

            # cv2.imwrite(f'{self.output_image_path}_{teeth_num}.png', cropped)
            # print(f"{self.output_image_path}_{teeth_num}.png is successfully saved")

            data_list.append({
                'img_dir': f'{self.output_image_path}_{teeth_num}.png',
                'teeth_num': teeth_num,
                'teeth_sin': teeth_sin,
                'teeth_cos': teeth_cos,
                'is_cavity': is_cavity
            })
            print(f"{self.output_image_path}_{teeth_num}.png csv data is successfully appended")

if __name__ == "__main__":
    print("train/val and lower/upper")
    train_or_val, lower_or_upper = input().split()
    json_path = f'./dataset/{train_or_val}/labels_json/{lower_or_upper}'
    image_path = f'./dataset/{train_or_val}/images/{lower_or_upper}'
    output_image_path = f'./cropped_dataset/{train_or_val}/{lower_or_upper}'
    output_csv_path = f'./cropped_dataset/{train_or_val}/{lower_or_upper}/csv'

    data_list = []

    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)

    for file in os.listdir(json_path):
        if file.endswith('.json'):
            crop_sys = CropTeeth(f'{json_path}/{file}', f'{image_path}/{file[:-5]}', f'{output_image_path}/{file[:-5]}')
            crop_sys.crop_tooth()
    
    df = pd.DataFrame(data_list)
    df.to_csv(f'{output_csv_path}/metadata.csv', index=False, encoding='utf-8')
    print("csv file is successfully created")
