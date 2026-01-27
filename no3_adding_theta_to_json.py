import cv2
import json
import os
import numpy as np
from sklearn.decomposition import PCA


class Theta():
    def __init__(self, json_path, img_path, oral_type):
        self.json_path = json_path
        self.img_path = img_path
        self.oral_type = oral_type
        self.tooth_dots = []
        self.tooth_nums = []
        self.thetas = {}
        
        self.img = cv2.imread(self.img_path)
        self.img_h, self.img_w, _ = self.img.shape
        self.full_mask_canvas = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
    
    def get_angle_simple(self, dots):
        # 1. 치아 마스크의 모든 y좌표의 중간값을 구함
        y_coords = dots[:, 1]
        y_mid = np.median(y_coords)

        # 2. 위쪽 절반과 아래쪽 절반의 무게중심(Mean) 계산
        upper_half = dots[dots[:, 1] < y_mid]
        lower_half = dots[dots[:, 1] >= y_mid]

        if len(upper_half) == 0 or len(lower_half) == 0:
            return 0

        x1, y1 = np.mean(upper_half, axis=0) # 위쪽 중심
        x2, y2 = np.mean(lower_half, axis=0) # 아래쪽 중심

        # 3. 하악(lower)은 아래에서 위로, 상악(upper)은 위에서 아래로 벡터 방향 고정
        if self.oral_type == 'lower':
            # 잇몸(위) -> 치아끝(아래)
            dx, dy = x1 - x2, y1 - y2
        else:
            # 잇몸(아래) -> 치아끝(위)
            dx, dy = x2 - x1, y2 - y1

        return np.arctan2(dy, dx)

    def draw_tooth(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for teeth in data['tooth']:
            polygon = np.array(teeth['segmentation'], dtype=np.int32)

            mask_canvas = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            cv2.fillPoly(mask_canvas, [polygon], 255)
            cv2.fillPoly(self.full_mask_canvas, [polygon], 255)

            self.tooth_nums.append(teeth['teeth_num'])
            self.get_inner_coords(mask_canvas)
    
    def get_inner_coords(self, mask_canvas):
        y_coords, x_coords = np.where(mask_canvas == 255)
        dots = np.column_stack((x_coords, y_coords))

        self.tooth_dots.append(dots)
        
    def main(self):
        self.draw_tooth()

        if len(self.tooth_dots) > 0:
            all_masks = np.concatenate(self.tooth_dots)
            image_tilt_angle = self.get_angle_simple(all_masks)

            for i, mask_coords in enumerate(self.tooth_dots):
                tooth_angle = self.get_angle_simple(mask_coords) - image_tilt_angle

                tooth_angle = (tooth_angle + np.pi) % (2 * np.pi) - np.pi

                self.thetas[self.tooth_nums[i]] = {
                    'sin': float(np.sin(tooth_angle)), 
                    'cos': float(np.cos(tooth_angle))
                }

        self.add_theta_in_json()
    
    def add_theta_in_json(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for teeth in data['tooth']:
            teeth['sin'] = self.thetas[teeth['teeth_num']]['sin']
            teeth['cos'] = self.thetas[teeth['teeth_num']]['cos']
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f'saved {os.path.join(f"{json_folder}/", file)} with theta value')
            

#setting
if __name__ == "__main__":
    print("train/val and lower/upper")
    train_or_val, oral_type = input().split()

    json_folder = f'./dataset/{train_or_val}/labels_json/{oral_type}'
    image_folder = f'./dataset/{train_or_val}/images/{oral_type}'

    os.makedirs(f'./dataset/{train_or_val}/masking_images/{oral_type}_mask', exist_ok=True)

    for file in os.listdir(json_folder):
        if file.endswith('.json'):
            print(f'read {os.path.join(f"{json_folder}/", file)}')
            get_theta = Theta(f"{json_folder}/{file}", f"{image_folder}/{file[:-5]}.png", oral_type)
            get_theta.main()
    print("finished")