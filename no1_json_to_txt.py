import json
import os
import cv2

def json_to_txt(json_path, txt_base_dir, img_base_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    file_name = os.path.splitext(os.path.basename(json_path))[0]
    txt_path = os.path.join(txt_base_dir, f"{file_name}.txt")

    
    img_path = os.path.join(img_base_dir, file_name + '.png')
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(txt_path, 'w') as f:
        for i in data['tooth']:

            # f.write(f'{i['teeth_num']}')
            f.write('0 ')
            
            for seg in i['segmentation']:
                x, y = seg[0], seg[1]
                # print(x, y)
                x /= w
                y /= h
                f.write(f'{x:.6f} {y:.6f} ')
            f.write('\n')
        print(f"created {txt_base_dir}/{file_name}.txt.")


#폴더 경로 오픈 로직
if __name__ == "__main__":
    print("train/val")
    train_or_val = input()
    json_folder = f'./dataset/data/{train_or_val}/labels_json'
    output_folder = f'./dataset/data/{train_or_val}/labels'
    image_folder = f'./dataset/data/{train_or_val}/images'

    json_folders = [f for f in os.listdir(json_folder) if os.path.isdir(os.path.join(json_folder, f))]

    print(json_folders)
    for path in json_folders:
        if not os.path.exists(f"{output_folder}/{path}"):
            print('폴더가 존재하지 않습니다.')
            os.makedirs(f"{output_folder}/{path}")
        else:
            print(f'{output_folder}/{path} 폴더가 존재하므로 건너뜁니다.')
            continue

        print('폴더가 존재하거나 폴더를 생성했습니다.')
        for file in os.listdir(f"{json_folder}/{path}"):
            if file.endswith('.json'):
                json_to_txt(os.path.join(f"{json_folder}/{path}", file), f"{output_folder}/{path}", f'{image_folder}/{path}')
                