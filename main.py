from ultralytics import YOLO
import numpy as np
import torch
import cv2
from no6_ready_classification_model import ToothClassification
from torchvision import transforms
from PIL import Image
import os

class Pipeline():
    def __init__(self, oral_type):
        self.best_pt = None
        self.oral_type = oral_type

    def set_seg_model(self, path):
        self.best_pt = YOLO(path)

    def run_seg(self, path):
        return self.best_pt.predict(source=path, save=False, device=0)
    
    def get_angle_simple(self, dots):
        '''
        크롭한 치아 이미지의 sincos를 구하는 영역
        '''

        if len(dots) == 0: return 0
        
        y_coords = dots[:, 1]
        y_mid = np.median(y_coords)

        upper_half = dots[dots[:, 1] < y_mid]
        lower_half = dots[dots[:, 1] >= y_mid]

        if len(upper_half) == 0 or len(lower_half) == 0:
            return 0

        y1, x1 = np.mean(upper_half, axis=0)
        y2, x2 = np.mean(lower_half, axis=0)

        if self.oral_type == 'upper':
            dx, dy = x2 - x1, y2 - y1 # 잇몸 -> 치아끝
        else:
            dx, dy = x1 - x2, y1 - y2 # 잇몸 -> 치아끝

        return np.arctan2(dy, dx)

    
    def run_pipeline(self, results, cls_model, device):
        cls_model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.oral_type == 'lower':
            label_map = {
                0:31, 1:32, 2:33, 3:34, 4:35, 5:36,
                6:41, 7:42, 8:43, 9:44, 10:45, 11:46
            }
        else:
            label_map = {
                0:11, 1:12, 2:13, 3:14, 4:15, 5:16,
                6:21, 7:22, 8:23, 9:24, 10:25, 11:26
            }

        for result in results: # result: val 구강 이미지
            img_bgr = result.orig_img
            h, w, _ = img_bgr.shape

            if result.masks is not None:
                all_masks = np.concatenate(result.masks.xy)
                image_tilt_angle = self.get_angle_simple(all_masks)

                for i, mask_coords in enumerate(result.masks.xy): # mask_coords: val 구강 이미지 내 한 마스크 좌표
                    tooth_angle = self.get_angle_simple(mask_coords) - image_tilt_angle
                    tooth_angle = (tooth_angle + np.pi) % (2 * np.pi) - np.pi

                    sincos = [float(np.sin(tooth_angle)), float(np.cos(tooth_angle))]
                    sincos_tensor = torch.tensor([sincos], dtype=torch.float32).to(device)
                    

                    pad = 10
                    x1 = np.min(mask_coords[:, 0])
                    x2 = np.max(mask_coords[:, 0])
                    y1 = np.min(mask_coords[:, 1])
                    y2 = np.max(mask_coords[:, 1])
                    
                    x1, y1 = int(max(0, x1-pad)), int(max(0, y1-pad))
                    x2, y2 = int(min(w, x2+pad)), int(min(h, y2+pad))


                    cropped_bgr = img_bgr[y1:y2, x1:x2]
                    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                    cropped = cv2.resize(cropped_rgb, (224,224))

                    img_pil = Image.fromarray(cropped)

                    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out_tooth, out_cavity = cls_model(img_tensor, sincos_tensor)
                        
                        pred_t_idx = torch.argmax(out_tooth, dim=1).item()
                        pred_c_idx = torch.argmax(out_cavity, dim=1).item()



                    tooth_number = label_map.get(pred_t_idx, "Unknown")
                    cavity_status = "cavity" if pred_c_idx == 1 else "normal"

                    if pred_c_idx: 
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 255)
                    
                    # 치아 바운딩 박스 생성 및 텍스트 area

                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    # print(f"tooth FDI no. {tooth_number} | {cavity_status}")
                    label_text = f"{tooth_number} | {cavity_status}"
                    cv2.putText(img_bgr, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                #생성된 시각화 이미지 저장
                save_path = f'./predict/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
                cv2.imwrite(os.path.join(save_path, f'{os.path.basename(result.path).split(".")[0]}.png'), img_bgr)
                print(f'image saved: {os.path.join(save_path, f'{os.path.basename(result.path).split(".")[0]}.png')}')


if __name__ == '__main__':
    '''
    lower!!!!
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('lower/upper (Default: lower)')
    oral_type = input()

    if oral_type != 'upper':
        oral_type = 'lower'

    pipe = Pipeline(oral_type)
    pipe.set_seg_model(f'./runs/segment/{oral_type}/weights/best.pt')
    # seg_predicted = pipe.run_seg(f'./dataset/val/images/{oral_type}/')

    cls_model = ToothClassification(num_classes=12).to(device)
    cls_model.load_state_dict(torch.load(f'./runs/cls/{oral_type}/best.pth'))

    seg_predicted = pipe.run_seg(f'./dataset/val/images/{oral_type}')
    pipe.run_pipeline(seg_predicted, cls_model, device)