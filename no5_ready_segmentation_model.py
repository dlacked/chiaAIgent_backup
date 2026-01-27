from ultralytics import YOLO
import os

'''
일단 lower seg만 돌리기
그러면 upper도 자동으로 구현하게 되어 있음
'''

class Segmentation:
    # def __init__(self, oral_type):
    def __init__(self):
        self.model = None
        # self.oral_type = oral_type

    def set_model(self, path="yolov8n-seg.pt"):
        self.model = YOLO(path)

        if path == "yolov8n-seg.pt":
            return False # 첫 학습
        return True # 사전 학습된 모델
    
    def train(self, trained):
        if trained == True:
            return
        
        self.model.train(
            # data=f'./data_{self.oral_type}.yaml',
            data='./data.yaml',
            epochs=40,
            imgsz=640,
            rect=True,
            batch=16,
            device=0,
            close_mosaic=0,

            project='runs/segment',
            # name=self.oral_type
            name='all'
        )

if __name__ == "__main__":
    # print('lower/upper Default: lower')
    # oral_type = input()
    # if oral_type != 'upper':
    #     oral_type = 'lower'
    seg_model = Segmentation()
    trained = seg_model.set_model() 
    # if(os.path.exists("./runs/")):
    #     print("segmentation model is already exists")
    #     trained = seg_model.set_model("./runs/segment/train2/weights/best.pt") # 사전 학습된 모델 호출
    # else:
    #     print("making a new segmentation model..")
    #     trained = seg_model.set_model()

    seg_model.train(trained) #set_model에 파라미터가 있는 경우 무시됨
    
