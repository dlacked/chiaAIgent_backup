import torch
from torch.utils.data import DataLoader
from torchvision import transforms # transform 정의를 위해 필요
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. train.py 파일에서 클래스들 임포트
# (파일 이름이 train.py라고 가정합니다. 다르면 파일명에 맞춰 수정하세요)
from no6_ready_classification_model import ToothClassification, ToothDataset

def validate_and_plot(model, loader, device, oral_type):
    model.eval()
    
    # 리스트 이름 통일: p(prediction), l(label)
    all_tn_p, all_tn_l = [], []
    all_c_p, all_c_l = [], []
    
    print(f"--- {oral_type} 검증 및 지표 산출 시작 ---")
    with torch.no_grad():
        for images, sincos, tn_labels, cs_labels in loader:
            images, sincos = images.to(device), sincos.to(device)
            tn_labels, cs_labels = tn_labels.to(device), cs_labels.to(device)
            
            # 모델 추론 (out_t: 치아번호, out_c: 충치여부)
            out_t, out_c = model(images, sincos)
            
            # 각 Head에서 최대값 인덱스 추출
            _, p_t = torch.max(out_t, 1)
            _, p_c = torch.max(out_c, 1)
            
            print(p_t, tn_labels, p_c, cs_labels)
            # 결과 저장
            all_tn_p.extend(p_t.cpu().numpy())
            all_tn_l.extend(tn_labels.cpu().numpy())
            all_c_p.extend(p_c.cpu().numpy())
            all_c_l.extend(cs_labels.cpu().numpy())

    # FDI 번호 라벨 설정
    if oral_type == 'lower':
        names = ['31','32','33','34','35','36','41','42','43','44','45','46']
    else:
        names = ['11','12','13','14','15','16','21','22','23','24','25','26']

    # --- 그래프 1: FDI 치아 번호 Confusion Matrix ---
    plt.figure(figsize=(12, 10))
    cm_t = confusion_matrix(all_tn_l, all_tn_p)
    sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title(f'FDI Tooth Numbering CM ({oral_type})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # --- 그래프 2: 충치 여부 Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    cm_c = confusion_matrix(all_c_l, all_c_p)
    sns.heatmap(cm_c, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Cavity'], yticklabels=['Normal', 'Cavity'])
    plt.title('Cavity Detection CM')
    plt.show()

    # --- 상세 리포트 출력 (변수명 수정 완료) ---
    print("\n" + "="*60)
    print(f"[{oral_type.upper()}] TOOTH NUMBERING CLASSIFICATION REPORT")
    print(classification_report(all_tn_l, all_tn_p, target_names=names))
    
    print("\n" + "="*60)
    print("CAVITY DETECTION CLASSIFICATION REPORT")
    # 여기서 all_c_l, all_c_p를 사용하여 에러를 방지했습니다.
    print(classification_report(all_c_l, all_c_p, target_names=['Normal', 'Cavity']))

if __name__ == "__main__":
    # 환경 설정
    oral_type = 'lower' # 발표용 데이터에 맞춰 선택
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 모델 객체 생성 및 가중치 로드
    model = ToothClassification(num_classes=12).to(device)
    model.load_state_dict(torch.load(f'./runs/cls/{oral_type}/best.pth', map_location=device))
    
    # 2. 검증용 데이터셋 설정 (Train 시와 동일한 전처리 적용)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 훈련 시 사용한 CSV 혹은 별도의 검증 CSV 경로 지정
    eval_dataset = ToothDataset(
        csv_path=f'./cropped_dataset/train/{oral_type}/csv/metadata.csv', 
        transform=val_transform,
        oral_type=oral_type
    )
    
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    # 3. 함수 실행
    validate_and_plot(model, eval_loader, device, oral_type)