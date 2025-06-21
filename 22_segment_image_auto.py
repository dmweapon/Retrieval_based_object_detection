# pip install torch torchvision torchaudio opencv-python pillow numpy
# pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install tqdm

import cv2
import os
from pathlib import Path
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import sys

def main():
    """
    사용자 입력을 받아 지정된 디렉토리의 모든 이미지에 대해 자동으로 세그멘테이션을 실행합니다.
    """
    # 1. 모델 설정
    print("--- 🤖 자동 세그멘테이션 스크립트 ---")
    print("\n[1/5] SAM 모델을 선택하세요.")
    print("1) vit_b | [VRAM] ~4GB")
    print("2) vit_l | [VRAM] ~6GB")
    print("3) vit_h | [VRAM] ~8GB")
    model_choice = input("선택 (1, 2, 3): ")
    model_map = {"1": "vit_b", "2": "vit_l", "3": "vit_h"}
    MODEL_TYPE = model_map.get(model_choice, "vit_b")

    CHECKPOINT_URLS = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    CHECKPOINT_PATH = Path("model") / Path(CHECKPOINT_URLS[MODEL_TYPE]).name
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_PATH.exists():
        import urllib.request
        print(f"⬇️ SAM 모델 다운로드 중: {CHECKPOINT_PATH.name}")
        urllib.request.urlretrieve(CHECKPOINT_URLS[MODEL_TYPE], CHECKPOINT_PATH)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH))
        sam.to(DEVICE)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"❌ 모델 로딩 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    # 2. 경로 설정
    print("\n[2/5] 원본 이미지 Root 디렉토리를 선택하세요.")
    dataset_dirs = sorted([d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("dataset_")])
    for i, d in enumerate(dataset_dirs):
        print(f"{i+1}) {d.name}")
    
    input_root = ""
    while True:
        try:
            dir_choice = int(input("번호 선택: ")) - 1
            if 0 <= dir_choice < len(dataset_dirs):
                input_root = dataset_dirs[dir_choice]
                break
            else:
                print("❌ 잘못된 번호입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")

    print(f"\n[3/5] 이미지 타입을 선택하세요.")
    print("1) original")
    print("2) natural")
    
    img_type_choice = input("번호 선택 (기본값: 1): ") or "1"
    img_type_map = {"1": "original", "2": "natural"}
    img_type_base = img_type_map.get(img_type_choice)

    while img_type_base is None:
        img_type_choice = input("잘못된 입력입니다. 1 또는 2 중에서 선택해주세요: ")
        img_type_base = img_type_map.get(img_type_choice)
        
    img_type = f"{img_type_base}_images"
    source_base_dir = Path(input_root) / img_type
    
    if not source_base_dir.exists():
        print(f"❌ 기본 경로를 찾을 수 없습니다: {source_base_dir}")
        sys.exit(1)
        
    all_class_dirs = sorted([d for d in source_base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    # 3. 클래스 선택
    print(f"\n[4/5] 모든 클래스를 세그먼트하시겠습니까?")
    process_all = input(" (y/n, 기본값: y): ").lower() or 'y'

    dirs_to_process = []
    if process_all == 'y':
        dirs_to_process = all_class_dirs
    else:
        print("\n처리할 클래스를 선택하세요.")
        for i, class_dir in enumerate(all_class_dirs):
            print(f"{i+1}) {class_dir.name}")
        
        while True:
            try:
                class_choice = int(input("번호 선택: ")) - 1
                if 0 <= class_choice < len(all_class_dirs):
                    dirs_to_process.append(all_class_dirs[class_choice])
                    break
                else:
                    print("❌ 잘못된 번호입니다.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")

    # 5. 세그멘테이션 실행
    print("\n[5/5] 자동 세그멘테이션을 시작합니다.")
    
    for class_dir in tqdm(dirs_to_process, desc="Overall Progress"):
        class_name = class_dir.name
        input_dir = source_base_dir / class_name
        output_dir = Path("dataset_segmented") / img_type / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n▶️ '{class_name}' 클래스 처리 중...")
        print(f"Source: {input_dir}")
        print(f"Target: {output_dir}")

        image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png")))
        
        if not image_paths:
            print(f"⚠️ 처리할 이미지가 없습니다: {input_dir}")
            continue

        for image_path in tqdm(image_paths, desc=f"Processing {class_name}", leave=False):
            try:
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    tqdm.write(f"⚠️ 이미지를 읽을 수 없습니다: {image_path.name}")
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                predictor.set_image(image_rgb)
                
                h, w, _ = image_rgb.shape
                input_point = np.array([[w / 2, h / 2]])
                input_label = np.array([1])

                masks, scores, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                
                best_mask_idx = np.argmax(scores)
                best_mask = masks[best_mask_idx]
                
                # 원본 이미지를 BGRA로 변환 후 알파 채널에 마스크 적용
                rgba_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
                rgba_image[:, :, 3] = best_mask.astype('uint8') * 255
                
                output_path = output_dir / f"{image_path.stem}.png"
                cv2.imwrite(str(output_path), rgba_image)

            except Exception as e:
                tqdm.write(f"❌ 처리 중 오류 발생 ({image_path.name}): {e}")

    print(f"\n✅ 완료!")

if __name__ == "__main__":
    main() 