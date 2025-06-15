# pip install torch torchvision tqdm pillow qdrant-client
# pip install transformers==4.40.1
# pip install git+https://github.com/openai/CLIP.git@main

import os
import sys
import uuid
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import clip
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# -------------------------- 모델 로딩 함수 --------------------------
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\U0001F4E6 CLIP ViT-B/32 모델 로딩 중...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# -------------------------- 이미지 임베딩 함수 --------------------------
def embed_image_with_clip(image_path, model, preprocess, device):
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input).squeeze().cpu().numpy().tolist()
        return embedding
    except Exception as e:
        print(f"\u274C 이미지 임베딩 실패: {image_path} / {e}")
        return None

# -------------------------- 메인 처리 --------------------------
def main():
    # 0. Qdrant host 및 port 입력
    print("[Q0] Qdrant Host 및 Port 입력")
    host_input = input("Qdrant 호스트를 입력해주세요 [기본값: localhost]: ").strip()
    host = host_input if host_input else "localhost"

    while True:
        port_input = input("Qdrant 포트 번호를 입력해주세요 (예: 6333) [기본값: 6333]: ").strip()
        try:
            port = int(port_input) if port_input else 6333
            client = QdrantClient(host=host, port=port)
            collections = client.get_collections().collections
            print("\u2705 Qdrant에 연결되었습니다.")
            break
        except Exception as e:
            print(f"\u274C 연결 실패: {e}")
            print("다시 입력해주세요.")

    # 1. dataset 경로 선택
    dataset_options = {
        "1": "dataset_cropped",
        "2": "dataset_segmented",
        "3": "dataset_augmented"
    }
    while True:
        print("[Q1] 사용할 Dataset 디렉토리 선택")
        for k, v in dataset_options.items():
            print(f"{k}) {v}")
        selected = input("선택지를 입력하세요: ").strip()
        if selected in dataset_options:
            root_dir = dataset_options[selected]
            break
        print("\u274C 잘못된 입력입니다. 다시 입력해주세요.")

    # 2. 이미지 타입 선택
    img_type_options = {"1": "original", "2": "natural"}
    while True:
        print("[Q2] 이미지 타입 선택 (original / natural)")
        for k, v in img_type_options.items():
            print(f"{k}) {v}")
        img_type_input = input("선택지를 입력하세요: ").strip()
        if img_type_input in img_type_options:
            img_type = img_type_options[img_type_input]
            break
        print("\u274C 잘못된 입력입니다. 다시 입력해주세요.")

    # 3. 클래스 디렉토리 선택
    while True:
        base_dir = Path(root_dir) / f"{img_type}_images"
        if not base_dir.exists():
            print(f"\u274C {base_dir} 디렉토리가 존재하지 않습니다.")
            sys.exit(1)

        class_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            print("\u274C 준비된 클래스가 없습니다. (하위 디렉토리가 없습니다)")
            continue

        working_dir = {}
        all_classes = [d.name for d in class_dirs]
        while True:
            class_all = input("모든 클래스를 임베딩할까요? (y/n): ").strip().lower()
            if class_all == "y":
                for cls in all_classes:
                    working_dir[cls] = base_dir / cls
                break
            elif class_all == "n":
                print("[Q3] 클래스 목록")
                for idx, name in enumerate(all_classes):
                    print(f"{idx+1}) {name}")
                while True:
                    selected = input("클래스 번호를 입력해주세요: ").strip()
                    try:
                        idx = int(selected) - 1
                        class_name = all_classes[idx]
                        class_path = base_dir / class_name
                        working_dir[class_name] = class_path
                        break
                    except:
                        print("\u274C 잘못된 입력입니다. 다시 입력해주세요.")
                break
            else:
                print("\u274C y 또는 n만 입력해주세요.")
        break

    # 4. collection 선택
    while True:
        if not collections:
            print("\u274C 생성된 collection이 없습니다. 먼저 collection을 만들어주세요.")
            sys.exit(1)
        print("\u2705 현재 collection 목록:")
        for idx, col in enumerate(collections):
            print(f"{idx + 1}) {col.name}")
        col_idx = input("사용할 collection 번호 입력: ").strip()
        try:
            collection_name = collections[int(col_idx) - 1].name
            break
        except:
            print("\u274C 올바르지 않은 선택입니다. 번호를 다시 입력해주세요.")

    # 5. CLIP 모델 불러오기
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model, preprocess, device = load_clip_model()

    # 6. 이미지 임베딩 및 Qdrant 저장
    result_summary = {}
    for cls_name, cls_path in working_dir.items():
        image_files = [f for f in cls_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        count = 0
        print(f"\n\U0001F4E6 클래스 '{cls_name}' 처리 중... ({len(image_files)}장)")
        for img_path in tqdm(image_files, desc=f"  → 임베딩 중"):
            vector = embed_image_with_clip(img_path, model, preprocess, device)
            if vector is None:
                continue
            payload = {
                "class_name": cls_name,
                "is_delegate": False,
                "delegate_type": "average",
                "image_path": str(img_path)
            }
            point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            client.upsert(collection_name=collection_name, points=[point])
            count += 1
        result_summary[cls_name] = count

    # 7. 결과 요약 출력
    print("\n\u2705 모든 작업이 완료되었습니다. 클래스별 임베딩 수:")
    for cls, cnt in result_summary.items():
        print(f"  - {cls}: {cnt}개")

    print("\n\U0001F6D1 서버 종료")
    sys.exit(0)

if __name__ == "__main__":
    main()