# pip install torch torchvision tqdm pillow qdrant-client
# pip install transformers==4.40.1
# pip install git+https://github.com/openai/CLIP.git@main

# pip install torch torchvision tqdm pillow qdrant-client
# pip install transformers==4.40.1
# pip install git+https://github.com/openai/CLIP.git@main

import os
import sys
import uuid
import hashlib
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
    print("📦 CLIP ViT-B/32 모델 로딩 중...")
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
        print(f"❌ 이미지 임베딩 실패: {image_path} / {e}")
        return None

# -------------------------- 고유 ID 생성 함수 --------------------------
def generate_id_from_path(img_path: Path) -> str:
    return hashlib.md5(str(img_path.resolve()).encode()).hexdigest()

# -------------------------- 메인 처리 --------------------------
def main():
    print("[Q0] Qdrant Host 및 Port 입력")
    host_input = input("Qdrant 호스트를 입력해주세요 [기본값: localhost]: ").strip()
    host = host_input if host_input else "localhost"

    while True:
        port_input = input("Qdrant 포트 번호를 입력해주세요 (예: 6333) [기본값: 6333]: ").strip()
        try:
            port = int(port_input) if port_input else 6333
            client = QdrantClient(host=host, port=port)
            collections = client.get_collections().collections
            print("✅ Qdrant에 연결되었습니다.")
            break
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            print("다시 입력해주세요.")

    # 1. CLIP 모델 불러오기
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model, preprocess, device = load_clip_model()

    while True:
        # 2. dataset 경로 선택
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
            print("❌ 잘못된 입력입니다. 다시 입력해주세요.")

        # 3. 이미지 타입 선택
        img_type_options = {"1": "original", "2": "natural"}
        while True:
            print("[Q2] 이미지 타입 선택 (original / natural)")
            for k, v in img_type_options.items():
                print(f"{k}) {v}")
            img_type_input = input("선택지를 입력하세요: ").strip()
            if img_type_input in img_type_options:
                img_type = img_type_options[img_type_input]
                break
            print("❌ 잘못된 입력입니다. 다시 입력해주세요.")

        # 4. 클래스 디렉토리 선택
        while True:
            base_dir = Path(root_dir) / f"{img_type}_images"
            if not base_dir.exists():
                print(f"❌ {base_dir} 디렉토리가 존재하지 않습니다.")
                sys.exit(1)

            class_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
            if not class_dirs:
                print("❌ 준비된 클래스가 없습니다.")
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
                            print("❌ 잘못된 입력입니다. 다시 입력해주세요.")
                    break
                else:
                    print("❌ y 또는 n만 입력해주세요.")
            break

        # 5. collection 선택
        while True:
            if not collections:
                print("❌ 생성된 collection이 없습니다. 먼저 collection을 만들어주세요.")
                sys.exit(1)
            print("✅ 현재 collection 목록:")
            for idx, col in enumerate(collections):
                print(f"{idx + 1}) {col.name}")
            col_idx = input("사용할 collection 번호 입력: ").strip()
            try:
                collection_name = collections[int(col_idx) - 1].name
                break
            except:
                print("❌ 올바르지 않은 선택입니다. 번호를 다시 입력해주세요.")

        # 6. 이미지 임베딩 및 Qdrant 저장
        result_summary = {}

        is_segmented = root_dir == "dataset_segmented"
        is_augmented = root_dir == "dataset_augmented"

        for cls_name, cls_path in working_dir.items():
            image_files = [f for f in cls_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            count = 0
            print(f"\n📦 클래스 '{cls_name}' 처리 중... ({len(image_files)}장)")
            for img_path in tqdm(image_files, desc=f"  → 임베딩 중"):
                vector = embed_image_with_clip(img_path, model, preprocess, device)
                if vector is None:
                    continue

                payload = {
                    "data_type": f"{img_type}_images",     # original_images 또는 natural_images
                    "is_cropped": True,
                    "is_segmented": is_segmented,
                    "is_augmented": is_augmented,
                    "class_name": cls_name,
                    "is_delegate": False,
                    "delegate_type": None,
                    "img_path": str(img_path)
                }

                point_id = generate_id_from_path(img_path)  # 이미지 경로 기반 고유 ID
                point = PointStruct(id=point_id, vector=vector, payload=payload)
                client.upsert(collection_name=collection_name, points=[point])
                count += 1
            result_summary[cls_name] = count

        # 7. 결과 요약 출력
        print("\n✅ 모든 작업이 완료되었습니다. 클래스별 임베딩 수:")
        for cls, cnt in result_summary.items():
            print(f"  - {cls}: {cnt}개")

        # 8. 다음 작업 여부 확인
        cont = input("\n➕ 다른 작업을 이어서 하시겠습니까? (y/n): ").strip().lower()
        if cont != 'y':
            print("\n🛑 서버 종료")
            sys.exit(0)

if __name__ == "__main__":
    main()