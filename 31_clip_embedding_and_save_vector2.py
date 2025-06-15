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

# -------------------------- 메인 처리 --------------------------
def main():
    # 1. dataset 경로 입력
    while True:
        root_dir = input("dataset 경로를 입력해주세요: ").strip()
        if root_dir and Path(root_dir).exists():
            break
        print("❌ 유효하지 않은 경로입니다. 다시 입력해주세요.")

    # 2. 이미지 타입 입력
    while True:
        img_type = input("사용할 이미지 타입 입력 (original / natural) [기본값: original]: ").strip().lower()
        if img_type == "":
            img_type = "original"
        if img_type in ["original", "natural"]:
            break
        print("❌ 잘못된 입력입니다. 'original' 또는 'natural' 중에서 선택해주세요.")
    base_dir = Path(root_dir) / f"{img_type}_images"
    if not base_dir.exists():
        print(f"❌ {base_dir} 디렉토리가 존재하지 않습니다.")
        sys.exit(1)

    # 3. 클래스 디렉토리 선택
    working_dir = {}
    all_classes = [d.name for d in base_dir.iterdir() if d.is_dir()]
    while True:
        class_all = input("모든 클래스를 임베딩할까요? (y/n): ").strip().lower()
        if class_all == "y":
            for cls in all_classes:
                working_dir[cls] = base_dir / cls
            break
        elif class_all == "n":
            while True:
                class_name = input("클래스 이름을 입력해주세요: ").strip()
                class_path = base_dir / class_name
                if class_path.exists():
                    working_dir[class_name] = class_path
                    break
                else:
                    print("❌ 해당 클래스 디렉토리가 존재하지 않습니다. 다시 입력해주세요.")
            break
        else:
            print("❌ y 또는 n만 입력해주세요.")

    # 4. Qdrant host 및 port 입력
    print("[Q1] Qdrant Host 및 Port 입력")
    host_input = input("호스트명을 입력해주세요 [기본값: localhost]: ").strip()
    host = host_input if host_input else "localhost"

    while True:
        port_input = input("포트 번호를 입력해주세요 (예: 6333) [기본값: 6333]: ").strip()
        try:
            port = int(port_input) if port_input else 6333
            client = QdrantClient(host=host, port=port)
            collections = client.get_collections().collections
            print("✅ Vector Database(Qdrant)에 정상적으로 연결되었습니다.")
            break
        except Exception as e:
            print(f"❌ Qdrant 연결 실패: {e}")
            print("다시 호스트명과 포트를 입력해주세요.")

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

    # 6. CLIP 모델 불러오기
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model, preprocess, device = load_clip_model()

    # 7. 이미지 임베딩 및 Qdrant 저장
    result_summary = {}
    for cls_name, cls_path in working_dir.items():
        image_files = [f for f in cls_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        count = 0
        print(f"\n📦 클래스 '{cls_name}' 처리 중... ({len(image_files)}장)")
        for img_path in tqdm(image_files, desc=f"  → 임베딩 중"):
            vector = embed_image_with_clip(img_path, model, preprocess, device)
            if vector is None:
                continue
            payload = {
                "class_name": cls_name,
                "is_delegate": False,
                "delegate_type": "average"
            }
            point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            client.upsert(collection_name=collection_name, points=[point])
            count += 1
        result_summary[cls_name] = count

    # 8. 결과 요약 출력
    print("\n✅ 모든 작업이 완료되었습니다. 클래스별 임베딩 수:")
    for cls, cnt in result_summary.items():
        print(f"  - {cls}: {cnt}개")

    print("\n🛑 서버 종료")
    sys.exit(0)

if __name__ == "__main__":
    main()