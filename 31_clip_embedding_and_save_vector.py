# pip install torch torchvision tqdm pillow qdrant-client
# pip install transformers==4.40.1
# pip install git+https://github.com/openai/CLIP.git@main

import os
import sys
import time
from pathlib import Path
import clip
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Qdrant 연결 설정 함수
def connect_qdrant():
    print("[Qdrant] 호스트와 포트 정보를 입력해주세요")
    host = input("Qdrant host [기본값: localhost]: ").strip() or "localhost"
    while True:
        port_input = input("Qdrant port [기본값: 6333]: ").strip()
        try:
            port = int(port_input) if port_input else 6333
            client = QdrantClient(host=host, port=port)
            client.get_collections()
            print("✅ Vector Database(Qdrant)에 정상적으로 연결되었습니다.")
            return client
        except Exception as e:
            print(f"❌ 연결 실패: {e}\n다시 시도해주세요.\n")

# 사용자 입력 유틸
def select_with_number(prompt, options, allow_back=False):
    while True:
        print(prompt)
        for i, opt in enumerate(options, 1):
            print(f"{i}) {opt}")
        if allow_back:
            print("b) 처음 질문으로")
        choice = input("번호를 입력해주세요: ").strip().lower()
        if allow_back and choice == 'b':
            return 'back'
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("❌ 올바르지 않은 입력입니다. 다시 입력해주세요.\n")

# 모델 로드 함수
def load_clip_model():
    model_path = Path("model") / "ViT-B-32.pt"
    model_path.parent.mkdir(exist_ok=True)
    if model_path.exists():
        print("✅ 이미 저장되어있는 CLIP 모델을 사용합니다.")
    else:
        print("📦 해당 모델이 저장되어있지 않아 새로 다운로드합니다.")
    model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    return model, preprocess

# 이미지 임베딩 함수
def embed_images_in_dir(model, preprocess, class_dir, class_name):
    image_files = list(Path(class_dir).glob("*.png"))
    embeddings = []
    for i, img_path in enumerate(image_files, 1):
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0)
            with torch.no_grad():
                embedding = model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                embeddings.append((img_path.name, embedding.squeeze().tolist()))
            print(f"[{i}/{len(image_files)}] ✅ 임베딩 완료: {img_path.name}")
        except Exception as e:
            print(f"[{i}/{len(image_files)}] ❌ 오류: {img_path.name} ({e})")
    return embeddings

# Qdrant 저장 함수
def save_embeddings_to_qdrant(client, collection_name, class_name, embeddings):
    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={
                "class_name": class_name,
                "is_delegate": False,
                "delegate_type": "average"
            },
        ) for i, (name, vec) in enumerate(embeddings)
    ]
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ {len(points)}개의 벡터를 Qdrant에 저장했습니다.")

# 메인 루프
def main():
    client = connect_qdrant()

    while True:
        # [요청1] dataset 선택
        dataset_dir = select_with_number(
            "[1단계] 사용할 dataset을 선택해주세요:",
            ["dataset_cropped", "dataset_segmented", "dataset_augmented"]
        )
        if dataset_dir == 'back':
            continue
        root_dir = Path(dataset_dir)

        # [요청2] 이미지 타입 선택
        while True:
            img_type = select_with_number(
                "[2단계] 사용할 이미지 타입을 선택해주세요:",
                ["original", "natural"],
                allow_back=True
            )
            if img_type == 'back':
                break
            base_dir = root_dir / f"{img_type}_images"
            class_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
            if not class_dirs:
                print("❌ 준비된 클래스가 없습니다. (하위 디렉토리가 없습니다)\n")
                continue
            else:
                break
        if img_type == 'back':
            continue

        # [Q3] 전체 클래스 여부
        all_classes = input("[3단계] 모든 클래스를 임베딩할까요? (y/n): ").strip().lower()
        working_dirs = {}

        if all_classes == 'y':
            for d in class_dirs:
                working_dirs[d.name] = d
        else:
            selected_class = select_with_number(
                "[3단계] 클래스 하나를 선택해주세요:",
                [d.name for d in class_dirs],
                allow_back=True
            )
            if selected_class == 'back':
                continue
            working_dirs[selected_class] = base_dir / selected_class

        # [Q4] collection 선택
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        collection_name = select_with_number(
            "[4단계] 저장할 Collection을 선택해주세요:",
            collection_names,
            allow_back=True
        )
        if collection_name == 'back':
            continue

        # [Q5] 모델 선택
        print("[5단계] 사용할 임베딩 모델을 선택해주세요:")
        print("1) CLIP ViT-B/32")
        model_choice = input("모델 번호 입력 [기본값: 1]: ").strip()
        model_choice = model_choice if model_choice else "1"
        if model_choice != "1":
            print("❌ 현재는 CLIP ViT-B/32만 지원됩니다.")
            continue

        model, preprocess = load_clip_model()

        # [Q6] 임베딩 및 저장
        total_counts = {}
        for class_name, class_path in working_dirs.items():
            print(f"\n🚀 {class_name} 클래스의 이미지 임베딩을 시작합니다...")
            embeddings = embed_images_in_dir(model, preprocess, class_path, class_name)
            save_embeddings_to_qdrant(client, collection_name, class_name, embeddings)
            total_counts[class_name] = len(embeddings)

        # [Q7] 요약 출력
        print("\n📊 클래스별 임베딩 수:")
        for k, v in total_counts.items():
            print(f" - {k}: {v}개")

        # [Q8] 추가 작업 여부
        again = input("\n다른 클래스도 이어서 작업할까요? (y/n): ").strip().lower()
        if again != 'y':
            print("👋 작업을 종료합니다.")
            break

if __name__ == "__main__":
    main()