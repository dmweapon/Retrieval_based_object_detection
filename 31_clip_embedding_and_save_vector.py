# pip install torch torchvision tqdm pillow qdrant-client
# pip install transformers==4.40.1
# pip install git+https://github.com/openai/CLIP.git@main

import os
from pathlib import Path
import torch
import clip
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, CollectionStatus

# -----------------------------
# Qdrant 호스트 및 포트 설정
# -----------------------------
def get_qdrant_connection():
    while True:
        host = input("[Q1] Qdrant 호스트를 입력해주세요 [기본값: localhost]: ").strip() or "localhost"
        port_input = input("[Q2] Qdrant 포트 번호를 입력해주세요 [기본값: 6333]: ").strip()
        port = 6333 if port_input == "" else int(port_input)

        try:
            client = QdrantClient(host=host, port=port, timeout=5.0)
            client.get_collections()  # 연결 테스트
            print("\n✅ Vector Database(Qdrant)에 정상적으로 연결되었습니다.")
            return client
        except Exception as e:
            print(f"\n❌ Qdrant 연결 실패: {e}\n다시 시도해주세요.\n")

# -----------------------------
# 유사도 거리 방식 선택
# -----------------------------
def select_distance():
    distance_options = {
        "1": Distance.COSINE,
        "2": Distance.EUCLID,
        "3": Distance.DOT,
        "4": Distance.MANHATTAN
    }
    while True:
        print("\n[Q3] 유사도 거리 방식을 선택하세요:")
        print("1) Cosine\n2) Euclid\n3) Dot\n4) Manhattan")
        choice = input("선택 (1-4): ").strip()
        if choice in distance_options:
            return distance_options[choice]
        print("잘못된 입력입니다. 다시 선택해주세요.\n")

# -----------------------------
# Collection 이름 입력
# -----------------------------
def input_collection_name():
    while True:
        name = input("\n[Q4] 사용할 Collection 이름을 입력해주세요: ").strip()
        if name:
            return name
        print("잘못된 입력입니다. 다시 입력해주세요.\n")

# -----------------------------
# Collection 생성 또는 존재 확인
# -----------------------------
def create_collection_if_needed(client, name, distance):
    collections = client.get_collections().collections
    if any(c.name == name for c in collections):
        print(f"\n✅ 이미 존재하는 Collection입니다: {name}")
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=512, distance=distance)
        )
        print(f"\n✅ Collection 생성 완료: {name}")

# -----------------------------
# dataset 디렉토리 선택
# -----------------------------
def select_dataset_dir():
    options = {"1": "dataset_cropped", "2": "dataset_segmented", "3": "dataset_augmented"}
    while True:
        print("\n[Q5] Dataset 디렉토리를 선택하세요:")
        for k, v in options.items():
            print(f"{k}) {v}")
        choice = input("선택: ").strip()
        if choice in options:
            return options[choice]
        print("잘못된 입력입니다. 다시 입력해주세요.\n")

# -----------------------------
# 이미지 타입 선택
# -----------------------------
def select_img_type():
    options = {"1": "original", "2": "natural"}
    while True:
        print("\n[Q6] 사용할 이미지 타입을 선택하세요:")
        for k, v in options.items():
            print(f"{k}) {v}")
        choice = input("선택: ").strip()
        if choice in options:
            return options[choice]
        print("잘못된 입력입니다. 다시 선택해주세요.\n")

# -----------------------------
# 클래스 디렉토리 선택
# -----------------------------
def select_class_name(base_dir):
    while True:
        subdirs = [d.name for d in Path(base_dir).iterdir() if d.is_dir()]
        if not subdirs:
            print("\n❌ 준비된 클래스가 없습니다. (하위 디렉토리가 없습니다)")
            return None

        print("\n[Q7] 클래스 디렉토리를 선택하세요:")
        for i, name in enumerate(subdirs, 1):
            print(f"{i}) {name}")
        print("b) 이전 질문으로")

        choice = input("선택: ").strip()
        if choice == "b":
            return "back"
        if choice.isdigit() and 1 <= int(choice) <= len(subdirs):
            return subdirs[int(choice) - 1]
        print("잘못된 입력입니다. 다시 입력해주세요.\n")

# -----------------------------
# 이미지 로드 및 임베딩
# -----------------------------
def embed_images(image_paths, model, preprocess):
    embeddings = []
    for path in image_paths:
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to("cuda")
            with torch.no_grad():
                embedding = model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.squeeze().cpu().numpy())
        except Exception as e:
            print(f"⚠️  {path} 임베딩 실패: {e}")
    return embeddings

# -----------------------------
# 벡터를 Qdrant에 저장
# -----------------------------
def save_embeddings_to_qdrant(embeddings, img_paths, collection_name, class_name):
    points = []
    for i, (embedding, img_path) in enumerate(zip(embeddings, img_paths)):
        relative_path = str(img_path.relative_to(Path.cwd()))
        payload = {
            "class_name": class_name,
            "is_delegate": False,
            "delegate_type": "average",
            "img_path": relative_path
        }
        point = PointStruct(id=f"{class_name}_{i}", vector=embedding, payload=payload)
        points.append(point)

    qdrant_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )

# -----------------------------
# 메인 실행 루프
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    qdrant_client = get_qdrant_connection()
    distance_metric = select_distance()
    collection_name = input_collection_name()
    create_collection_if_needed(qdrant_client, collection_name, distance_metric)

    while True:
        dataset_dir = select_dataset_dir()
        img_type = select_img_type()

        base_dir = Path(dataset_dir) / img_type
        class_name = select_class_name(base_dir)
        if class_name is None:
            continue
        if class_name == "back":
            continue

        class_path = base_dir / class_name
        image_paths = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
        if not image_paths:
            print("\n❌ 이미지가 존재하지 않습니다. 다른 클래스를 선택해주세요.")
            continue

        print(f"\n🔍 {len(image_paths)}개의 이미지를 임베딩 중...")
        vectors = embed_images(image_paths, model, preprocess)

        print("💾 벡터를 Qdrant에 저장 중...")
        save_embeddings_to_qdrant(vectors, image_paths, collection_name, class_name)
        print(f"✅ 완료: {class_name} 클래스의 {len(vectors)}개 벡터 저장 완료\n")

        again = input("계속해서 다른 클래스를 작업하시겠습니까? (y/n): ").strip().lower()
        if again != "y":
            print("\n👋 프로그램을 종료합니다.")
            break
