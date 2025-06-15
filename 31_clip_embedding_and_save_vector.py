import os
import sys
import clip
import torch
import qdrant_client
from PIL import Image
from pathlib import Path
from qdrant_client.http import models
from qdrant_client.models import PointStruct
from tqdm import tqdm


def get_valid_input(prompt, options):
    while True:
        print(prompt)
        for key, val in options.items():
            print(f"{key}) {val}")
        choice = input("선택지를 입력하세요: ").strip().lower()
        if choice in options:
            return options[choice]
        elif choice == 'b':
            return 'back'
        else:
            print("❌ 잘못된 입력입니다. 다시 시도해주세요.")


def connect_qdrant():
    print("[Qdrant 연결 설정]")
    host = input("Qdrant 호스트를 입력해주세요 [기본값: localhost]: ").strip() or "localhost"
    port_input = input("Qdrant 포트 번호를 입력해주세요 (예: 6333) [기본값: 6333]: ").strip()
    port = 6333 if port_input == "" else int(port_input)
    try:
        client = qdrant_client.QdrantClient(host=host, port=port)
        client.get_collections()
        print("✅ Vector Database(Qdrant)에 정상적으로 연결되었습니다.")
        return client
    except Exception as e:
        print("❌ Qdrant 연결에 실패했습니다:", e)
        sys.exit(1)


def list_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def select_directory(prompt, directories):
    while True:
        print(prompt)
        for i, d in enumerate(directories, 1):
            print(f"{i}) {d}")
        print("b) 이전 질문으로")
        choice = input("선택지를 입력하세요: ").strip().lower()
        if choice == 'b':
            return 'back'
        if choice.isdigit() and 1 <= int(choice) <= len(directories):
            return directories[int(choice) - 1]
        else:
            print("❌ 잘못된 입력입니다. 다시 시도해주세요.")


def get_embedding_model(model_dir):
    model_options = {'1': 'clip-ViT-B/32'}
    model_choice = get_valid_input("[Q6] 사용할 Embedding 모델 선택", model_options)
    model_name = model_choice if model_choice != 'back' else 'back'

    if model_name == 'clip-ViT-B/32':
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            print("✅ 이미 저장되어있는 모델을 사용합니다.")
        else:
            print("⬇️ 해당 모델이 저장되어있지 않아 새로 다운로드합니다.")
        model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        return model_name, model, preprocess
    return None, None, None


def save_embeddings_to_qdrant(client, collection_name, vectors, class_name, img_paths):
    points = []
    for i, vector in enumerate(vectors):
        point = PointStruct(
            id=i,
            vector=vector,
            payload={
                "class_name": class_name,
                "is_delegate": False,
                "delegate_type": "average",
                "img_path": img_paths[i]
            }
        )
        points.append(point)
    client.upsert(collection_name=collection_name, points=points)


def main():
    model_dir = "model"
    client = connect_qdrant()

    while True:
        # [Q1] dataset 경로 입력
        dataset_options = {'1': 'dataset_cropped', '2': 'dataset_segmented', '3': 'dataset_augmented'}
        root_dir = get_valid_input("[Q1] 사용할 Dataset 경로 선택", dataset_options)
        if root_dir == 'back':
            continue

        # [Q2] 이미지 타입 입력
        while True:
            img_type_options = {'1': 'original', '2': 'natural'}
            img_type = get_valid_input("[Q2] 사용할 이미지 타입 선택", img_type_options)
            if img_type == 'back':
                break
            base_dir = os.path.join(root_dir, f"{img_type}_images")
            if not os.path.exists(base_dir) or not list_subdirectories(base_dir):
                print("❌ 준비된 클래스가 없습니다. (하위 디렉토리가 없습니다)")
                continue
            else:
                break

        if img_type == 'back':
            continue

        # [Q3] 클래스 선택
        all_classes = list_subdirectories(base_dir)
        all_choice = input("모든 클래스를 임베딩할까요? (y/n): ").strip().lower()
        if all_choice == 'y':
            working_dirs = [os.path.join(base_dir, cls) for cls in all_classes]
        else:
            class_name = select_directory("[Q3] 임베딩할 클래스 선택", all_classes)
            if class_name == 'back':
                continue
            working_dirs = [os.path.join(base_dir, class_name)]

        # [Q4] collection 선택
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        collection_map = {str(i + 1): name for i, name in enumerate(collection_names)}
        print("[Q4] 현재 Qdrant에 저장된 Collections")
        for key, val in collection_map.items():
            print(f"{key}) {val}")
        selected_idx = input("사용할 Collection 번호를 입력하세요: ").strip()
        collection_name = collection_map.get(selected_idx)
        if not collection_name:
            print("❌ 잘못된 입력입니다. 프로그램을 종료합니다.")
            break

        # [Q5] 모델 선택
        model_name, model, preprocess = get_embedding_model(model_dir)
        if model_name == 'back':
            continue

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()

        # [Embedding 수행]
        total_summary = {}
        for dir_path in working_dirs:
            class_name = Path(dir_path).name
            image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".png")]
            embeddings = []
            img_paths = []
            for img_path in tqdm(image_files, desc=f"[임베딩] {class_name}"):
                try:
                    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = model.encode_image(image).squeeze().cpu().tolist()
                    embeddings.append(embedding)
                    img_paths.append(str(Path(img_path).relative_to(Path.cwd())))
                except Exception as e:
                    print(f"❌ 이미지 처리 오류: {img_path}, 에러: {e}")

            save_embeddings_to_qdrant(client, collection_name, embeddings, class_name, img_paths)
            total_summary[class_name] = len(embeddings)

        print("\n[작업 요약]")
        for cls, count in total_summary.items():
            print(f"클래스: {cls} - 임베딩 수: {count}")

        again = input("\n📌 다른 디렉토리도 계속 작업할까요? (y/n): ").strip().lower()
        if again != 'y':
            print("👋 프로그램을 종료합니다.")
            break


if __name__ == "__main__":
    main()