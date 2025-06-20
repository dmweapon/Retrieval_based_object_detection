import sys
import uuid
import hashlib
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

# -------------------- 대표 벡터 계산 함수들 --------------------
def compute_average(vectors):
    return np.mean(vectors, axis=0)

def compute_centroid(vectors):
    avg = compute_average(vectors)
    distances = np.linalg.norm(vectors - avg, axis=1)
    return vectors[np.argmin(distances)]

def compute_weighted_average(vectors, alpha=2.0):
    mean_vec = compute_average(vectors)
    weights = np.exp(-alpha * np.linalg.norm(vectors - mean_vec, axis=1))
    weights /= np.sum(weights)
    return np.sum(vectors * weights[:, np.newaxis], axis=0)

def compute_medoid(vectors):
    distances = np.linalg.norm(vectors[:, np.newaxis] - vectors, axis=2)
    total_distances = np.sum(distances, axis=1)
    return vectors[np.argmin(total_distances)]

# -------------------- 고유 ID 생성 함수 --------------------
def generate_delegate_id(payload, delegate_type):
    key = f"{payload.get('class_name')}::{delegate_type}::{payload.get('data_type')}::{payload.get('is_segmented')}::{payload.get('is_augmented')}"
    return hashlib.md5(key.encode()).hexdigest()

# -------------------- 대표 벡터 저장 함수 --------------------
def save_delegate_vector(client, collection_name, base_payload, vec, vec_type):
    payload = {
        **base_payload,
        "is_delegate": True,
        "delegate_type": vec_type,
    }
    point_id = generate_delegate_id(payload, vec_type)
    point = PointStruct(id=point_id, vector=vec.tolist(), payload=payload)
    client.upsert(collection_name=collection_name, points=[point])

# -------------------- 메인 함수 --------------------
def main():
    print("[Q0] Qdrant Host 및 Port 입력")
    host = input("Qdrant 호스트를 입력해주세요 [기본값: localhost]: ").strip() or "localhost"

    while True:
        try:
            port_input = input("Qdrant 포트 번호를 입력해주세요 [기본값: 6333]: ").strip()
            port = int(port_input) if port_input else 6333
            client = QdrantClient(host=host, port=port)
            collections = client.get_collections().collections
            break
        except Exception as e:
            print(f"❌ 연결 실패: {e}")

    while True:
        # [1] Collection 선택
        if not collections:
            print("❌ 생성된 collection이 없습니다.")
            sys.exit(1)

        print("\n[Q1] 작업할 collection 선택:")
        for idx, c in enumerate(collections):
            count = client.count(c.name, exact=True).count
            print(f"{idx+1}) {c.name} ({count}개)")
        while True:
            try:
                col_idx = int(input("→ collection 번호 입력: ")) - 1
                collection_name = collections[col_idx].name
                break
            except:
                print("❌ 잘못된 입력입니다.")

        # [2] 클래스 선택
        results = client.scroll(
            collection_name=collection_name,
            limit=9999,
            with_payload=True
        )[0]
        all_classes = sorted(set(
            p.payload.get("class_name")
            for p in results if p.payload.get("is_delegate") != True
        ))

        if not all_classes:
            print("❌ 해당 collection에 클래스 데이터가 없습니다.")
            continue

        print("\n[Q2] 대표벡터를 생성할 클래스 선택:")
        for i, name in enumerate(all_classes):
            print(f"{i+1}) {name}")
        while True:
            try:
                class_idx = int(input("→ 클래스 번호 입력: ")) - 1
                class_name = all_classes[class_idx]
                break
            except:
                print("❌ 잘못된 입력입니다.")

        # [3] 전처리 단계별 대표 벡터 생성
        preprocessing_conditions = {
            "pre_a": [
                FieldCondition(key="is_cropped", match=MatchValue(value=True)),
                FieldCondition(key="is_segmented", match=MatchValue(value=False)),
                FieldCondition(key="is_augmented", match=MatchValue(value=False)),
            ],
            "pre_b": [
                FieldCondition(key="is_segmented", match=MatchValue(value=True)),
                FieldCondition(key="is_augmented", match=MatchValue(value=False)),
            ],
            "pre_c": [
                FieldCondition(key="is_augmented", match=MatchValue(value=True)),
            ]
        }

        for pre_key, conditions in preprocessing_conditions.items():
            print(f"\n--- [작업] '{class_name}' 클래스의 '{pre_key}' 대표 벡터 생성 ---")
            
            # 해당 조건의 벡터 불러오기
            results, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="class_name", match=MatchValue(value=class_name)),
                    FieldCondition(key="is_delegate", match=MatchValue(value=False)),
                    *conditions
                ]),
                with_vectors=True, with_payload=True, limit=10000
            )

            if not results:
                print(f"  -> ❌ '{pre_key}' 조건에 해당하는 벡터가 없습니다. 건너뜁니다.")
                continue

            vectors_np = np.array([r.vector for r in results])
            base_payload = {
                k: results[0].payload.get(k)
                for k in ["data_type", "is_cropped", "is_segmented", "is_augmented", "class_name"]
                if k in results[0].payload
            }

            print(f"  -> 📦 총 {len(vectors_np)}개의 벡터 로드 완료")

            print("  -> 평균 벡터 저장 중...")
            save_delegate_vector(client, collection_name, base_payload, compute_average(vectors_np), "average")

            print("  -> 중심 벡터 저장 중...")
            save_delegate_vector(client, collection_name, base_payload, compute_centroid(vectors_np), "centroid")

            print("  -> 가중 평균 벡터 저장 중...")
            save_delegate_vector(client, collection_name, base_payload, compute_weighted_average(vectors_np), "weighted")

            print("  -> Medoid 벡터 저장 중...")
            save_delegate_vector(client, collection_name, base_payload, compute_medoid(vectors_np), "medoid")

            print("  -> ✅ 대표 벡터 저장 완료!")

        cont = input("\n➕ 다른 클래스 또는 collection에 대해 작업할까요? (y/n): ").strip().lower()
        if cont != "y":
            print("\n🛑 작업 종료")
            break

if __name__ == "__main__":
    main()

