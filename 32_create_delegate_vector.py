import sys
import numpy as np
import uuid
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

# -------------------- 대표 벡터 계산 및 저장 --------------------
def save_delegate_vector(client, collection_name, class_name, vec, vec_type):
    payload = {
        "class_name": class_name,
        "is_delegate": True,
        "delegate_type": vec_type,
    }
    point = PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=payload)
    client.upsert(collection_name=collection_name, points=[point])

# -------------------- 메인 --------------------
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
            print(f"\u274C 연결 실패: {e}")

    while True:
        # [1] Collection 선택
        if not collections:
            print("\u274C 생성된 collection이 없습니다.")
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
                print("\u274C 잘못된 입력입니다.")

        # [2] 클래스 선택
        vectors = client.scroll(
            collection_name=collection_name,
            limit=9999,
            with_payload=True
        )[0]
        all_classes = sorted(set(p.payload.get("class_name") for p in vectors if p.payload.get("is_delegate") != True))

        if not all_classes:
            print("\u274C 해당 collection에 클래스 데이터가 없습니다.")
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
                print("\u274C 잘못된 입력입니다.")

        # [3] 해당 클래스의 벡터 불러오기
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="class_name", match=MatchValue(value=class_name)),
                    FieldCondition(key="is_delegate", match=MatchValue(value=False))
                ]
            ),
            with_vectors=True,
            with_payload=True,
            limit=9999
        )[0]
        if not results:
            print(f"\u274C 클래스 '{class_name}'에 해당하는 벡터가 없습니다.")
            continue

        vectors_np = np.array([r.vector for r in results])
        print(f"\n\U0001F4E6 총 {len(vectors_np)}개의 벡터 로드 완료")

        # [4] 대표 벡터 계산 및 저장
        print("→ 평균 벡터 저장 중...")
        save_delegate_vector(client, collection_name, class_name, compute_average(vectors_np), "average")

        print("→ 중심 벡터 저장 중...")
        save_delegate_vector(client, collection_name, class_name, compute_centroid(vectors_np), "centroid")

        print("→ 가중 평균 벡터 저장 중...")
        save_delegate_vector(client, collection_name, class_name, compute_weighted_average(vectors_np), "weighted")

        print("→ Medoid 벡터 저장 중...")
        save_delegate_vector(client, collection_name, class_name, compute_medoid(vectors_np), "medoid")

        print("\n✅ 대표 벡터 저장 완료!")

        # [5] 다음 작업 여부
        cont = input("\n➕ 다른 클래스 또는 collection에 대해 작업할까요? (y/n): ").strip().lower()
        if cont != "y":
            print("\n🛑 작업 종료")
            break

if __name__ == "__main__":
    main()