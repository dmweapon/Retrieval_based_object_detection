import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import time


# -----------------------------
# Qdrant 연결
# -----------------------------
def get_qdrant_connection():
    while True:
        host = input("Qdrant 호스트를 입력해주세요 [기본값: localhost]: ").strip() or "localhost"
        port_input = input("Qdrant 포트 번호를 입력해주세요 [기본값: 6333]: ").strip()
        port = 6333 if not port_input else int(port_input)
        try:
            client = QdrantClient(
                host=host,
                port=port,
                timeout=5.0,
                prefer_grpc=True
            )
            client.get_collections()
            print("✅ Qdrant 연결 성공 (gRPC 모드)")
            return client
        except Exception as e:
            print(f"❌ 연결 실패: {e}\n다시 입력해주세요.\n")


# -----------------------------
# Collection 선택
# -----------------------------
def select_collection(client: QdrantClient):
    collections = client.get_collections().collections
    if not collections:
        print("❌ 생성된 collection이 없습니다.")
        return None
    print("\n📦 현재 존재하는 collections:")
    for idx, col in enumerate(collections, 1):
        count = client.count(col.name, exact=True).count
        print(f"  {idx}) {col.name} ({count})")

    while True:
        choice = input("작업할 collection 번호 선택: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(collections):
            return collections[int(choice) - 1].name
        print("❌ 잘못된 선택입니다.\n")


# -----------------------------
# 클래스 목록 선택
# -----------------------------
def select_class(client: QdrantClient, collection_name: str):
    result = client.scroll(collection_name, with_payload=True, limit=100_000)
    class_names = sorted(set(p.payload["class_name"] for p in result[0]))

    if not class_names:
        print("❌ 해당 collection에 class가 존재하지 않습니다.")
        return None

    print("\n🎯 클래스 목록:")
    for idx, cls in enumerate(class_names, 1):
        print(f"  {idx}) {cls}")

    while True:
        choice = input("대표벡터를 생성할 클래스 번호 선택: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(class_names):
            return class_names[int(choice) - 1]
        print("❌ 잘못된 선택입니다.\n")


# -----------------------------
# 벡터 불러오기
# -----------------------------
def get_vectors_by_class(client: QdrantClient, collection_name: str, class_name: str):
    ### Qdrant Python v1.6.4 이상에서만 사용 가능한 문법 (pip show qdrant-client | grep Version)
    # result = client.scroll(
    #     collection_name,
    #     with_payload=True,
    #     limit=100_000,
    #     filter={"must": [{"key": "class_name", "match": {"value": class_name}}, {"key": "is_delegate", "match": {"value": False}}]}
    # )

    ### Qdrant Python 구버전 대응용 코드
    result = client.scroll(
        collection_name=collection_name,
        with_payload=True,
        limit=100_000,
        query_filter=Filter(
            must=[
                FieldCondition(key="class_name", match=MatchValue(value=class_name)),
                FieldCondition(key="is_delegate", match=MatchValue(value=False)),
            ]
        )
    )
    vectors = [np.array(p.vector) for p in result[0]]
    return np.stack(vectors) if vectors else None


# -----------------------------
# 대표 벡터 계산
# -----------------------------
def compute_average(vectors): return np.mean(vectors, axis=0)
def compute_centroid(vectors): return np.median(vectors, axis=0)
def compute_weighted_average(vectors, alpha=2.0):
    mean = np.mean(vectors, axis=0)
    weights = np.exp(-alpha * np.linalg.norm(vectors - mean, axis=1))
    weighted = np.average(vectors, axis=0, weights=weights)
    return weighted


def compute_medoid(vectors):
    dists = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2)
    return vectors[np.argmin(np.sum(dists, axis=1))]


# -----------------------------
# 대표 벡터 저장
# -----------------------------
def save_delegate_vector(client, collection_name, class_name, vector, delegate_type):
    payload = {
        "class_name": class_name,
        "is_delegate": True,
        "delegate_type": delegate_type,
        "timestamp": time.time()
    }
    id = f"{class_name}_delegate_{delegate_type}"
    client.upsert(collection_name=collection_name, wait=True, points=[PointStruct(id=id, vector=vector.tolist(), payload=payload)])


# -----------------------------
# 메인 실행
# -----------------------------
def main():
    client = get_qdrant_connection()

    while True:
        collection = select_collection(client)
        if not collection:
            break

        class_name = select_class(client, collection)
        if not class_name:
            continue

        vectors = get_vectors_by_class(client, collection, class_name)
        if vectors is None:
            print("❌ 해당 클래스의 벡터를 불러올 수 없습니다.")
            continue

        print(f"\n🔧 {class_name} 클래스에서 대표 벡터 생성 중...")

        avg = compute_average(vectors)
        cen = compute_centroid(vectors)
        wavg = compute_weighted_average(vectors, alpha=2.0)
        med = compute_medoid(vectors)

        save_delegate_vector(client, collection, class_name, avg, "average")
        save_delegate_vector(client, collection, class_name, cen, "centroid")
        save_delegate_vector(client, collection, class_name, wavg, "weighted")
        save_delegate_vector(client, collection, class_name, med, "medoid")

        print("✅ 대표 벡터 저장 완료: average / centroid / weighted / medoid")

        again = input("\n다른 클래스를 계속 작업하시겠습니까? (y/n): ").strip().lower()
        if again != "y":
            print("👋 종료합니다.")
            break


if __name__ == "__main__":
    main()