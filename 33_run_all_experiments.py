# pip install numpy qdrant-client scikit-learn tqdm

import os
import csv
import sys
import uuid
import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sklearn.metrics import accuracy_score

# -------------------- 하이퍼파라미터 --------------------

# 사용자 입력 받기
root_dir = input("데이터 루트 디렉토리를 입력해주세요 (예: dataset_segmented): ").strip()
original_image_dir = "original_images"
natural_image_dir = "natural_images"

# Qdrant 설정
qdrant_host = input("Qdrant 호스트 [기본값: localhost]: ").strip() or "localhost"
qdrant_port = input("Qdrant 포트 [기본값: 6333]: ").strip()
qdrant_port = int(qdrant_port) if qdrant_port else 6333

# 실험 설정
cases = ['case_a', 'case_b', 'case_c']
delegate_types = ['average', 'centroid', 'weighted', 'medoid']

# -------------------- 유틸리티 함수 --------------------

def get_output_csv_path():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    count = 1
    base_dir = Path("results")
    while True:
        subdir = base_dir / f"{today}-{count}"
        file_path = subdir / f"result_{today}-{count}.csv"
        if not file_path.exists():
            subdir.mkdir(parents=True, exist_ok=True)
            return file_path
        count += 1

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------- Qdrant Client 초기화 --------------------

client = QdrantClient(host=qdrant_host, port=qdrant_port)
collections = [c.name for c in client.get_collections().collections]

# natural 이미지에 대한 통계 저장용
class_image_count = defaultdict(int)

# -------------------- 실험 수행 --------------------

all_results = []

print("\n실험을 시작합니다. 총 12개의 실험을 순차적으로 실행합니다.\n")

for case in cases:
    collection_name = f"clip_embedding_{case}"
    if collection_name not in collections:
        print(f"\u274C {collection_name} 컬렉션이 존재하지 않습니다. 스킵합니다.")
        continue

    # natural 이미지 루프
    for class_name in sorted(os.listdir(f"{root_dir}/{natural_image_dir}")):
        class_dir = Path(root_dir) / natural_image_dir / class_name
        if not class_dir.is_dir():
            continue

        for img_file in tqdm(list(class_dir.glob("*.png")), desc=f"{case}/{class_name}"):
            # 테스트 이미지 임베딩 벡터 불러오기
            scroll_result, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="img_path", match=MatchValue(value=str(img_file))),
                        FieldCondition(key="is_delegate", match=MatchValue(value=False))
                    ]
                ),
                with_vectors=True,
                with_payload=True
            )
            if not scroll_result:
                continue

            test_vec = np.array(scroll_result[0].vector)
            class_image_count[class_name] += 1

            # 각 대표 벡터 타입별 유사도 비교
            for dtype in delegate_types:
                scroll_delegate, _ = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="delegate_type", match=MatchValue(value=dtype)),
                            FieldCondition(key="is_delegate", match=MatchValue(value=True))
                        ]
                    ),
                    with_vectors=True,
                    with_payload=True,
                    limit=9999
                )

                best_score = -1
                best_class = None

                for p in scroll_delegate:
                    ref_vec = np.array(p.vector)
                    score = cosine_similarity(test_vec, ref_vec)
                    if score > best_score:
                        best_score = score
                        best_class = p.payload.get("class_name")

                all_results.append({
                    "experiment_id": f"{case}_{dtype}",
                    "case": case,
                    "delegate_type": dtype,
                    "image_path": str(img_file),
                    "true_class": class_name,
                    "predicted_class": best_class,
                    "similarity_score": best_score
                })

# -------------------- CSV 저장 --------------------

output_path = get_output_csv_path()

with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment_id", "case", "delegate_type", "image_path",
        "true_class", "predicted_class", "similarity_score"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"\n✅ 실험 결과 저장 완료: {output_path}")

# -------------------- 요약 통계 출력 --------------------

print("\n📊 natural 이미지 사용 통계:")
print("클래스 수:", len(class_image_count))
for cname, count in sorted(class_image_count.items()):
    print(f" - {cname}: {count}장")