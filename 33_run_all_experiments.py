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

# 사용자 입력 받기 - 테스트 그룹 선택
print("[Q0] 실험할 테스트 그룹을 선택하세요:")
testgroup_options = {
    "1": ("TestGroup1", "dataset_segmented"),
    "2": ("TestGroup2", "dataset_cropped")
}
for k, v in testgroup_options.items():
    print(f"{k}) {v[0]} ({v[1]})")
while True:
    selected = input("→ 번호 입력: ").strip()
    if selected in testgroup_options:
        test_group, root_dir = testgroup_options[selected]
        break
    print("❌ 잘못된 입력입니다. 다시 선택해주세요.")

original_image_dir = "original_images"
natural_image_dir = "natural_images"

# Qdrant 설정
qdrant_host = input("Qdrant 호스트 [기본값: localhost]: ").strip() or "localhost"
qdrant_port = input("Qdrant 포트 [기본값: 6333]: ").strip()
qdrant_port = int(qdrant_port) if qdrant_port else 6333

# Qdrant 클라이언트 초기화
client = QdrantClient(host=qdrant_host, port=qdrant_port)
collections = [c.name for c in client.get_collections().collections]
if not collections:
    print("❌ 사용 가능한 컬렉션이 없습니다.")
    sys.exit(1)

print("\n[Q1] 사용할 Qdrant 컬렉션을 선택하세요:")
for idx, name in enumerate(collections):
    print(f"{idx+1}) {name}")
while True:
    try:
        col_idx = int(input("→ 컬렉션 번호 입력: ")) - 1
        collection_name = collections[col_idx]
        break
    except:
        print("❌ 잘못된 입력입니다. 다시 입력해주세요.")

# 실험 설정
cases = ['pre_a', 'pre_b', 'pre_c']
delegate_types = ['average', 'centroid', 'weighted', 'medoid']

# -------------------- 유틸리티 함수 --------------------
def get_output_csv_path():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    count = 1
    result_dir = Path("results")
    while True:
        subdir = result_dir / f"{today}-{count}"
        file_path = subdir / f"result_{today}-{count}.csv"
        if not file_path.exists():
            subdir.mkdir(parents=True, exist_ok=True)
            return file_path
        count += 1

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------- 실험 수행 --------------------

all_results = []
class_image_count = defaultdict(int)
all_scores = defaultdict(list)  # 점수 저장을 위한 딕셔너리

print("\n실험을 시작합니다. 총 12개의 실험을 순차적으로 실행합니다.\n")

for case in cases:
    # natural 이미지 루프
    for class_name in sorted(os.listdir(f"{root_dir}/{natural_image_dir}")):
        class_dir = Path(root_dir) / natural_image_dir / class_name
        if not class_dir.is_dir():
            continue

        for img_file in tqdm(list(class_dir.glob("*.png")), desc=f"{case}/{class_name}"):
            # 테스트 이미지 벡터 조회
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
            test_payload = scroll_result[0].payload
            class_image_count[class_name] += 1

            # 각 대표 벡터 타입별 유사도 비교
            for dtype in delegate_types:
                # case에 따른 필터 조건 동적 생성
                filter_must = [
                    FieldCondition(key="delegate_type", match=MatchValue(value=dtype)),
                    FieldCondition(key="is_delegate", match=MatchValue(value=True)),
                    FieldCondition(key="class_name", match=MatchValue(value=class_name)),
                    FieldCondition(key="data_type", match=MatchValue(value=test_payload.get("data_type"))),
                ]
                if case == 'pre_a': # is_cropped=True, is_segmented=False, is_augmented=False
                    filter_must.extend([
                        FieldCondition(key="is_cropped", match=MatchValue(value=True)),
                        FieldCondition(key="is_segmented", match=MatchValue(value=False)),
                        FieldCondition(key="is_augmented", match=MatchValue(value=False)),
                    ])
                elif case == 'pre_b': # is_segmented=True
                     filter_must.extend([
                        FieldCondition(key="is_segmented", match=MatchValue(value=True)),
                        FieldCondition(key="is_augmented", match=MatchValue(value=False)),
                    ])
                elif case == 'pre_c': # is_augmented=True
                     filter_must.append(
                         FieldCondition(key="is_augmented", match=MatchValue(value=True))
                     )

                scroll_delegate, _ = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=filter_must),
                    with_vectors=True,
                    with_payload=True,
                    limit=1
                )

                if not scroll_delegate:
                    continue

                ref_vec = np.array(scroll_delegate[0].vector)
                best_score = cosine_similarity(test_vec, ref_vec)
                best_class = scroll_delegate[0].payload.get("class_name")

                all_results.append({
                    "experiment_id": f"{case}_{dtype}",
                    "case": case,
                    "delegate_type": dtype,
                    "image_path": str(img_file),
                    "true_class": class_name,
                    "predicted_class": best_class,
                    "similarity_score": best_score
                })

                # 점수를 딕셔너리에 임시 저장
                all_scores[f"{case}_{dtype}"].append(best_score)

# -------------------- CSV 및 NPY 저장 --------------------

output_path = get_output_csv_path()

# CSV 파일 저장
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment_id", "case", "delegate_type", "image_path",
        "true_class", "predicted_class", "similarity_score"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"\n✅ 실험 결과 저장 완료: {output_path}")

# NPY 파일 저장 및 요약 출력
print("\n🗂️ NPY 파일 저장 및 요약:")
score_dir = output_path.parent / "score_distribution"
score_dir.mkdir(parents=True, exist_ok=True)

for key, scores_list in sorted(all_scores.items()):
    score_path = score_dir / f"{key}_scores.npy"
    scores_np = np.array(scores_list)
    np.save(score_path, scores_np)

    print(f"\n- 파일: {score_path}")
    if len(scores_np) > 0:
        print(f"  저장된 점수 개수: {len(scores_np)}")
        print(f"  점수 미리보기 (최대 5개): {scores_np[:5]}")
        print(f"  평균 점수: {np.mean(scores_np):.4f}")
    else:
        print(f"  저장된 점수 없음")

# -------------------- 요약 통계 출력 --------------------

print("\n📊 natural 이미지 사용 통계:")
print("클래스 수:", len(class_image_count))
for cname, count in sorted(class_image_count.items()):
    print(f" - {cname}: {count}장")