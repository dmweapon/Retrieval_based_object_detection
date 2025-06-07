import shutil
from pathlib import Path

# 디렉토리 설정
manual_dir = Path("dataset_manual")        # 수동 라벨링된 이미지들이 있는 디렉토리
dataset_dir = Path("dataset_jpeg")         # 원본 이미지들이 위치한 디렉토리

# 복원 통계
moved = 0
skipped = 0

# 수동 디렉토리 내부의 클래스 디렉토리 탐색
for class_dir in manual_dir.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    target_dir = dataset_dir / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for img_path in class_dir.glob("*.jpg"):
        label_path = img_path.with_suffix(".txt")
        target_img = target_dir / img_path.name
        target_label = target_dir / label_path.name

        if not label_path.exists():
            print(f"⚠️ 라벨 파일 없음: {label_path.name} → 이미지 이동도 생략")
            skipped += 1
            continue

        try:
            shutil.move(str(img_path), str(target_img))
            shutil.move(str(label_path), str(target_label))
            print(f"✅ 이미지 및 라벨 이동 완료: {img_path.name} → {target_dir.name}")
            moved += 1
        except Exception as e:
            print(f"❌ 이동 실패: {img_path.name}, 오류: {e}")
            skipped += 1

# 요약 출력
print("\n📦 복원 요약")
print(f"  복원된 이미지 수 : {moved}")
print(f"  건너뛴 항목 수   : {skipped}")