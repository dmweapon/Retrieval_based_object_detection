import shutil
from pathlib import Path

# 디렉토리 설정
manual_dir = Path("dataset_manual")        # 수동 라벨링된 이미지들이 있는 디렉토리
dataset_dir = Path("dataset_jpeg")         # 원본 이미지들이 위치한 디렉토리

# 1. 복원할 이미지 유형 선택
print("어떤 유형의 이미지를 복원하시겠습니까?")
print("1) original")
print("2) natural")
choice = input("번호를 선택해주세요 (1 또는 2): ").strip()

img_type_map = {'1': 'original_images', '2': 'natural_images'}
img_type_subdir = img_type_map.get(choice)

while img_type_subdir is None:
    choice = input("잘못된 입력입니다. 1 또는 2 중에서 선택해주세요: ").strip()
    img_type_subdir = img_type_map.get(choice)

# 복원 통계
moved = 0
skipped = 0

# 수동 디렉토리 내부의 클래스 디렉토리 탐색
if not manual_dir.exists():
    print(f"⚠️ 수동 라벨링 디렉토리가 존재하지 않습니다: {manual_dir}")
    exit()

for class_dir in manual_dir.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    # 2. 정확한 복원 경로 설정
    target_dir = dataset_dir / img_type_subdir / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # 3. 해당 클래스 내의 모든 이미지 파일에 대해 복원 작업 수행
    if not any(class_dir.glob("*.jpg")):
        print(f"ℹ️ '{class_name}' 클래스 디렉토리에 복원할 이미지가 없습니다.")
        continue
        
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
            print(f"✅ 이미지 및 라벨 이동 완료: {img_path.name} → {target_dir}")
            moved += 1
        except Exception as e:
            print(f"❌ 이동 실패: {img_path.name}, 오류: {e}")
            skipped += 1

# 요약 출력
print("\n📦 복원 요약")
print(f"  복원된 이미지 수 : {moved}")
print(f"  건너뛴 항목 수   : {skipped}")