import os
import cv2
from pathlib import Path
import sys

# 설정
origin_dataset_dir = Path("dataset_jpeg")
cropped_dataset_dir = Path("dataset_cropped")
margin_ratio = 0.2
processed_count = 0  # ✅ 처리된 객체 이미지 수

# 1. 이미지 유형 입력
print("어떤 유형의 이미지를 크롭하시겠습니까?")
print("1) original")
print("2) natural")
choice = input("번호를 선택해주세요 (1 또는 2): ").strip()

img_type_map = {'1': 'original', '2': 'natural'}
img_type = img_type_map.get(choice)

while img_type is None:
    choice = input("잘못된 입력입니다. 1 또는 2 중에서 선택해주세요: ").strip()
    img_type = img_type_map.get(choice)

target_subdir = f"{img_type}_images"
img_root_dir = origin_dataset_dir / target_subdir

# 2. 전체 디렉토리 작업 여부 입력
all_class_dirs = sorted([d for d in img_root_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
class_dirs = []

all_dirs_choice = input("\n모든 하위 디렉토리를 작업할까요? (y 또는 n): ").strip().lower()

if all_dirs_choice == "y":
    class_dirs = all_class_dirs
    print(f"\n✅ 모든 {len(class_dirs)}개 클래스에 대해 크롭 작업을 시작합니다.")
elif all_dirs_choice == 'n':
    print("\n📄 작업할 클래스를 선택하세요:")
    for idx, dir_path in enumerate(all_class_dirs):
        print(f"  {idx + 1}: {dir_path.name}")
        
    while True:
        try:
            choice_str = input("클래스 번호를 입력하세요: ").strip()
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(all_class_dirs):
                selected_dir = all_class_dirs[choice_idx]
                class_dirs.append(selected_dir)
                print(f"\n✅ '{selected_dir.name}' 클래스에 대해서만 크롭 작업을 시작합니다.")
                break
            else:
                print(f"❌ 잘못된 번호입니다. 1에서 {len(all_class_dirs)} 사이의 숫자를 입력해주세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
else:
    print("❌ 유효하지 않은 입력입니다. 'y' 또는 'n'을 입력해주세요.")
    sys.exit(1)

# 바운딩 박스 + margin 크롭 함수
def crop_with_label(image_path, label_path, save_dir):
    global processed_count
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        # ✅ 바운딩 박스 없는 경우 스킵
        print(f"[스킵] 객체 없음: {image_path.name}")
        return

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        _, x_center, y_center, bbox_w, bbox_h = map(float, parts)
        x_center *= w
        y_center *= h
        bbox_w *= w
        bbox_h *= h

        x1 = int(x_center - bbox_w / 2)
        y1 = int(y_center - bbox_h / 2)
        x2 = int(x_center + bbox_w / 2)
        y2 = int(y_center + bbox_h / 2)

        # 마진 적용
        margin_x = int(bbox_w * margin_ratio)
        margin_y = int(bbox_h * margin_ratio)
        new_x1 = max(x1 - margin_x, 0)
        new_y1 = max(y1 - margin_y, 0)
        new_x2 = min(x2 + margin_x, w)
        new_y2 = min(y2 + margin_y, h)

        cropped = image[new_y1:new_y2, new_x1:new_x2]

        # ✅ 확장자 소문자 통일
        ext = image_path.suffix.lower()
        save_name = f"{image_path.stem}_cropped_obj{i}{ext}"
        save_path = save_dir / save_name
        cv2.imwrite(str(save_path), cropped)

        print(f"[저장 완료] {save_path}")
        processed_count += 1

# 3. 메인 처리 루프
for cls_dir in class_dirs:
    cls_name = cls_dir.name
    save_cls_dir = cropped_dataset_dir / target_subdir / cls_name
    save_cls_dir.mkdir(parents=True, exist_ok=True)

    for image_path in cls_dir.glob("*.jpg"):
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"[경고] 라벨 파일 없음: {label_path}")
            continue
        crop_with_label(image_path, label_path, save_cls_dir)

# ✅ 최종 출력
print(f"\n✅ 모든 이미지 크롭이 완료되었습니다.")
print(f"총 처리된 객체 수: {processed_count}개")