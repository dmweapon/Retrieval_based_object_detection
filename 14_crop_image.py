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
img_type = input("original 이미지를 작업할까요? natural 이미지를 작업할까요?: ").strip().lower()
while img_type not in ["original", "natural"]:
    img_type = input("잘못된 입력입니다. original 또는 natural 중 선택해주세요: ").strip().lower()

target_subdir = f"{img_type}_images"
img_root_dir = origin_dataset_dir / target_subdir

# 2. 전체 디렉토리 작업 여부 입력
all_dirs = input("모든 하위 디렉토리를 작업할까요? (y 또는 n): ").strip().lower()

if all_dirs == "y":
    class_dirs = sorted([d for d in img_root_dir.iterdir() if d.is_dir()])
else:
    class_name = input("작업할 클래스 이름을 입력하세요: ").strip()
    class_dir = img_root_dir / class_name
    if not class_dir.exists():
        print(f"[에러] 클래스 디렉토리 '{class_dir}'가 존재하지 않습니다.")
        sys.exit(1)
    class_dirs = [class_dir]

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