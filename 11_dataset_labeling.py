import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import shutil
import numpy as np

# 사용자 설정
dir_dataset_jpeg = Path("./dataset_jpeg")  # JPEG 이미지가 저장된 디렉토리
dir_model = Path("./model/yolov8s.pt")     # YOLOv8s 모델 경로 (.pt 파일)

# 이미지 유형 선택
print("어떤 유형의 이미지를 라벨링할까요?")
print("1) original")
print("2) natural")

choice = input("번호를 선택해주세요 (1 또는 2): ").strip()
image_type_map = {'1': 'original', '2': 'natural'}
image_type = image_type_map.get(choice)

while image_type is None:
    choice = input("잘못된 입력입니다. 1 또는 2 중에서 선택해주세요: ").strip()
    image_type = image_type_map.get(choice)

dir_dataset_sampled = dir_dataset_jpeg / f"{image_type}_images"

# 클래스 디렉토리 수집 (숨김 디렉토리 제외)
all_class_dirs = [d for d in dir_dataset_sampled.iterdir() if d.is_dir() and not d.name.startswith('.')]
if not all_class_dirs:
    print(f"❌ 디렉토리 없음: {dir_dataset_sampled} 하위에 클래스 디렉토리를 찾을 수 없습니다.")
    exit(1)

# 라벨링할 클래스 선택
class_dirs_to_process = []
label_all_choice = input("\n모든 클래스를 라벨링하시겠습니까? (y/n): ").strip().lower()

if label_all_choice == 'y':
    class_dirs_to_process = all_class_dirs
    print(f"\n✅ 모든 {len(all_class_dirs)}개 클래스에 대해 라벨링을 시작합니다.")
elif label_all_choice == 'n':
    all_class_names_for_selection = sorted([d.name for d in all_class_dirs])
    print("\n📄 클래스 이름 목록:")
    for idx, name in enumerate(all_class_names_for_selection):
        print(f"  {idx + 1}: {name}")

    while True:
        try:
            choice_str = input("라벨링할 클래스의 번호를 입력하세요: ").strip()
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(all_class_names_for_selection):
                selected_class_name = all_class_names_for_selection[choice_idx]
                selected_dir = next(d for d in all_class_dirs if d.name == selected_class_name)
                class_dirs_to_process.append(selected_dir)
                print(f"\n✅ '{selected_class_name}' 클래스에 대해서만 라벨링을 시작합니다.")
                break
            else:
                print(f"❌ 잘못된 번호입니다. 1에서 {len(all_class_names_for_selection)} 사이의 숫자를 입력해주세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except StopIteration:
            print("❌ 내부 오류: 선택한 클래스 디렉토리를 찾을 수 없습니다.")
            exit(1)
else:
    print("❌ 유효하지 않은 입력입니다. 'y' 또는 'n'을 입력해주세요.")
    exit(1)

# 각 클래스 디렉토리에 classes.txt 생성 여부
create_class_txt_in_each_dir = input("\n각 클래스 디렉토리에 classes.txt 파일을 생성할까요? (y/n): ").strip().lower()
if create_class_txt_in_each_dir not in ["y", "n"]:
    print("❌ 유효하지 않은 입력입니다. 'y' 또는 'n'을 입력해주세요.")
    exit(1)
create_class_txt_in_each_dir = (create_class_txt_in_each_dir == "y")

# 기존 라벨 덮어쓰기 여부
overwrite = input("기존 라벨링 파일이 있을 경우 덮어쓰시겠습니까? (y/n): ").strip().lower()
if overwrite not in ["y", "n"]:
    print("❌ 유효하지 않은 입력입니다. 'y' 또는 'n'을 입력해주세요.")
    exit(1)
overwrite = (overwrite == "y")
print(f"\n🔁 라벨 파일 덮어쓰기 설정: {'활성화' if overwrite else '비활성화'}")

# 최대 탐지 객체 수 입력
max_input = input("탐지할 최대 객체 수를 입력하세요 (예: 1, 3, 5 또는 a=모두 탐지): ").strip().lower()
if max_input == 'a':
    max_object_count = None  # 제한 없음
else:
    try:
        max_object_count = int(max_input)
        if max_object_count <= 0:
            raise ValueError
    except ValueError:
        print("❌ 유효하지 않은 입력입니다. 양의 정수 또는 'a'를 입력해주세요.")
        exit(1)

# YOLO 모델 로드
print("📦 YOLOv8s 모델 로드 중...")
try:
    model = YOLO(str(dir_model))
    print(f"✅ 모델 로드 완료: {dir_model.name}")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit(1)

# 클래스 이름 정렬 및 ID 매핑 (전체 클래스 기준)
all_class_names = sorted([d.name for d in all_class_dirs])
class_name_to_id = {name: idx for idx, name in enumerate(all_class_names)}

# classes.txt 저장 (전체 클래스 기준)
classes_txt_path = dir_dataset_sampled / "classes.txt"
try:
    with open(classes_txt_path, "w") as f:
        for name in all_class_names:
            f.write(f"{name}\n")
    print(f"\n📄 classes.txt 생성 완료: {classes_txt_path}")
except Exception as e:
    print(f"❌ classes.txt 저장 실패: {e}")
    exit(1)

# classes.txt 복사
if create_class_txt_in_each_dir:
    for class_dir in all_class_dirs:
        dst = class_dir / "classes.txt"
        try:
            shutil.copy(classes_txt_path, dst)
        except Exception as e:
            print(f"❌ 복사 실패: {dst}, 오류: {e}")

# 라벨링 시작
print("\n🚀 라벨링 작업 시작...\n")
total_images = 0
labeled_count = 0
skipped_existing = 0
skipped_no_object = 0
failed_images = 0
not_detected_images = []

for class_dir in class_dirs_to_process:
    class_name = class_dir.name
    class_id = class_name_to_id[class_name]

    for img_file in sorted(class_dir.glob("*.jpg")):
        total_images += 1
        label_file = img_file.with_suffix(".txt")

        if label_file.exists() and not overwrite:
            skipped_existing += 1
            print(f"⏩ 라벨 존재, 건너뜀: {label_file}")
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            failed_images += 1
            print(f"⚠️ 이미지 로드 실패: {img_file}")
            continue

        height, width = image.shape[:2]

        try:
            results = model.predict(source=image, conf=0.25, verbose=False)
        except Exception as e:
            failed_images += 1
            print(f"❌ 탐지 실패: {img_file.name}, 오류: {e}")
            continue

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            skipped_no_object += 1
            not_detected_images.append(str(img_file))
            print(f"❗️ 객체 미탐지: {img_file}")
            continue

        print(f"🔍 탐지 성공: {img_file.name}, 객체 수: {len(boxes)}")

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            if max_object_count is not None:
                top_idx = confs.argsort()[::-1][:max_object_count]
                xyxy = xyxy[top_idx]

            with open(label_file, "w") as f:
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            labeled_count += 1
            print(f"✅ 라벨 저장 완료: {label_file}")
        except Exception as e:
            failed_images += 1
            print(f"❌ 라벨 저장 실패: {label_file}, 오류: {e}")

# 요약 출력
print("\n📊 작업 요약")
print(f"총 이미지 수               : {total_images}")
print(f"  ↳ 라벨 생성 완료         : {labeled_count}")
print(f"  ↳ 라벨 존재하여 건너뜀    : {skipped_existing}")
print(f"  ↳ 객체 탐지 실패 (미탐지) : {skipped_no_object}")
print(f"  ↳ 로딩/탐지 실패          : {failed_images}")

if skipped_no_object > 0:
    show_list = input("\n🧐 객체 탐지 실패(미탐지) 이미지 목록을 출력할까요? (y/n): ").strip().lower()
    if show_list == "y":
        print("\n📂 객체 미탐지 이미지 목록:")
        for path in not_detected_images:
            print(f" - {path}")

print("\n🏁 전체 라벨링 작업 완료.")
