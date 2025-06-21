import os
from PIL import Image
import pyheif
import re

# HEIC 이미지를 JPEG로 변환하는 함수
def convert_heic_to_jpeg(heic_path, output_dir):
    try:
        heif_file = pyheif.read(heic_path)
        image = Image.frombytes(
            mode=heif_file.mode,
            size=heif_file.size,
            data=heif_file.data,
            decoder_name="raw",
        )
        base_name = os.path.basename(heic_path).replace(" ", "_")
        file_name = os.path.splitext(base_name)[0] + ".jpg"
        output_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        image.save(output_path, "JPEG")
        print(f"Converted: {heic_path} -> {output_path}")
    except Exception as e:
        print(f"Error converting {heic_path}: {e}")

# 파일 이름 검사 함수
def contains_parentheses_with_number(file_name):
    return re.search(r"\(\d+\)", file_name)

# HEIC 이미지 변환 실행
def convert_all_heic_to_jpeg(image_dir, output_dir):
    if not os.path.exists(image_dir):
        print(f"입력한 디렉토리가 존재하지 않습니다: {image_dir}")
        return

    heic_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.heic')]
    total_files = len(heic_files)
    completed_files = 0
    skipped_files = 0

    for heic_file in heic_files:
        heic_path = os.path.join(image_dir, heic_file)
        if contains_parentheses_with_number(heic_file):
            os.remove(heic_path)
            print(f"Skipped and Deleted: {heic_path}")
            skipped_files += 1
            continue
        convert_heic_to_jpeg(heic_path, output_dir)
        completed_files += 1
        print(f"Progress: {completed_files}/{total_files}")

    print(f"{completed_files}개의 파일 변환 완료, {skipped_files}개의 파일 건너뜀.")

# --- 실행 흐름 시작 ---

# 변환 대상 데이터 유형 선택
image_type = input("어떤 이미지를 변환할까요? (original 또는 natural): ").strip().lower()
while image_type not in ['original', 'natural']:
    image_type = input("잘못된 입력입니다. original 또는 natural 중 선택해주세요: ").strip().lower()

base_input_dir = f"./dataset_heic/{image_type}_images"
base_output_dir = f"./dataset_jpeg/{image_type}_images"

# 모든 클래스 처리 여부 확인
convert_all = input("모든 클래스를 변환할까요? (y/n): ").strip().lower()

if convert_all == 'y':
    if not os.path.exists(base_input_dir):
        print(f"입력 디렉토리가 존재하지 않습니다: {base_input_dir}")
    else:
        class_names = [d for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))]
        for class_name in class_names:
            image_dir = os.path.join(base_input_dir, class_name)
            output_dir = os.path.join(base_output_dir, class_name)
            print(f"\n[{class_name}] 클래스 이미지 변환 시작:")
            convert_all_heic_to_jpeg(image_dir, output_dir)
else:
    class_name = input("변환할 클래스 이름을 입력하세요: ").strip()
    image_dir = os.path.join(base_input_dir, class_name)
    output_dir = os.path.join(base_output_dir, class_name)
    convert_all_heic_to_jpeg(image_dir, output_dir)