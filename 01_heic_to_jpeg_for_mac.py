import os
from PIL import Image, features
import pyheif
import re  # 정규 표현식 사용을 위해 추가

# 클래스 이름 입력 받기
class_name = input("작업할 클래스 이름을 입력하세요: ")

# HEIC 이미지가 있는 디렉토리 (사용자 입력에 따라 경로 설정)
image_dir = f"./dataset/original_images/{class_name}"  # 원본 이미지 디렉토리
output_dir = f"./dataset/converted_images/{class_name}"  # 변환된 이미지 디렉토리

# HEIC 이미지를 JPEG로 변환하는 함수
def convert_heic_to_jpeg(heic_path, output_dir):
    try:
        # HEIC 파일 읽기
        heif_file = pyheif.read(heic_path)

        # Pillow 이미지 객체로 변환
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )

        # 파일 이름 설정 (띄어쓰기를 언더바로 변경)
        base_name = os.path.basename(heic_path).replace(" ", "_")
        file_name = os.path.splitext(base_name)[0] + ".jpg"
        output_path = os.path.join(output_dir, file_name)

        # JPEG로 저장
        os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리 생성
        image.save(output_path, "JPEG")
        print(f"Converted: {heic_path} -> {output_path}")
    except Exception as e:
        print(f"Error converting {heic_path}: {e}")

# 파일 이름 검사 함수
def contains_parentheses_with_number(file_name):
    # 정규 표현식으로 "(숫자)" 패턴 탐지
    return re.search(r"\(\d+\)", file_name)

# HEIC 이미지 변환 실행
def convert_all_heic_to_jpeg(image_dir, output_dir):
    if not os.path.exists(image_dir):
        print(f"입력한 클래스 이름에 해당하는 디렉토리가 존재하지 않습니다: {image_dir}")
        return

    heic_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.heic')]
    total_files = len(heic_files)
    completed_files = 0
    skipped_files = 0

    for heic_file in heic_files:
        heic_path = os.path.join(image_dir, heic_file)

        # 파일 이름 검사
        if contains_parentheses_with_number(heic_file):
            # 파일 삭제 및 건너뛰기
            os.remove(heic_path)
            print(f"Skipped and Deleted: {heic_path}")
            skipped_files += 1
            continue

        # 변환 수행
        convert_heic_to_jpeg(heic_path, output_dir)

        # 진행 상태 출력
        completed_files += 1
        print(f"Progress: {completed_files}/{total_files}")

    print(f"{completed_files}개의 파일 변환 완료, {skipped_files}개의 파일 건너뜀.")

# Mac M1에서 Pillow와 pyheif 문제 해결을 위한 종속성 확인
try:
    if not features.check("webp"):
        raise RuntimeError("Pillow가 WebP 지원으로 빌드되지 않았습니다.")
    if not features.check("jpeg"):
        raise RuntimeError("Pillow가 JPEG 지원으로 빌드되지 않았습니다.")
    print("Pillow는 WebP 및 JPEG 변환을 지원합니다.")
except Exception as e:
    print(f"환경 구성 문제: {e}")
    print("Pillow와 pyheif 종속성을 확인하세요.")
    exit(1)

# 이미지 변환 시작
convert_all_heic_to_jpeg(image_dir, output_dir)