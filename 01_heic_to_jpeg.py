import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

# --- 초기 라이브러리 확인 및 안내 ---
try:
    from PIL import Image
    # pillow-heif가 설치되어 있으면 HEIF 오프너를 등록합니다.
    # (Ubuntu/Windows 권장)
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_LIB_TYPE = "pillow-heif"
except ImportError:
    try:
        # pyheif는 macOS에서 주로 사용됩니다.
        import pyheif
        from PIL import Image
        HEIF_LIB_TYPE = "pyheif"
    except ImportError:
        print("❌ 필요한 라이브러리가 설치되지 않았습니다.")
        print("이 스크립트를 실행하려면 HEIC/HEIF 이미지 처리 라이브러리가 필요합니다.")
        print("  - Ubuntu/Windows: pip install pillow-heif")
        print("  - macOS:          pip install pyheif")
        sys.exit(1)

# --- 상수 정의 ---
HEIC_ROOT = Path("dataset_heic")
JPEG_ROOT = Path("dataset_jpeg")

def contains_parentheses_with_number(file_name):
    """파일 이름에 '(숫자)' 패턴이 있는지 확인합니다 (예: 'IMG_001(1).HEIC')."""
    return re.search(r"\(\d+\)", file_name)

def convert_heic_to_jpeg(heic_path, jpeg_path):
    """
    단일 HEIC 파일을 JPEG로 변환합니다.
    설치된 라이브러리(pillow-heif 또는 pyheif)를 자동으로 사용합니다.
    """
    try:
        if HEIF_LIB_TYPE == "pillow-heif":
            # pillow-heif는 Image.open()으로 바로 처리 가능
            image = Image.open(heic_path)
        else: # "pyheif"
            heif_file = pyheif.read(heic_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        
        # JPEG로 저장
        image.save(jpeg_path, "JPEG")
        return True, None
    except Exception as e:
        return False, e

def main():
    """메인 실행 함수"""
    print("--- 📸 HEIC to JPEG 변환 스크립트 ---")
    print(f"사용 중인 HEIC 라이브러리: {HEIF_LIB_TYPE}")

    if not HEIC_ROOT.exists():
        print(f"❌ 원본 HEIC 디렉토리를 찾을 수 없습니다: '{HEIC_ROOT}'")
        sys.exit(1)

    # 1. 변환할 이미지 타입 선택 (original / natural)
    print(f"\n[1/3] '{HEIC_ROOT}' 디렉토리에서 변환할 이미지 타입을 선택하세요.")
    print("1) original_images")
    print("2) natural_images")
    
    img_type_choice = input("번호 선택 (기본값: 1): ") or "1"
    img_type_map = {"1": "original_images", "2": "natural_images"}
    img_type = img_type_map.get(img_type_choice)

    while not img_type:
        img_type_choice = input("❌ 잘못된 입력입니다. 1 또는 2 중에서 선택해주세요: ")
        img_type = img_type_map.get(img_type_choice)

    source_base_dir = HEIC_ROOT / img_type
    output_base_dir = JPEG_ROOT / img_type

    if not source_base_dir.exists():
        print(f"❌ 소스 디렉토리를 찾을 수 없습니다: {source_base_dir}")
        sys.exit(1)

    all_class_dirs = sorted([d for d in source_base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    if not all_class_dirs:
        print(f"⚠️ 처리할 클래스 디렉토리가 없습니다: {source_base_dir}")
        sys.exit(0)

    # 2. 변환할 클래스 선택 (전체 / 특정)
    print("\n[2/3] 모든 클래스를 변환하시겠습니까?")
    process_all = input(" (y/n, 기본값: y): ").lower().strip() or 'y'

    dirs_to_process = []
    if process_all == 'y':
        dirs_to_process = all_class_dirs
    else:
        print("\n변환할 클래스를 선택하세요.")
        for i, class_dir in enumerate(all_class_dirs):
            print(f"{i+1}) {class_dir.name}")
        
        while True:
            try:
                choice = int(input("번호 선택: ")) - 1
                if 0 <= choice < len(all_class_dirs):
                    dirs_to_process.append(all_class_dirs[choice])
                    break
                else:
                    print("❌ 잘못된 번호입니다.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")

    # 3. 변환 실행
    print("\n[3/3] 이미지 변환을 시작합니다.")
    
    skipped_files = []
    converted_count = 0
    error_count = 0

    # tqdm을 사용하여 전체 클래스 진행률 표시
    for class_dir in tqdm(dirs_to_process, desc="Overall Progress"):
        class_name = class_dir.name
        output_dir = output_base_dir / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        heic_files = list(class_dir.glob('*.HEIC')) + list(class_dir.glob('*.heic'))
        
        # tqdm을 사용하여 클래스 내 파일 진행률 표시
        for heic_path in tqdm(heic_files, desc=f"Processing {class_name}", leave=False):
            # 파일 이름에 (숫자)가 포함된 경우 건너뛰기
            if contains_parentheses_with_number(heic_path.name):
                skipped_files.append(str(heic_path))
                continue

            # 출력 파일 경로 설정 및 중복 변환 방지
            jpeg_path = output_dir / f"{heic_path.stem}.jpeg"
            if jpeg_path.exists():
                continue

            success, error = convert_heic_to_jpeg(heic_path, jpeg_path)
            if success:
                converted_count += 1
            else:
                error_count += 1
                tqdm.write(f"⚠️ 변환 실패: {heic_path} -> {error}")

    print("\n--- ✨ 변환 완료! ---")
    print(f"✅ 성공: {converted_count}개 파일")
    print(f"❌ 실패: {error_count}개 파일")
    print(f"⏭️ 건너뜀 (중복 의심 또는 이미 존재): {len(skipped_files)}개 파일")

    if skipped_files:
        print("\n--- 건너뛴 파일 목록 (중복 의심) ---")
        for f in skipped_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main() 