import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
from tqdm import tqdm

# 증강 파라미터
BRIGHTNESS_RATES = [1.2, 1.1, 1.05, 0.95, 0.9, 0.8]
ROTATION_ANGLES = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
NOISE_TYPES = ['gaussian', 'blur', 's&p']
NOISE_LEVELS = [0.01, 0.02, 0.03]

# 입력/출력 디렉토리
INPUT_ROOT = Path("dataset_segmented")
OUTPUT_ROOT = Path("dataset_augmented")

# 1. 이미지 유형 입력
img_type = input("original 이미지를 작업할까요? natural 이미지를 작업할까요?: ").strip().lower()
while img_type not in ["original", "natural"]:
    img_type = input("잘못된 입력입니다. original 또는 natural 중 선택해주세요: ").strip().lower()

target_subdir = f"{img_type}_images"
img_root_dir = INPUT_ROOT / target_subdir

# 2. 전체 디렉토리 작업 여부 입력
all_dirs = input("모든 하위 디렉토리를 작업할까요? (y 또는 n): ").strip().lower()
while all_dirs not in ["y", "n"]:
    all_dirs = input("잘못된 입력입니다. y 또는 n 중 선택해주세요: ").strip().lower()

if all_dirs == "y":
    class_dirs = sorted([d for d in img_root_dir.iterdir() if d.is_dir()])
else:
    class_name = input("작업할 클래스 이름을 입력하세요: ").strip()
    class_dir = img_root_dir / class_name
    if not class_dir.exists():
        print(f"[에러] 클래스 디렉토리 '{class_dir}'가 존재하지 않습니다.")
        sys.exit(1)
    class_dirs = [class_dir]

# --- 증강 함수 ---
def add_gaussian_noise(img, amount):
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 25, np_img.shape) * amount
    noisy = np.clip(np_img + noise, 0, 255)
    return Image.fromarray(noisy.astype(np.uint8))

def add_blur(img, amount):
    radius = int(2 * amount)
    return img.filter(ImageFilter.GaussianBlur(radius))

def add_salt_and_pepper(img, amount):
    np_img = np.array(img)
    total = np_img.size // np_img.shape[2]
    num_noise = int(amount * total)
    for _ in range(num_noise):
        y = random.randint(0, np_img.shape[0]-1)
        x = random.randint(0, np_img.shape[1]-1)
        if random.random() < 0.5:
            np_img[y, x] = 0
        else:
            np_img[y, x] = 255
    return Image.fromarray(np_img)

def augment_and_save(img_path, save_dir):
    try:
        img = Image.open(img_path).convert('RGBA')
    except Exception as e:
        print(f"❌ 이미지 열기 실패: {img_path} - {e}")
        return

    basename = img_path.stem
    ext = img_path.suffix.lower()

    # 밝기 조정
    for rate in BRIGHTNESS_RATES:
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(rate)
        bright_img.save(save_dir / f"{basename}_brightness_{int((rate-1)*100):+d}{ext}")

    # 회전
    for angle in ROTATION_ANGLES:
        rot_img = img.rotate(angle, expand=True, fillcolor=(0,0,0,0))
        rot_img.save(save_dir / f"{basename}_rot{angle}{ext}")

    # 노이즈 (각 타입별, 강도별)
    for noise_type in NOISE_TYPES:
        for level in NOISE_LEVELS:
            if noise_type == 'gaussian':
                noisy_img = add_gaussian_noise(img, level)
            elif noise_type == 'blur':
                noisy_img = add_blur(img, int(level * 100))
            elif noise_type == 's&p':
                noisy_img = add_salt_and_pepper(img, level)
            else:
                continue
            noisy_img.save(save_dir / f"{basename}_{noise_type}noise_{int(level*100)}{ext}")

# --- 리팩토링된 증강 프로세스 ---
def process_all():
    for class_dir in tqdm(class_dirs, desc="클래스 디렉토리"):
        out_class_dir = OUTPUT_ROOT / target_subdir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted([f for f in class_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        for img_file in tqdm(img_files, desc=f"{class_dir.name}", leave=False):
            augment_and_save(img_file, out_class_dir)

# 실행부
if __name__ == "__main__":
    process_all()
    print("✅ 증강 완료!")