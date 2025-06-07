import os
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO

# =============== 설정 ===============
root_dataset_dir = Path("./dataset_jpeg")

input_choice = input("라벨링 검수할 디렉토리 선택(original / natural): ").strip().lower()
if input_choice == "original":
    selected_dir = root_dataset_dir / "original_images"
elif input_choice == "natural":
    selected_dir = root_dataset_dir / "natural_images"
else:
    raise ValueError("잘못된 입력입니다. 'original' 또는 'natural'만 입력 가능합니다.")

dir_model = Path("./model/yolov8s.pt")
# padding = 0.25
padding = 0
confidence = 0.25
max_object_count = 4
progress_file = Path("saved_labeling_check_progress.txt")
manual_dir = Path("dataset_manual")
manual_dir.mkdir(exist_ok=True)
# ===================================

# ✅ 클래스 목록 로드
classes_txt = selected_dir / "classes.txt"
if not classes_txt.exists():
    raise FileNotFoundError(f"클래스 목록 파일이 존재하지 않습니다: {classes_txt}")
with open(classes_txt) as f:
    class_list = [line.strip() for line in f if line.strip()]
class_name_to_id = {name: idx for idx, name in enumerate(class_list)}

model = YOLO(str(dir_model))

# def draw_yolo_boxes(image, boxes, highlight_idx=None, padding=0.0):
#     h, w = image.shape[:2]
#     for idx, (cls_id, x, y, bw, bh) in enumerate(boxes):
#         pw = bw * padding
#         ph = bh * padding
#         x1 = max(0, int((x - bw / 2 - pw) * w))
#         y1 = max(0, int((y - bh / 2 - ph) * h))
#         x2 = min(w - 1, int((x + bw / 2 + pw) * w))
#         y2 = min(h - 1, int((y + bh / 2 + ph) * h))

#         color = (0, 255, 0) if idx != highlight_idx else (0, 0, 255)  # 초록 or 빨강
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

#         # 번호 텍스트 스타일 유지
#         cv2.putText(
#             image,
#             f"{idx + 1}",
#             (x1, max(y1 - 10, 10)),  # 너무 위로 올라가지 않도록 최소 y=10 보정
#             cv2.FONT_HERSHEY_SIMPLEX,
#             10,              # 폰트 크기
#             color,          # 박스 색과 동일
#             10,              # 두께
#             cv2.LINE_AA     # 안티앨리어싱
#         )
#     return image

# YOLO포멧의  바운딩박스 그리기
def draw_yolo_boxes(image, boxes, highlight_idx=None, padding=0.0):
    h, w = image.shape[:2]
    for idx, (cls_id, x, y, bw, bh) in enumerate(boxes):
        pw = bw * padding
        ph = bh * padding
        x1 = max(0, int((x - bw / 2 - pw) * w))
        y1 = max(0, int((y - bh / 2 - ph) * h))
        x2 = min(w - 1, int((x + bw / 2 + pw) * w))
        y2 = min(h - 1, int((y + bh / 2 + ph) * h))

        color = (0, 255, 0) if idx != highlight_idx else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)

        # 넘버링 중앙에 표시
        label = f"{idx + 1}"
        font_scale = 10
        thickness = 10
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 텍스트 크기 계산
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 + text_height) // 2

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
    return image

import numpy as np

# YOLO모델 객체 탐지
def run_yolo_detection(image, label_path, cls_id):
    results = model.predict(source=image, conf=confidence, verbose=False)
    h, w = image.shape[:2]
    boxes = []

    pred = results[0]
    # confidence 기준 상위 max_object_count개 선택
    if pred.boxes and len(pred.boxes) > 0:
        confs = pred.boxes.conf.cpu().numpy()
        top_idx = np.argsort(confs)[::-1][:max_object_count]
        xyxy = pred.boxes.xyxy.cpu().numpy()

        with open(label_path, "w") as f:
            for i in top_idx:
                x1, y1, x2, y2 = xyxy[i]
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
                boxes.append((cls_id, x_center, y_center, bw, bh))
    return boxes

# 수동 라벨링이 필요한 이미지는 manual_dir/{class_name}이라는 dir로 이동
def move_to_manual(img_path: Path):
    class_name = img_path.parent.name
    target_class_dir = manual_dir / class_name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    # classes.txt 복사
    source_classes_txt = selected_dir / "classes.txt"
    target_classes_txt = target_class_dir / "classes.txt"
    if not target_classes_txt.exists():
        shutil.copy(source_classes_txt, target_classes_txt)

    # 이미지 이동
    target_img = target_class_dir / img_path.name
    shutil.move(str(img_path), str(target_img))

    # 라벨 삭제
    label_path = img_path.with_suffix(".txt")
    if label_path.exists():
        os.remove(str(label_path))

    print(f"📂 수동 라벨링 디렉토리로 이동: {target_img}")

def sort_key(p): return (p.parent.name, p.name)

# 이미지 수집
image_paths = sorted(list(selected_dir.rglob("*.jpg")), key=sort_key)
total_images = len(image_paths)

# 진행 위치 복원
resume_idx = 0
if progress_file.exists():
    with open(progress_file, "r") as f:
        last_path = f.read().strip()
    if last_path in [str(p) for p in image_paths]:
        last_idx = [str(p) for p in image_paths].index(last_path)
        ans = input("이전 검수 위치가 저장되어 있습니다. 이어서 진행할까요? (y/n): ").strip().lower()
        resume_idx = last_idx + 1 if ans == "y" else 0

accepted_images = 0
rejected_images = 0
skipped_images = 0

i = resume_idx
while 0 <= i < total_images:
    img_file = image_paths[i]
    label_file = img_file.with_suffix(".txt")
    image = cv2.imread(str(img_file))

    if image is None:
        print(f"⚠️ 이미지 로드 실패: {img_file}")
        i += 1
        continue

    with open(progress_file, "w") as f:
        f.write(str(img_file))

    # class_id 결정
    class_dir = img_file.parent.name
    if class_dir not in class_name_to_id:
        print(f"🚫 클래스 '{class_dir}'가 classes.txt에 존재하지 않습니다.")
        i += 1
        continue
    class_id = class_name_to_id[class_dir]

    boxes = []
    if label_file.exists():
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append(tuple(map(float, parts)))
    else:
        print("⚠️ 라벨 파일 없음 → 수동 라벨링 디렉토리로 이동")
        move_to_manual(img_file)
        i += 1
        continue

    selected_idx = None

    while True:
        display_image = draw_yolo_boxes(image.copy(), boxes, selected_idx, padding=padding)
        cv2.imshow("검증 이미지", display_image)
        print(f"\n🔍 검증 대상 [{i+1}/{total_images}]: {img_file}")
        print(f"클래스명: {class_dir}, 클래스 ID: {class_id}")
        print(f"탐지된 박스 수: {len(boxes)}")
        print("숫자(1~n): 박스 선택 → Enter로 확정")
        print("'r': 재탐지 | 'm': 라벨 삭제 | 'n': 다음 | 'b': 이전")

        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # Enter
            if selected_idx is None and len(boxes) == 1:
                selected_idx = 0
                print("🔹 박스 1개 자동 선택됨")

            if selected_idx is not None and 0 <= selected_idx < len(boxes):
                x, y, bw, bh = boxes[selected_idx][1:]
                with open(label_file, "w") as f:
                    f.write(f"{class_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
                print(f"✅ 박스 {selected_idx+1} 저장 완료")
                accepted_images += 1
                cv2.destroyAllWindows()
                i += 1
                break
            else:
                print("🚫 박스를 먼저 선택하세요.")

        elif chr(key).isdigit():
            idx = int(chr(key)) - 1
            if 0 <= idx < len(boxes):
                selected_idx = idx
                print(f"🔴 박스 {selected_idx+1} 선택됨 (빨간색 표시)")

        elif key == ord('r'):
            print("🔁 YOLO 재탐지 수행...")
            boxes = run_yolo_detection(image, label_file, class_id)
            if len(boxes) == 0:
                print("🚫 재탐지 실패 - 수동 라벨링으로 이동")
                move_to_manual(img_file)
                cv2.destroyAllWindows()
                i += 1
                break
            selected_idx = None
            cv2.destroyAllWindows()
            continue

        elif key == ord('m'):
            if label_file.exists():
                os.remove(label_file)
                print(f"🗑️ 라벨 삭제 완료: {label_file.name}")
            move_to_manual(img_file)
            rejected_images += 1
            cv2.destroyAllWindows()
            i += 1
            break

        elif key == ord('n'):
            print("⏭️ 이미지 스킵됨.")
            skipped_images += 1
            cv2.destroyAllWindows()
            i += 1
            break

        elif key == ord('b'):
            print("🔙 이전 이미지로 이동")
            cv2.destroyAllWindows()
            i -= 1
            break

        else:
            print("🚫 잘못된 키 입력입니다.")

if progress_file.exists():
    progress_file.unlink()

print("\n📊 검증 결과 요약")
print(f"총 이미지 수         : {total_images}")
print(f"확정 저장된 이미지    : {accepted_images}")
print(f"라벨 삭제된 이미지 수 : {rejected_images}")
print(f"스킵된 이미지 수      : {skipped_images}")