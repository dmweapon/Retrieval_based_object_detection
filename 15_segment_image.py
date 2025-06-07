# Refactored and fully working version of the code with step linkage and gallery selection fixes
# Compatible with Gradio 5.33.10

import gradio as gr
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import zipfile
from segment_anything import sam_model_registry, SamPredictor

# ------------------- 모델 선택 프롬프트 -------------------
print("어떤 모델을 사용할건지 목록에서 선택해주세요 (1, 2, 3 중 입력)")
print("1) vit_b (메모리 사용량: 약 14~16GB)")
print("2) vit_l (메모리 사용량: 약 20~22GB)")
print("3) vit_h (메모리 사용량: 약 28~32GB)")
model_choice = input("선택: ")
model_map = {"1": "vit_b", "2": "vit_l", "3": "vit_h"}
MODEL_TYPE = model_map.get(model_choice, "vit_b")

CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}
CHECKPOINT_PATH = Path("model") / Path(CHECKPOINT_URLS[MODEL_TYPE]).name
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
if not CHECKPOINT_PATH.exists():
    import urllib.request
    print(f"⬇️ SAM 모델 다운로드 중: {CHECKPOINT_PATH.name}")
    urllib.request.urlretrieve(CHECKPOINT_URLS[MODEL_TYPE], CHECKPOINT_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH))
sam.to(DEVICE)
predictor = SamPredictor(sam)

# ------------------- 상태 -------------------
state = {
    "input_dir": None,
    "output_dir": None,
    "image_paths": [],
    "selected_path": None,
    "selected_image": None,
    "masks": [],
}

# ------------------- 함수 -------------------
def update_paths(input_root, output_root, img_type, class_name):
    input_dir = Path(input_root) / f"{img_type}_images" / class_name
    output_dir = Path(output_root) / f"{img_type}_images" / class_name
    state["input_dir"] = input_dir
    state["output_dir"] = output_dir
    return str(input_dir), str(output_dir)

def load_images():
    input_dir = state.get("input_dir")
    if not input_dir or not input_dir.exists():
        return f"❌ 입력 디렉토리가 존재하지 않습니다: {input_dir}", []
    image_paths = sorted([p for p in input_dir.glob("*.jpg") if p.is_file()])
    state["image_paths"] = image_paths
    image_list = [(str(p), p.name) for p in image_paths]
    return f"✅ {len(image_paths)}개 이미지 로드됨", image_list

def select_image(evt: gr.SelectData):
    if not evt or not isinstance(evt.value, dict):
        return None
    image_path_str = evt.value.get("value")
    if not image_path_str:
        return None
    path = Path(image_path_str)
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    state["selected_path"] = path
    state["selected_image"] = image
    return image

def segment_with_click(evt: gr.SelectData):
    image = state.get("selected_image")
    if image is None:
        return []
    image_np = np.array(image)
    predictor.set_image(image_np)
    input_point = np.array([[evt.index[0], evt.index[1]]])
    input_label = np.array([1])
    try:
        masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
    except Exception as e:
        print("Segmentation 오류:", e)
        return []
    state["masks"] = masks
    return [Image.fromarray(np.dstack((image_np, m.astype(np.uint8)*255))) for m in masks]

def apply_selected_mask(mask_img):
    if state["selected_image"] is None or mask_img is None:
        return "❌ 이미지 또는 마스크가 없습니다."
    out_dir = state.get("output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = state["selected_path"].stem
    save_path = out_dir / f"{stem}_removebg.png"
    mask_np = np.array(mask_img)
    if mask_np.shape[2] == 4:
        alpha_mask = mask_np[:, :, 3] > 128
        rgb = np.array(state["selected_image"])
        rgba = np.dstack([rgb, alpha_mask.astype(np.uint8) * 255])
        Image.fromarray(rgba).save(save_path)
        return f"✅ 저장 완료: {save_path}"
    return "❌ 유효한 마스크가 아닙니다."

def load_output_images():
    output_dir = state.get("output_dir")
    if not output_dir or not output_dir.exists():
        return []
    return [(str(p), p.name) for p in sorted(output_dir.glob("*.png")) if p.is_file()]

def download_output():
    output_dir = state.get("output_dir")
    if not output_dir or not output_dir.exists():
        return None
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in output_dir.glob("*.png"):
            zipf.write(file, arcname=file.name)
    return str(zip_path)

# ------------------- Gradio UI -------------------
with gr.Blocks(title="Retriever-Based Object Segmentation") as demo:
    gr.Markdown("### 디렉토리 설정")
    with gr.Row():
        input_root = gr.Textbox(label="Input Root Dir", value="dataset_cropped")
        output_root = gr.Textbox(label="Output Root Dir", value="dataset_segmented")
        img_type = gr.Radio(["original", "natural"], label="이미지 유형", value="original")
        class_name = gr.Textbox(label="클래스 이름")
    input_dir_display = gr.Textbox(label="입력 데이터 디렉토리")
    output_dir_display = gr.Textbox(label="출력 데이터 디렉토리")
    apply_dir_btn = gr.Button("경로 적용")
    apply_dir_btn.click(fn=update_paths, inputs=[input_root, output_root, img_type, class_name], outputs=[input_dir_display, output_dir_display])

    gr.Markdown("### Step 1: 이미지 로드 및 선택")
    load_btn = gr.Button("이미지 로드")
    gallery = gr.Gallery(label="이미지 목록", columns=4, height=400, allow_preview=True)
    selected_img = gr.Image(label="선택된 이미지", type="pil")
    load_status = gr.Textbox(label="로드 상태")
    load_btn.click(fn=load_images, outputs=[load_status, gallery])
    gallery.select(fn=select_image, inputs=None, outputs=selected_img)

    gr.Markdown("### Step 2: 세그멘테이션")
    segment_display = gr.Image(label="클릭할 이미지", type="pil")
    mask_gallery = gr.Gallery(label="마스크 후보", columns=3, height=400, allow_preview=True)
    selected_mask_display = gr.Image(label="선택된 마스크")
    confirm_btn = gr.Button("선택한 마스크 적용 및 저장")
    save_status = gr.Textbox(label="저장 결과")
    selected_img.change(
        fn=lambda img_path: Image.open(img_path).convert("RGB") if img_path else None,
        inputs=selected_img,
        outputs=segment_display
    )
    segment_display.select(fn=segment_with_click, outputs=mask_gallery)
    mask_gallery.select(fn=lambda x: x, inputs=None, outputs=selected_mask_display)
    confirm_btn.click(fn=apply_selected_mask, inputs=selected_mask_display, outputs=save_status)

    gr.Markdown("### Step 3: 출력 확인 및 다운로드")
    output_gallery = gr.Gallery(label="출력 이미지 목록", columns=4, height=300)
    selected_output_image_display = gr.Image(label="선택된 출력 이미지")
    output_btn = gr.Button("출력 이미지 로드")
    download_output_btn = gr.Button("전체 다운로드")
    download_file = gr.File(label="압축파일")
    output_btn.click(fn=load_output_images, outputs=output_gallery)
    output_gallery.select(fn=select_image, outputs=selected_output_image_display)
    download_output_btn.click(fn=download_output, outputs=download_file)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7890)