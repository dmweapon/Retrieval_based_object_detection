# Refactored 15_segment_image.py with path list displays and filtered load option
import shutil
import gradio as gr
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import zipfile
from segment_anything import sam_model_registry, SamPredictor

# 서버 시작 시 이전 zip 삭제
if Path("temp").exists():
    shutil.rmtree("temp")

print("어떤 모델을 사용할건지 목록에서 선택해주세요 (1, 2, 3 중 입력)")
print("1) vit_b | [VRAM 사용량] 2~3 건 처리시: 2~3GB, 최대 처리시: 14~16GB")
print("2) vit_l | [VRAM 사용량] 2~3 건 처리시: 4~6GB, 최대 처리시: 20~22GB")
print("3) vit_h | [VRAM 사용량] 2~3 건 처리시: 6~8GB, 최대 처리시: 28~32GB")
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

state = {
    "input_dir": None,
    "output_dir": None,
    "image_paths": [],
    "selected_path": None,
    "selected_image": None,
    "masks": [],
    "selected_mask_array": None,
}

def update_paths(input_root, output_root, img_type, class_name):
    input_dir = Path(input_root) / f"{img_type}_images" / class_name
    output_dir = Path(output_root) / f"{img_type}_images" / class_name
    state["input_dir"] = input_dir
    state["output_dir"] = output_dir
    return str(input_dir), str(output_dir)

def load_images():
    input_dir = state.get("input_dir")
    if not input_dir or not input_dir.exists():
        return "❌ 입력 디렉토리가 존재하지 않습니다", [], ""
    image_paths = sorted([p for p in input_dir.glob("*.jpg") if p.is_file()])
    state["image_paths"] = image_paths
    return f"✅ {len(image_paths)}개 이미지 로드됨", [(str(p), p.name) for p in image_paths], "\n".join(map(str, image_paths))

def load_unprocessed_images():
    input_dir = state.get("input_dir")
    output_dir = state.get("output_dir")
    if not input_dir or not output_dir:
        return "❌ 경로 미지정", [], ""
    processed_stems = {p.stem.replace("_rmbg", "") for p in output_dir.glob("*_rmbg.png")}
    image_paths = sorted([p for p in input_dir.glob("*.jpg") if p.stem not in processed_stems])
    state["image_paths"] = image_paths
    return f"✅ {len(image_paths)}개 미처리 이미지 로드됨", [(str(p), p.name) for p in image_paths], "\n".join(map(str, image_paths))

# def select_image(evt: gr.SelectData):
#     try:
#         print("✅ gallery selected")
#         value = evt.value
#         image_path_str = value.get("image", {}).get("path") if isinstance(value, dict) else value
#         if not image_path_str:
#             return "❌ 이미지 경로 없음"
#         path = Path(image_path_str)
#         if not path.exists():
#             return f"❌ 경로 존재하지 않음: {image_path_str}"
#         state["selected_path"] = path
#         return str(path)
#     except Exception as e:
#         return f"❌ 예외 발생: {e}"
def select_image(evt: gr.SelectData):
    """
    갤러리에서 이미지 선택 시 경로를 업데이트.
    단일 이미지인 경우 index가 None일 수 있어, 이때 기본 0으로 설정.
    """
    try:
        idx = 0
        if evt.index is not None:
            # evt.index가 튜플일 경우 첫 번째 사용
            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        path = state["image_paths"][idx]
        state["selected_path"] = path
        return str(path)
    except Exception as e:
        print(f"❌ select_image 오류: {e}")
        return ""

def pass_selected_image_to_step2():
    path = state.get("selected_path")
    if not path or not path.exists():
        return None, "경로 없음"
    try:
        image = Image.open(path).convert("RGB")
        state["selected_image"] = image
        return image, str(path)
    except Exception as e:
        print(f"이미지 로드 오류: {e}")
        return None, f"로드 실패: {e}"

def segment_with_click(evt: gr.SelectData):
    print("📌 segment_with_click() called")
    image = state.get("selected_image")
    if image is None:
        print("❌ selected_image is None")
        return []
    image_np = np.array(image)
    predictor.set_image(image_np)
    input_point = np.array([[evt.index[0], evt.index[1]]])
    input_label = np.array([1])
    print("🎯 Click at:", input_point.tolist())
    try:
        masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
    except Exception as e:
        print("Segmentation 오류:", e)
        return []
    state["masks"] = masks
    previews = []
    for i, m in enumerate(masks):
        rgba = np.dstack((image_np, m.astype(np.uint8) * 255))
        previews.append(Image.fromarray(rgba))
    print(f"✅ {len(previews)}개 마스크 생성 완료")
    return previews

def select_mask_by_index(evt: gr.SelectData):
    idx = evt.index
    print("📌 select_mask_by_index() called with index:", idx)
    masks = state.get("masks")
    image = state.get("selected_image")
    if image is None or masks is None or idx >= len(masks):
        print("❌ 유효하지 않은 마스크 선택")
        return None
    mask = masks[idx]
    state["selected_mask_array"] = mask
    rgba = np.dstack((np.array(image), mask.astype(np.uint8) * 255))
    return Image.fromarray(rgba)

def apply_selected_mask(_):
    print("📌 apply_selected_mask() 호출됨")
    selected_image = state.get("selected_image")
    selected_mask = state.get("selected_mask_array")
    if selected_image is None or selected_mask is None:
        print("❌ 이미지 또는 마스크 없음")
        return "❌ 이미지 또는 마스크가 없습니다."
    out_dir = state.get("output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = state["selected_path"].stem
    save_path = out_dir / f"{stem}_rmbg.png"
    rgb = np.array(selected_image)
    alpha = selected_mask.astype(np.uint8) * 255
    rgba = np.dstack([rgb, alpha])
    print("✅ 최종 저장 이미지 shape:", rgba.shape)
    Image.fromarray(rgba).save(save_path)
    return f"✅ 저장 완료: {save_path}"

def load_output_images():
    output_dir = state.get("output_dir")
    if not output_dir or not output_dir.exists():
        return [], ""
    png_paths = sorted([p for p in output_dir.glob("*.png") if p.is_file()])
    return [(str(p), p.name) for p in png_paths], "\n".join(map(str, png_paths))

def download_output():
    output_dir = state.get("output_dir")
    if not output_dir or not output_dir.exists():
        return None

    # ✅ 압축파일 저장 디렉토리: ./temp
    zip_dir = Path("temp")
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"{output_dir.name}.zip"

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in output_dir.glob("*.png"):
            zipf.write(file, arcname=file.name)

    print(f"✅ 압축 완료: {zip_path}")
    return str(zip_path)

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
    with gr.Row():
        load_btn = gr.Button("이미지 로드")
        load_filtered_btn = gr.Button("미처리 이미지만 로드")
    load_status = gr.Textbox(label="로드 상태")
    image_path_list = gr.Textbox(label="이미지 목록 (path)", lines=5)
    gallery = gr.Gallery(label="이미지 목록", columns=4, height=400, allow_preview=True)
    selected_path_display = gr.Textbox(label="선택된 이미지 경로 (Step1)", interactive=False)
    load_btn.click(fn=load_images, outputs=[load_status, gallery, image_path_list])
    load_filtered_btn.click(fn=load_unprocessed_images, outputs=[load_status, gallery, image_path_list])
    gallery.select(fn=select_image, outputs=selected_path_display)

    next_step_btn = gr.Button("다음 단계로 진행")

    gr.Markdown("### Step 2: 세그멘테이션")
    segment_display = gr.Image(label="클릭할 이미지", type="pil")
    segment_path_display = gr.Textbox(label="현재 세그멘테이션 이미지 경로 (Step2)", interactive=False)
    next_step_btn.click(fn=pass_selected_image_to_step2, outputs=[segment_display, segment_path_display])

    mask_gallery = gr.Gallery(label="마스크 후보", columns=3, height=400, allow_preview=True)
    selected_mask_display = gr.Image(label="선택된 마스크", type="pil")
    confirm_btn = gr.Button("선택한 마스크 적용 및 저장")
    save_status = gr.Textbox(label="저장 결과")
    segment_display.select(fn=segment_with_click, outputs=mask_gallery)
    mask_gallery.select(fn=select_mask_by_index, outputs=selected_mask_display)
    confirm_btn.click(fn=apply_selected_mask, inputs=[], outputs=save_status)

    gr.Markdown("### Step 3: 출력 확인 및 다운로드")
    output_btn = gr.Button("출력 이미지 로드")
    output_path_list = gr.Textbox(label="출력 이미지 목록 (path)", lines=5)
    output_gallery = gr.Gallery(label="출력 이미지 목록", columns=4, height=300, allow_preview=True)
    selected_output_image_display = gr.Image(label="선택된 출력 이미지")
    download_output_btn = gr.Button("전체 다운로드")
    download_file = gr.File(label="압축파일")
    output_btn.click(fn=load_output_images, outputs=[output_gallery, output_path_list])
    output_gallery.select(fn=select_image, outputs=selected_output_image_display)
    download_output_btn.click(fn=download_output, outputs=download_file)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7890)