import streamlit as st
import cv2
import numpy as np
from PIL import Image
from cellpose.models import Cellpose

st.set_page_config(layout="wide")

st.title("🦠 Malaria RBC Detection (Cellpose)")
st.write("현미경 이미지를 업로드하면 RBC를 자동 분할하고 감염률을 계산합니다.")

# -------------------------------------------------
# ✅ 모델 캐싱 (중요 - Cloud에서 필수)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return Cellpose(gpu=False, model_type='cyto')  # Cloud 안정버전

model = load_model()

# -------------------------------------------------
# 이미지 업로드
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "이미지 업로드",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    if img_bgr is None:
        st.error("이미지를 읽을 수 없습니다.")
        st.stop()

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
             caption="Original Image",
             use_container_width=True)

    if st.button("🔍 AI 분석 시작"):

        with st.spinner("AI 모델 로딩 및 세포 분석 중..."):

            orig_bgr = img_bgr.copy()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # -------------------------------------------------
            # Cellpose 분할
            # -------------------------------------------------
            masks, flows, styles, diams = model.eval(
                img_rgb,
                diameter=None,
                channels=[0, 0]
            )

            total_cells = int(np.max(masks))
            height, width = masks.shape
            cell_areas = np.bincount(masks.flatten())

            # -------------------------------------------------
            # Edge 필터링
            # -------------------------------------------------
            MARGIN = 30
            edge_cells = (
                set(masks[:MARGIN, :].flatten()) |
                set(masks[-MARGIN:, :].flatten()) |
                set(masks[:, :MARGIN].flatten()) |
                set(masks[:, -MARGIN:].flatten())
            )
            edge_cells.discard(0)

            inner_cells_areas = [
                cell_areas[cid]
                for cid in range(1, total_cells + 1)
                if cid not in edge_cells
            ]

            if len(inner_cells_areas) > 0:
                standard_area = np.median(inner_cells_areas)
            else:
                standard_area = np.median(cell_areas[1:])

            valid_cells_set = set()

            for cell_id in range(1, total_cells + 1):
                if cell_id in edge_cells and cell_areas[cell_id] < (standard_area * 0.9):
                    continue
                if cell_areas[cell_id] < (standard_area * 0.4):
                    continue
                valid_cells_set.add(cell_id)

            # -------------------------------------------------
            # 감염 판별 (HSV 보라색 기준)
            # -------------------------------------------------
            PARASITE_THR = 10
            lower_purple = np.array([120, 100, 50])
            upper_purple = np.array([170, 255, 255])

            infected_cells_set = set()
            cell_contours = {}

            for cell_id in range(1, total_cells + 1):

                cell_mask = (masks == cell_id).astype(np.uint8)
                contours, _ = cv2.findContours(
                    cell_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cell_contours[cell_id] = contours

                y_idx, x_idx = np.where(cell_mask == 1)
                if len(y_idx) == 0:
                    continue

                y_min, y_max = np.min(y_idx), np.max(y_idx)
                x_min, x_max = np.min(x_idx), np.max(x_idx)

                roi_hsv = hsv_img[y_min:y_max+1, x_min:x_max+1]
                roi_mask = cell_mask[y_min:y_max+1, x_min:x_max+1]

                parasite_mask = cv2.inRange(
                    roi_hsv,
                    lower_purple,
                    upper_purple
                )

                parasite_in_cell = cv2.bitwise_and(
                    parasite_mask,
                    parasite_mask,
                    mask=roi_mask
                )

                if cv2.countNonZero(parasite_in_cell) > PARASITE_THR:
                    infected_cells_set.add(cell_id)

            # -------------------------------------------------
            # 결과 시각화
            # -------------------------------------------------
            output = orig_bgr.copy()

            for cell_id in range(1, total_cells + 1):

                contours = cell_contours.get(cell_id, [])
                if not contours:
                    continue

                if cell_id in valid_cells_set:
                    if cell_id in infected_cells_set:
                        color = (0, 0, 255)   # Red
                    else:
                        color = (0, 255, 0)   # Green
                else:
                    color = (128, 128, 128)   # Gray

                cv2.drawContours(output, contours, -1, color, 2)

            tot_valid = len(valid_cells_set)
            tot_inf = len(infected_cells_set.intersection(valid_cells_set))
            parasitemia = (tot_inf / tot_valid * 100) if tot_valid > 0 else 0

        # -------------------------------------------------
        # 출력
        # -------------------------------------------------
        st.success("분석 완료!")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                caption="Detection Result",
                use_container_width=True
            )

        with col2:
            st.metric("Valid RBC", tot_valid)
            st.metric("Infected RBC", tot_inf)
            st.metric("Parasitemia (%)", f"{parasitemia:.2f}")

        st.info("🔴 Red: Infected | 🟢 Green: Normal | ⚫ Gray: Excluded")
