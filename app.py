
import PIL
import streamlit as st
from ultralytics import YOLO


model_path = r"C:\Users\ADMIN\OneDrive\Desktop\best (3).pt"

# Page layout
st.set_page_config(page_title="YOLO Object Detection", layout="wide", initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    st.header("Image selection")
    source_img = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))
    confidence = float(st.slider("Select model confidence", 25, 100, 40)) / 100
    
st.title("YOLO Object Detection")
st.caption("STEP 1: Upload an image by clicking on 'Browse files'")
st.caption("STEP 2: Click on the DETECT OBJECTS button to see the results")

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error("Unable to load the model. Check the specified path: " + model_path)
    st.error(ex)

if st.sidebar.button('DETECT OBJECTS') and source_img:
    uploaded_image = PIL.Image.open(source_img)
    res = model.predict(uploaded_image, conf=confidence,max_det=2000)
    boxes = res[0].boxes
    num_boxes = len(boxes)  # Total count of detected boxes

    class_name_map = { 
                0: 'C 32 2.5' ,
                1: 'C 38 2.9' ,
                2: 'C 48 2.9' ,
                3: 'R 20 40 1.9' ,
                4: 'R 25 75 1.9' ,
                5: 'R 48 96 2.0' ,
                6: 'R 48 96 2.9' ,
                7: 'R 60 40 1.9' ,
                8: 'R 80 40 1.2' ,
                9: 'R 96 48 2.0' ,
                10: 'R 96 48 2.9' ,
                11: 'S 20 20 1.2' ,
                12: 'S 20 20 1.5' ,
                13: 'S 20 20 1.9' ,
                14: 'S 25 25 1.9' ,
                15: 'S 25 25 2.5' ,
                16: 'S 38 38 1.9' ,
                17: 'S 40 40 2.5' ,
                18: 'S 50 50 1.5' ,
                19: 'S 50 50 1.9' ,
                20: 'S 50 50 4.0' ,
                21: 'S 60 60 2.0' ,
                22: 'S 72 72 4.0' ,
                23: 'S 72 72 4.8' ,
    }

    class_counts = {}
    for box in boxes:
        class_label = int(box.cls)  # Convert to integer
        class_name = class_name_map.get(class_label , "Unknown")
        if class_name not in class_counts:
            class_counts[class_name] = 1
        else:
            class_counts[class_name] += 1

    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption='Detected image with classes', use_column_width=True)

    if num_boxes == 0:
        st.error("No objects detected in the image.")
    else:
        st.write(f"Number of detected objects: {num_boxes}")

    st.write("Count of each class:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")

    try:
        with st.expander("Detected results"):
            for box in boxes:
                st.write(box.data)
    except Exception as ex:
        st.write("No image uploaded yet!!!")
