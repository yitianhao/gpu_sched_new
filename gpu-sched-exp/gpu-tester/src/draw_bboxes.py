import glob
import os
import cv2
import pandas as pd

fname = '../results/traffic/nocontrol/vision_results.csv'
# fname = '../results/traffic/controlsync/vision_results.csv'
df = pd.read_csv(fname)

# predictions = {'predictions': [{'x': 1012.0, 'y': 593.5, 'width': 406.0, 'height': 443.0, 'confidence': 0.7369905710220337, 'class': 'Paper', 'image_path': 'example.jpg', 'prediction_type': 'ObjectDetectionModel'}], 'image': {'width': 1436, 'height': 956}}
def draw_bboxes(df, img):
    for idx, row in df.iterrows():
        if row['score'] < 0.3:
            continue
        x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(img, start_point, end_point, color=(0,255,0), thickness=2)



out_dir = os.path.join(os.path.dirname(fname), "boxed_imgs/")
os.system(f'rm {out_dir}/*.jpg')
img_paths = sorted(glob.glob(os.path.join('../data-set/traffic/720x1280_12fps/*.jpg')))
df_boxes = None
for idx, img_path in enumerate(img_paths):
    frame_id = idx + 1
    img = cv2.imread(img_path)
    mask = df['frame_id'] == frame_id
    # print(frame_id, len(df[mask]))
    # if len(df[mask]
    # import pdb
    # pdb.set_trace()
    if len(df[mask]) > 0:
        df_boxes = df[mask]
    draw_bboxes(df_boxes, img)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    print(out_path)
    cv2.imwrite(out_path, img)
    # if idx > 725:
    #     break

