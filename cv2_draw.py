import cv2
import numpy as np

def draw_bbox_opencv(img_path, obj_coord, save_path=None):
    """
    根据归一化坐标绘制矩形框（obj_coord）并显示图像
    
    参数:
        img_path: 图像路径
        obj_coord: [x_min, y_min, x_max, y_max]，归一化坐标（0~1之间）
        save_path: 若指定，则保存绘制后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # # 归一化坐标转为像素坐标
    # x_min = int(obj_coord[0] * w)
    # y_min = int(obj_coord[1] * h)
    # x_max = int(obj_coord[2] * w)
    # y_max = int(obj_coord[3] * h)
    
    # 归一化坐标转为像素坐标
    x_min = obj_coord[0]
    y_min = obj_coord[1]
    x_max = obj_coord[2]
    y_max = obj_coord[3]

    # 绘制矩形框（蓝色）
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # # 显示图像
    # cv2.imshow("BBox", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 可选保存
    if save_path:
        cv2.imwrite(save_path, img)
        
image_path = "/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/VRSBench/Images/Images_val/05888_0000.png"
obj_coord = [156, 416, 234, 570]
save_path = "/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/eval/VRSBench/ground_truth/test.jpg"
draw_bbox_opencv(image_path, obj_coord, save_path=save_path)