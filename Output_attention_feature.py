import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    img = cv2.imread('data/cars/MVI_39271__img00675.jpg')
    cbam_feature_map = np.load('runs/detect/yolov5s_INSNl_BiFPN_CBAM_INSNs(modify)/MVI_39271__img00675/stage9_CBAM_features.npy')#yolov5s_INSNl_BiFPN_CBAM_INSNs(modify)

    # 或者使用最大池化来捕捉最显著的特征
    max_attention_map = np.max(cbam_feature_map, axis=0)

    max_attention_map = (max_attention_map - max_attention_map.min()) / (max_attention_map.max() - max_attention_map.min())
    img_height, img_width = 540, 960
    upsampled_attention_map = cv2.resize(max_attention_map, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    # 加载输入图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    img = cv2.resize(img, (upsampled_attention_map.shape[1], upsampled_attention_map.shape[0]))  # 调整到与注意力图相同的大小

    # 将注意力图转换为热力图，并叠加到原始图像上
    heatmap = cv2.applyColorMap(np.uint8(255 * (1-upsampled_attention_map)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(img) / 255
    overlay = overlay / np.max(overlay)  # 归一化叠加结果

    # 可视化叠加后的结果
    plt.imshow(overlay)
    plt.title('Attention Overlay on Original Image')
    plt.axis('off')
    plt.show()