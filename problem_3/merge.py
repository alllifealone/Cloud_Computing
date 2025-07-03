from PIL import Image
import os

# 图像数量与参数设置
image_folder = '.'  # 如果图片在当前目录
output_path = 'merged_wordclouds.png'
image_prefix = 'topic_'
image_suffix = '_wordcloud.png'
num_images = 9
grid_cols = 3
grid_rows = 3

# 加载所有图像
images = []
for i in range(num_images):
    image_path = os.path.join(image_folder, f"{image_prefix}{i}{image_suffix}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图像：{image_path}")
    img = Image.open(image_path)
    images.append(img)

# 假设所有图像大小一致
img_width, img_height = images[0].size

# 创建网格画布
merged_image = Image.new('RGB', (grid_cols * img_width, grid_rows * img_height), color='white')

# 粘贴每张图像到合适位置
for idx, img in enumerate(images):
    row = idx // grid_cols
    col = idx % grid_cols
    merged_image.paste(img, (col * img_width, row * img_height))

# 保存合并后的图像
merged_image.save(output_path)
print(f"✅ 已保存合并图像为：{output_path}")
