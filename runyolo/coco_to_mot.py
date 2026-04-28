import json
import os


def coco_to_mot(coco_annotations_path, output_txt_path, image_width=3840, image_height=2160):
    """
    将COCO格式的标注转换为MOT格式的gt.txt文件
    输出格式：frame,id,bb_left,bb_top,bb_width,bb_height,-1,-1,1
    """

    # 读取COCO标注文件
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    print(f"加载了 {len(coco_data['images'])} 张图像和 {len(coco_data['annotations'])} 个标注")

    # 创建图像ID到帧号的映射
    image_id_to_frame = {}
    frame_counter = 1

    # 按图像ID排序
    sorted_images = sorted(coco_data['images'], key=lambda x: x['id'])

    for image in sorted_images:
        image_id_to_frame[image['id']] = frame_counter
        frame_counter += 1

    # 按图像ID分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # 生成MOT格式的gt.txt
    with open(output_txt_path, 'w') as f:
        total_annotations = 0

        # 按图像ID排序处理
        sorted_image_ids = sorted(annotations_by_image.keys())

        for image_id in sorted_image_ids:
            frame_number = image_id_to_frame[image_id]
            annotations = annotations_by_image[image_id]

            # 为当前帧的每个检测分配对象ID（从1开始）
            object_id_counter = 1
            object_id_mapping = {}  # 用于跟踪同一对象在不同帧中的ID

            # 处理该图像中的所有标注
            for ann in annotations:
                # COCO格式： [x_min, y_min, width, height] (绝对像素坐标)
                x_min, y_min, width, height = ann['bbox']

                # 确保坐标在合理范围内
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                width = max(1, width)
                height = max(1, height)

                # 确保不超出图像边界
                if x_min + width > image_width:
                    width = image_width - x_min
                if y_min + height > image_height:
                    height = image_height - y_min

                # 使用简单的递增ID作为对象ID
                object_id = object_id_counter
                object_id_counter += 1

                # MOT格式：<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, 1
                line = f"{frame_number},{object_id},{x_min:.6f},{y_min:.6f},{width:.6f},{height:.6f},-1,-1,1\n"
                f.write(line)
                total_annotations += 1

    print(f"转换完成！")
    print(f"处理了 {len(annotations_by_image)} 个帧")
    print(f"总共 {total_annotations} 个检测框")
    print(f"结果保存到: {output_txt_path}")


def coco_to_mot_with_consistent_ids(coco_annotations_path, output_txt_path, image_width=3840, image_height=2160):
    """
    将COCO格式的标注转换为MOT格式的gt.txt文件
    保持对象ID在不同帧中的一致性
    """

    # 读取COCO标注文件
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    print(f"加载了 {len(coco_data['images'])} 张图像和 {len(coco_data['annotations'])} 个标注")

    # 创建图像ID到帧号的映射
    image_id_to_frame = {}
    frame_counter = 1

    # 按图像ID排序
    sorted_images = sorted(coco_data['images'], key=lambda x: x['id'])

    for image in sorted_images:
        image_id_to_frame[image['id']] = frame_counter
        frame_counter += 1

    # 按图像ID分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # 为每个类别分配固定的对象ID
    category_to_object_id = {}
    next_object_id = 1

    # 生成MOT格式的gt.txt
    with open(output_txt_path, 'w') as f:
        total_annotations = 0

        # 按图像ID排序处理
        sorted_image_ids = sorted(annotations_by_image.keys())

        for image_id in sorted_image_ids:
            frame_number = image_id_to_frame[image_id]
            annotations = annotations_by_image[image_id]

            # 处理该图像中的所有标注
            for ann in annotations:
                # COCO格式： [x_min, y_min, width, height]
                x_min, y_min, width, height = ann['bbox']

                # 确保坐标在合理范围内
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                width = max(1, width)
                height = max(1, height)

                # 确保不超出图像边界
                if x_min + width > image_width:
                    width = image_width - x_min
                if y_min + height > image_height:
                    height = image_height - y_min

                # 获取类别ID
                category_id = ann['category_id']

                # 为每个类别分配固定的对象ID
                if category_id not in category_to_object_id:
                    category_to_object_id[category_id] = next_object_id
                    next_object_id += 1

                object_id = category_to_object_id[category_id]

                # MOT格式：<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, 1
                line = f"{frame_number},{object_id},{x_min:.6f},{y_min:.6f},{width:.6f},{height:.6f},-1,-1,1\n"
                f.write(line)
                total_annotations += 1

    print(f"转换完成！")
    print(f"处理了 {len(annotations_by_image)} 个帧")
    print(f"总共 {total_annotations} 个检测框")
    print(f"分配了 {len(category_to_object_id)} 个对象ID")
    print(f"结果保存到: {output_txt_path}")


def coco_to_mot_exact_format(coco_annotations_path, output_txt_path, image_width=3840, image_height=2160):
    """
    精确匹配示例格式的转换函数
    """

    # 读取COCO标注文件
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    print(f"加载了 {len(coco_data['images'])} 张图像和 {len(coco_data['annotations'])} 个标注")

    # 创建图像ID到帧号的映射
    image_id_to_frame = {}

    # 按图像ID排序
    sorted_images = sorted(coco_data['images'], key=lambda x: x['id'])

    for frame_number, image in enumerate(sorted_images, 1):
        image_id_to_frame[image['id']] = frame_number

    # 按图像ID分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # 为每个检测框分配固定的对象ID（基于类别）
    category_ids = sorted(set(ann['category_id'] for ann in coco_data['annotations']))
    category_to_id = {cat_id: idx + 1 for idx, cat_id in enumerate(category_ids)}

    # 生成MOT格式的gt.txt
    with open(output_txt_path, 'w') as f:
        total_annotations = 0

        # 按图像ID排序处理
        sorted_image_ids = sorted(annotations_by_image.keys())

        for image_id in sorted_image_ids:
            frame_number = image_id_to_frame[image_id]
            annotations = annotations_by_image[image_id]

            # 处理该图像中的所有标注
            for ann in annotations:
                # COCO格式： [x_min, y_min, width, height]
                x_min, y_min, width, height = ann['bbox']

                # 使用类别ID作为对象ID
                object_id = category_to_id[ann['category_id']]

                # MOT格式：<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, 1
                line = f"{frame_number},{object_id},{x_min:.6f},{y_min:.6f},{width:.6f},{height:.6f},-1,-1,1\n"
                f.write(line)
                total_annotations += 1

    print(f"转换完成！")
    print(f"处理了 {len(annotations_by_image)} 个帧")
    print(f"总共 {total_annotations} 个检测框")
    print(f"结果保存到: {output_txt_path}")


def main():
    # 配置路径
    coco_annotations_path = "/home/runyolo/video1_annotations_hbb.json"  # 替换为你的COCO标注文件路径
    output_txt_path = "/home/runyolo/3.txt"  # 输出的gt.txt文件路径

    # 图像尺寸（根据你的数据修改）
    image_width = 3840
    image_height = 2160

    print("选择转换模式：")
    print("1. 基本转换（每帧重新分配对象ID）")
    print("2. 基于类别的固定对象ID")
    print("3. 精确格式匹配（推荐）")

    choice = input("请输入选择（1, 2 或 3）: ").strip()

    if choice == "1":
        coco_to_mot(coco_annotations_path, output_txt_path, image_width, image_height)
    elif choice == "2":
        coco_to_mot_with_consistent_ids(coco_annotations_path, output_txt_path, image_width, image_height)
    elif choice == "3":
        coco_to_mot_exact_format(coco_annotations_path, output_txt_path, image_width, image_height)
    else:
        print("无效选择，使用精确格式匹配模式")
        coco_to_mot_exact_format(coco_annotations_path, output_txt_path, image_width, image_height)


if __name__ == "__main__":
    main()