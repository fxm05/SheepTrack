import os
import glob


def yolo_to_mot(yolo_bbox, img_width, img_height):
    """
    将YOLO格式的bbox转换为MOT格式的绝对坐标
    YOLO: [class, center_x, center_y, width, height] (归一化)
    MOT: [x, y, width, height] (绝对像素坐标)
    """
    class_id, center_x, center_y, width, height = yolo_bbox

    # 转换为绝对像素坐标
    x_center_abs = center_x * img_width
    y_center_abs = center_y * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # 计算左上角坐标
    x_top_left = x_center_abs - (width_abs / 2)
    y_top_left = y_center_abs - (height_abs / 2)

    return x_top_left, y_top_left, width_abs, height_abs


def convert_yolo_folder_to_mot(input_folder, output_file, img_width=2560, img_height=1440):
    """
    将文件夹中的YOLO格式文件转换为MOT格式的gt.txt
    """
    # 获取所有txt文件并按文件名中的数字排序
    txt_files = sorted(glob.glob(os.path.join(input_folder, "*.txt")),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))

    with open(output_file, 'w') as f_out:
        for frame_id, txt_file in enumerate(txt_files, 1):
            # 确保frame_id从1开始，按顺序递增到298
            if frame_id > 298:
                break

            try:
                with open(txt_file, 'r') as f_in:
                    lines = f_in.readlines()

                for obj_id, line in enumerate(lines, 1):
                    # 解析YOLO格式
                    data = list(map(float, line.strip().split()))
                    if len(data) < 5:
                        continue

                    # 转换为MOT格式坐标
                    x, y, w, h = yolo_to_mot(data, img_width, img_height)

                    # 写入MOT格式 (frame_id, object_id, x, y, w, h, -1, -1, 1.0)
                    mot_line = f"{frame_id},{obj_id},{x:.6f},{y:.6f},{w:.6f},{h:.6f},-1,-1,1\n"
                    f_out.write(mot_line)

            except Exception as e:
                print(f"处理文件 {txt_file} 时出错: {e}")

    print(f"转换完成！输出文件: {output_file}")
    print(f"总共处理了 {len(txt_files)} 个文件，frame_id从1到{len(txt_files)}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_folder = "/home/runyolo/21"  # 替换为你的txt文件文件夹路径
    output_file = "/home/runyolo/3.txt"
    img_width = 2560  # 替换为你的图像宽度
    img_height = 1440  # 替换为你的图像高度

    convert_yolo_folder_to_mot(input_folder, output_file, img_width, img_height)