"""
从跟踪视频中提取ReID训练数据集
自动按track_id组织图像文件夹
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm


def extract_reid_dataset(
        video_paths,
        output_dir,
        model_path='best.pt',
        tracker='botsort.yaml',
        min_images_per_id=5,
        max_images_per_id=50,
        sample_interval=5
):
    """
    从视频中提取ReID数据集

    Args:
        video_paths: 视频路径列表
        output_dir: 输出目录
        model_path: YOLO模型路径
        tracker: 跟踪器配置
        min_images_per_id: 每个ID最少图像数（少于此数的ID会被过滤）
        max_images_per_id: 每个ID最多图像数
        sample_interval: 采样间隔（每N帧保存一次）
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    # 统计信息
    id_counts = {}
    total_images = 0

    for video_idx, video_path in enumerate(video_paths):
        print(f"\n处理视频 {video_idx + 1}/{len(video_paths)}: {video_path}")

        # 运行跟踪
        results = model.track(
            source=video_path,
            tracker=tracker,
            save=False,
            stream=True,
            verbose=False,
            persist=True
        )

        frame_idx = 0

        for result in tqdm(results, desc='提取帧'):
            # 只处理采样帧
            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue

            if result.boxes.id is None:
                frame_idx += 1
                continue

            frame = result.orig_img
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, track_ids, confidences):
                # 过滤低置信度
                if conf < 0.8:
                    continue

                # 检查是否超过最大数量
                if tid in id_counts and id_counts[tid] >= max_images_per_id:
                    continue

                # 裁剪图像
                x1, y1, x2, y2 = map(int, box)

                # 边界检查
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                # 过滤太小的图像
                if crop.shape[0] < 32 or crop.shape[1] < 32:
                    continue

                # 保存
                sheep_dir = output_dir / f'sheep_{tid:04d}'
                sheep_dir.mkdir(parents=True, exist_ok=True)

                img_path = sheep_dir / f'v{video_idx}_f{frame_idx:06d}.jpg'
                cv2.imwrite(str(img_path), crop)

                # 更新计数
                id_counts[tid] = id_counts.get(tid, 0) + 1
                total_images += 1

            frame_idx += 1

    # 过滤少于最小数量的ID
    print("\n清理数据...")
    valid_ids = 0
    removed_ids = 0

    for sheep_dir in output_dir.iterdir():
        if not sheep_dir.is_dir():
            continue

        num_images = len(list(sheep_dir.glob('*.jpg')))

        if num_images < min_images_per_id:
            # 删除文件夹
            import shutil
            shutil.rmtree(sheep_dir)
            removed_ids += 1
        else:
            valid_ids += 1

    # 统计
    print("\n" + "=" * 70)
    print("数据集提取完成！")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"有效ID数: {valid_ids}")
    print(f"移除ID数: {removed_ids} (少于{min_images_per_id}张图像)")
    print(f"总图像数: {total_images}")
    print(f"平均每ID: {total_images / valid_ids:.1f}张图像" if valid_ids > 0 else "N/A")


def split_train_val(data_dir, train_ratio=0.8):
    """
    划分训练集和验证集

    Args:
        data_dir: 数据目录（提取后的输出目录）
        train_ratio: 训练集比例
    """
    data_dir = Path(data_dir)
    train_dir = data_dir.parent / (data_dir.name + '_train')
    val_dir = data_dir.parent / (data_dir.name + '_val')

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有ID
    sheep_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    np.random.seed(42)
    np.random.shuffle(sheep_dirs)

    split_point = int(len(sheep_dirs) * train_ratio)

    train_ids = sheep_dirs[:split_point]
    val_ids = sheep_dirs[split_point:]

    # 移动文件
    import shutil

    for sheep_dir in tqdm(train_ids, desc='训练集'):
        dst = train_dir / sheep_dir.name
        shutil.copytree(sheep_dir, dst)

    for sheep_dir in tqdm(val_ids, desc='验证集'):
        dst = val_dir / sheep_dir.name
        shutil.copytree(sheep_dir, dst)

    print("\n" + "=" * 70)
    print("数据集划分完成！")
    print("=" * 70)
    print(f"训练集: {train_dir} ({len(train_ids)}只羊)")
    print(f"验证集: {val_dir} ({len(val_ids)}只羊)")


def augment_dataset(data_dir, aug_factor=2):
    """
    数据增强（可选）

    Args:
        data_dir: 数据目录
        aug_factor: 增强倍数
    """
    import albumentations as A

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
    ])

    data_dir = Path(data_dir)

    for sheep_dir in tqdm(data_dir.iterdir(), desc='数据增强'):
        if not sheep_dir.is_dir():
            continue

        images = list(sheep_dir.glob('*.jpg'))

        for img_path in images:
            img = cv2.imread(str(img_path))

            for aug_idx in range(aug_factor):
                augmented = transform(image=img)['image']

                aug_path = sheep_dir / f'{img_path.stem}_aug{aug_idx}.jpg'
                cv2.imwrite(str(aug_path), augmented)

    print(f"✅ 数据增强完成，每张图像增强{aug_factor}次")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='提取ReID数据集')
    parser.add_argument('--videos', nargs='+', required=True, help='视频路径列表')
    parser.add_argument('--output', default='data/sheep_reid_raw', help='输出目录')
    parser.add_argument('--model', default='best.pt', help='YOLO模型路径')
    parser.add_argument('--min_images', type=int, default=5, help='最少图像数')
    parser.add_argument('--max_images', type=int, default=50, help='最多图像数')
    parser.add_argument('--interval', type=int, default=5, help='采样间隔')
    parser.add_argument('--split', action='store_true', help='是否划分训练/验证集')
    parser.add_argument('--augment', type=int, default=0, help='数据增强倍数')

    args = parser.parse_args()

    # 提取数据集
    extract_reid_dataset(
        video_paths=args.videos,
        output_dir=args.output,
        model_path=args.model,
        min_images_per_id=args.min_images,
        max_images_per_id=args.max_images,
        sample_interval=args.interval
    )

    # 划分训练/验证集
    if args.split:
        split_train_val(args.output)

    # 数据增强
    if args.augment > 0:
        augment_dataset(args.output, aug_factor=args.augment)

    print("\n✅ 全部完成！")
    print("\n下一步:")
    print("  python train_sheep_reid.py")