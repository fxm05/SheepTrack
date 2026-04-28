from ultralytics import YOLO

# Load your trained model
model = YOLO("/home/runyolo/runs/detect/train/weights/best.pt")

# Perform tracking on your video or image sequence
results = model.track(
    source="/root/autodl-tmp/datasets/test/test1/222.mp4",
    tracker="/home/ultralytics/ultralytics/cfg/trackers/botsort.yaml"
)
# print(type(results))
# print(results[0].boxes)
# print(results[1])
# print(results[2])
# Save results in MOTChallenge format (frame, id, bbox, conf)
with open('/home/runyolo/1.txt', 'w') as f:
    for frame_id, result in enumerate(results):
        for box in result.boxes:
            # if box.id is None:  # 再检查单个box
                # continue
            bbox = box.xyxy[0].tolist()  # Convert from tensor to list
            track_id = box.id.item()  # Get track id
            conf = box.conf.item()  # Get confidence score
            f.write(f'{frame_id+1},{int(track_id)},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,{conf}\n')