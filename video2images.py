import cv2
import os

def extract_frames_from_video(video_path, output_folder):
    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Mở video
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra xem video có thể mở không
    if not cap.isOpened():
        print(f"Can not open video: {video_path}")
        return

    frame_count = 0

    while True:
        # Đọc từng frame
        ret, frame = cap.read()

        # Kiểm tra xem video còn frame không
        if not ret:
            break

        # Lưu frame thành ảnh
        frame_count += 1
        image_path = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(image_path, frame)

    # Đóng video và giải phóng bộ nhớ
    cap.release()

    print(f"Extracted {frame_count} frames from video and save to {output_folder}.")

# Thực hiện trích xuất frames từ nhiều video và lưu vào các thư mục riêng
video_folder = "src/skeleton_data/skeletons2"  # Thay đổi đường dẫn tới thư mục chứa video của bạn
output_root_folder = "src/skeleton_data/skeletons2_images"  # Thư mục gốc cho việc lưu các thư mục con

# Lặp qua tất cả video trong thư mục
for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    video_name = os.path.splitext(video_file)[0]
    output_folder = os.path.join(output_root_folder, video_name)

    # Thực hiện trích xuất frames từ video và lưu vào thư mục con tương ứng
    extract_frames_from_video(video_path, output_folder)
