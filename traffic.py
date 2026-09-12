# pip install ultralytics
# pip install opencv-python
# pip install numpy

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# تحميل نموذج YOLO12x المُدرب مسبقًا
try:
    model = YOLO('yolo12x.pt') # تم التعديل لاستخدام نموذج قياسي، يمكنك تغييره إلى yolo12x.pt إذا كان لديك
    print("تم تحميل النموذج بنجاح.")
except Exception as e:
    print(f"حدث خطأ أثناء تحميل النموذج: {e}")
    exit()

# فتح ملف الفيديو
# !!! هام: تأكد من استبدال 'vid.mp4' بالمسار الصحيح لملف الفيديو الخاص بك
video_path = "vid.mp4" 
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"خطأ: لا يمكن فتح ملف الفيديو في المسار: {video_path}")
        print("يرجى التأكد من وجود ملف الفيديو بالاسم الصحيح في نفس مجلد الكود.")
        exit()
except Exception as e:
    print(f"حدث خطأ أثناء محاولة فتح الفيديو: {e}")
    exit()

# الحصول على أبعاد الفيديو ومعدل الإطارات
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# إعداد كائن VideoWriter لحفظ الفيديو
output_path = "video_output_with_blockage_status.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # أو استخدم 'XVID' لملفات .avi
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
print(f"سيتم حفظ الفيديو الناتج في المسار: {output_path}")


# تعريف خطوط العد والتقسيم
horizontal_line_y = int(frame_height * 0.8) # خط العد الأفقي
vertical_line_x = int(frame_width * 0.5)   # خط تقسيم المسارات العمودي

# متغيرات لتخزين بيانات التتبع
track_history = defaultdict(list)
counted_down_ids = set()
counted_up_ids = set()
wrong_way_counted_ids = set() 

# عدادات منفصلة لكل مسار
left_lane_up = 0
left_lane_down = 0
right_lane_up = 0
right_lane_down = 0
left_lane_wrong_way_count = 0
right_lane_wrong_way_count = 0

# متغيرات حالة انسداد الطريق بدلاً من العدادات
is_left_lane_blocked = False
is_right_lane_blocked = False

# ثوابت لكشف التوقف
STOP_THRESHOLD_PIXELS = 5  # أقصى حركة بالبكسل لاعتبار المركبة متوقفة
STOP_THRESHOLD_FRAMES = 15 # عدد الإطارات التي يجب أن تظل فيها المركبة ثابتة

# منطق تعلم اتجاه الأغلبية بناءً على الفارق
left_lane_up_count_learning = 0
left_lane_down_count_learning = 0
right_lane_up_count_learning = 0
right_lane_down_count_learning = 0
dominant_left_direction = 0
dominant_right_direction = 0
DIRECTION_CONFIRMATION_THRESHOLD = 5 

# قاموس لتخزين معلومات تتبع الاتجاه لكل مركبة
track_info = defaultdict(lambda: {
    'start_y': 0, 'start_frame': 0, 'direction': 0, 'counted_initial': False,
    'last_pos': (0, 0), 'stopped_frames': 0, 'is_stopped': False
})
frame_counter = 0
INITIAL_PERIOD_FRAMES = 90 # فترة العد الفوري في بداية الفيديو (3 ثواني)

# تعريف الألوان
vehicle_colors = {
    'car': (255, 0, 0), 'motorcycle': (0, 255, 0),
    'bus': (0, 0, 255), 'truck': (255, 255, 0)
}
down_color = (0, 0, 255)
up_color = (0, 255, 0)
wrong_way_color = (255, 0, 255)
blocked_status_color = (0, 0, 255) # Red for blocked
clear_status_color = (0, 255, 0)   # Green for clear

tracked_vehicles = list(vehicle_colors.keys())

print("بدء معالجة الفيديو وتعلم اتجاهات الحركة...")

# متغير لتخزين آخر إطار (أصلي)
last_frame = None
# *** إضافة جديدة: متغير لتخزين آخر إطار مع التحليلات ***
final_processed_frame = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("انتهت معالجة الفيديو.")
        break
    
    # تخزين نسخة من الإطار الأصلي قبل الرسم عليه
    last_frame = frame.copy()
    
    frame_counter += 1
    results = model.track(frame, persist=True, verbose=False)

    track_ids = []
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            class_name = model.names[cls]
            if class_name not in tracked_vehicles:
                continue

            x, y, w, h = box
            center_x, center_y = int(x), int(y)
            
            info = track_info[track_id]

            # منطق كشف التوقف
            last_x, last_y = info['last_pos']
            if last_x > 0 and last_y > 0:
                distance = ((center_x - last_x)**2 + (center_y - last_y)**2)**0.5
                if distance < STOP_THRESHOLD_PIXELS:
                    info['stopped_frames'] += 1
                else:
                    info['stopped_frames'] = 0
                    info['is_stopped'] = False
            
            if info['stopped_frames'] > STOP_THRESHOLD_FRAMES:
                info['is_stopped'] = True

            info['last_pos'] = (center_x, center_y)
            
            if info['start_frame'] == 0:
                info['start_y'] = center_y
                info['start_frame'] = frame_counter
            
            if info['direction'] == 0 and (frame_counter - info['start_frame']) > 5:
                dy = center_y - info['start_y']
                if abs(dy) > 10:
                    info['direction'] = 1 if dy > 0 else -1
                    if frame_counter <= INITIAL_PERIOD_FRAMES and not info['counted_initial']:
                        info['counted_initial'] = True
                        if info['direction'] == 1:
                            counted_down_ids.add(track_id)
                            if center_x < vertical_line_x: left_lane_down += 1
                            else: right_lane_down += 1
                        else:
                            counted_up_ids.add(track_id)
                            if center_x < vertical_line_x: left_lane_up += 1
                            else: right_lane_up += 1

            track = track_history[track_id]
            track.append((center_x, center_y))
            if len(track) > 30:
                track.pop(0)

            is_wrong_way = False
            if len(track) > 1:
                prev_x, prev_y = track[-2]
                if info['direction'] != 0:
                    if center_x < vertical_line_x:
                        if info['direction'] == 1: left_lane_down_count_learning +=1
                        else: left_lane_up_count_learning += 1
                        if dominant_left_direction == 0 and abs(left_lane_up_count_learning - left_lane_down_count_learning) >= DIRECTION_CONFIRMATION_THRESHOLD:
                            dominant_left_direction = -1 if left_lane_up_count_learning > left_lane_down_count_learning else 1
                    else:
                        if info['direction'] == 1: right_lane_down_count_learning += 1
                        else: right_lane_up_count_learning += 1
                        if dominant_right_direction == 0 and abs(right_lane_up_count_learning - right_lane_down_count_learning) >= DIRECTION_CONFIRMATION_THRESHOLD:
                            dominant_right_direction = -1 if right_lane_up_count_learning > right_lane_down_count_learning else 1

                if prev_y < horizontal_line_y and center_y >= horizontal_line_y and track_id not in counted_down_ids:
                    counted_down_ids.add(track_id)
                    if center_x < vertical_line_x: left_lane_down += 1
                    else: right_lane_down += 1
                elif prev_y > horizontal_line_y and center_y <= horizontal_line_y and track_id not in counted_up_ids:
                    counted_up_ids.add(track_id)
                    if center_x < vertical_line_x: left_lane_up += 1
                    else: right_lane_up += 1
                
                current_direction = info['direction']
                if current_direction != 0:
                    if center_x < vertical_line_x and dominant_left_direction != 0:
                        if current_direction != dominant_left_direction: is_wrong_way = True
                    elif center_x > vertical_line_x and dominant_right_direction != 0:
                        if current_direction != dominant_right_direction: is_wrong_way = True
                
                if is_wrong_way and track_id not in wrong_way_counted_ids:
                    wrong_way_counted_ids.add(track_id)
                    if center_x < vertical_line_x: left_lane_wrong_way_count += 1
                    else: right_lane_wrong_way_count += 1
                
            box_color = wrong_way_color if is_wrong_way else vehicle_colors.get(class_name, (200, 200, 200))
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = "Wrong Way!" if is_wrong_way else f"{class_name} ID:{track_id}"
            if info['is_stopped']: label += " (Stopped)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # منطق كشف حالة انسداد الطريق
    stopped_vehicles = {tid: track_info[tid] for tid in track_ids if track_info[tid]['is_stopped']}
    stopped_left = [info for info in stopped_vehicles.values() if info['last_pos'][0] < vertical_line_x]
    stopped_right = [info for info in stopped_vehicles.values() if info['last_pos'][0] >= vertical_line_x]

    is_left_lane_blocked = any(stopped_left[i]['direction'] != stopped_left[j]['direction'] for i in range(len(stopped_left)) for j in range(i + 1, len(stopped_left)) if stopped_left[i]['direction'] != 0 and stopped_left[j]['direction'] != 0)
    is_right_lane_blocked = any(stopped_right[i]['direction'] != stopped_right[j]['direction'] for i in range(len(stopped_right)) for j in range(i + 1, len(stopped_right)) if stopped_right[i]['direction'] != 0 and stopped_right[j]['direction'] != 0)

    cv2.line(frame, (0, horizontal_line_y), (frame_width, horizontal_line_y), (0, 255, 255), 2)
    cv2.line(frame, (vertical_line_x, 0), (vertical_line_x, frame_height), (255, 255, 0), 2)
    
    # عرض الإحصائيات
    y_offset_left = 40
    cv2.putText(frame, "Left Lane Stats", (30, y_offset_left), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_offset_left += 40
    if dominant_left_direction == 0:
        cv2.putText(frame, f"Up: {left_lane_up}", (30, y_offset_left), cv2.FONT_HERSHEY_SIMPLEX, 0.9, up_color, 2)
        y_offset_left += 35
        cv2.putText(frame, f"Down: {left_lane_down}", (30, y_offset_left), cv2.FONT_HERSHEY_SIMPLEX, 0.9, down_color, 2)
    elif dominant_left_direction == -1:
        cv2.putText(frame, f"Correct (Up): {left_lane_up}", (30, y_offset_left), cv2.FONT_HERSHEY_SIMPLEX, 0.9, up_color, 2)
    else:
        cv2.putText(frame, f"Correct (Down): {left_lane_down}", (30, y_offset_left), cv2.FONT_HERSHEY_SIMPLEX, 0.9, down_color, 2)
    y_offset_left += 35
    cv2.putText(frame, f"Wrong Way: {left_lane_wrong_way_count}", (30, y_offset_left), cv2.FONT_HERSHEY_SIMPLEX, 0.9, wrong_way_color, 2)
    y_offset_left += 35
    block_status_left = "Blocked" if is_left_lane_blocked else "Clear"
    block_color_left = blocked_status_color if is_left_lane_blocked else clear_status_color
    cv2.putText(frame, f"Status: {block_status_left}", (30, y_offset_left), cv2.FONT_HERSHEY_SIMPLEX, 0.9, block_color_left, 2)

    y_offset_right = 40
    right_x_base = frame_width - 300
    cv2.putText(frame, "Right Lane Stats", (right_x_base, y_offset_right), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_offset_right += 40
    if dominant_right_direction == 0:
        cv2.putText(frame, f"Up: {right_lane_up}", (right_x_base, y_offset_right), cv2.FONT_HERSHEY_SIMPLEX, 0.9, up_color, 2)
        y_offset_right += 35
        cv2.putText(frame, f"Down: {right_lane_down}", (right_x_base, y_offset_right), cv2.FONT_HERSHEY_SIMPLEX, 0.9, down_color, 2)
    elif dominant_right_direction == -1:
        cv2.putText(frame, f"Correct (Up): {right_lane_up}", (right_x_base, y_offset_right), cv2.FONT_HERSHEY_SIMPLEX, 0.9, up_color, 2)
    else:
        cv2.putText(frame, f"Correct (Down): {right_lane_down}", (right_x_base, y_offset_right), cv2.FONT_HERSHEY_SIMPLEX, 0.9, down_color, 2)
    y_offset_right += 35
    cv2.putText(frame, f"Wrong Way: {right_lane_wrong_way_count}", (right_x_base, y_offset_right), cv2.FONT_HERSHEY_SIMPLEX, 0.9, wrong_way_color, 2)
    y_offset_right += 35
    block_status_right = "Blocked" if is_right_lane_blocked else "Clear"
    block_color_right = blocked_status_color if is_right_lane_blocked else clear_status_color
    cv2.putText(frame, f"Status: {block_status_right}", (right_x_base, y_offset_right), cv2.FONT_HERSHEY_SIMPLEX, 0.9, block_color_right, 2)

    out.write(frame)
    
    # *** إضافة جديدة: تخزين نسخة من الإطار النهائي بالتحليلات ***
    final_processed_frame = frame.copy()

    cv2.imshow("Vehicle Tracking with Blockage Status", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# حفظ آخر إطار أصلي كصورة
if last_frame is not None:
    last_frame_path = "last_frame.jpg"
    cv2.imwrite(last_frame_path, last_frame)
    print(f"تم حفظ الإطار الأخير الأصلي في المسار: {last_frame_path}")

# *** إضافة جديدة: حفظ آخر إطار مع التحليلات ***
if final_processed_frame is not None:
    last_frame_analysis_path = "last_frame_with_analysis.jpg"
    cv2.imwrite(last_frame_analysis_path, final_processed_frame)
    print(f"تم حفظ الإطار الأخير مع التحليلات في المسار: {last_frame_analysis_path}")


cap.release()
out.release() 
cv2.destroyAllWindows()
print("تم إغلاق كل شيء بنجاح.")
