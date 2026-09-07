import cv2
import mediapipe as mp

# 1. เริ่มใช้งานโมดูลตรวจจับมือและตัววาดกราฟิกของ MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ตั้งค่าโมเดล Hand Tracking
hands = mp_hands.Hands(
    static_image_mode=False,        # ตั้งค่าเป็น False เพื่อบอกว่าเราประมวลผลเป็นวิดีโอต่อเนื่อง
    max_num_hands=2,                # จำนวนมือสูงสุดที่ต้องการให้ระบบตรวจจับพร้อมกัน
    min_detection_confidence=0.5,   # ค่าความมั่นใจขั้นต่ำในการตรวจจับครั้งแรก
    min_tracking_confidence=0.5    # ค่าความมั่นใจขั้นต่ำในการติดตามพิกัดมือต่อเนื่อง
)

# 2. เชื่อมต่อไปยังกล้องเว็บแคม (ปกติกล้องหลักในคอมคือเลข 0)
cap = cv2.VideoCapture(0)

print("กำลังเปิดกล้อง... กดปุ่ม 'q' บนคีย์บอร์ดเพื่อปิดโปรแกรม")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("ไม่สามารถอ่านเฟรมจากกล้องได้")
        break

    # พลิกภาพแนวนอนเพื่อให้เหมือนการมองกระจกเงา (ช่วยให้การควบคุมเป็นธรรมชาติขึ้น)
    frame = cv2.flip(frame, 1)

    # แปลงสีของภาพจาก BGR (ที่ OpenCV ใช้) เป็น RGB (ที่ MediaPipe ต้องการ)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ส่งเฟรมภาพไปให้ AI ของ MediaPipe ประมวลผลหาพิกัดมือ
    results = hands.process(rgb_frame)

    # 3. ถ้า AI ตรวจพบพิกัดมือ ให้ทำการวาดจุดพิกัดลงบนหน้าจอ
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # วาดจุด (Landmarks) ทั้ง 21 จุดและเส้นเชื่อมโยงกระดูกมือ (Connections)
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4), # สีของจุด (เขียว)
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)                 # สีของเส้นเชื่อม (แดง)
            )

    # 4. แสดงผลลัพธ์ผ่านหน้าต่างแสดงวิดีโอของ OpenCV
    cv2.imshow('My First Hand Tracking', frame)

    # วิธีปิดโปรแกรม: เช็คว่ามีการกดปุ่ม 'q' บนคีย์บอร์ดหรือไม่
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ล้างระบบและปิดหน้าต่างทั้งหมดเมื่อเลิกใช้งาน
cap.release()
cv2.destroyAllWindows()