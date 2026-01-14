from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("Models/best_yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------------------------
    # 1. SEGMENTACIÓN DE LA HOJA (HSV)
    # --------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hoja_area = 0
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        hoja_area = cv2.contourArea(c)
        cv2.drawContours(frame, [c], -1, (255, 0, 0), 3)

    # --------------------------------------
    # 2. DETECCIÓN YOLO
    # --------------------------------------
    results = model.predict(source=frame, imgsz=640, conf=0.75,
                            save=False, verbose=False)

    roya_area = 0

    color_box = (0, 0, 255)
    color_text = (0, 0, 255)
    thickness = 2

    # --------------------------------------
    # 3. DIBUJAR CAJAS
    # --------------------------------------
    for box in results[0].boxes:
        cls = int(box.cls[0])

        if cls == 0:  # clase roya
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            roya_area += w * h

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, thickness)
            cv2.putText(frame, "Roya",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color_text, 2)

    # --------------------------------------
    # 4. ÍNDICE DE DAÑO (LIMITADO A 100%)
    # --------------------------------------
    porcentaje = 0
    if hoja_area > 0:
        porcentaje = (roya_area / hoja_area) * 100

    porcentaje = min(porcentaje, 100)  # 🔒 límite máximo

    # --------------------------------------
    # 5. CÁLCULO DEL GRADO
    # --------------------------------------
    if porcentaje == 0:
        grado = 0
    elif 1 <= porcentaje <= 5:
        grado = 1
    elif 6 <= porcentaje <= 20:
        grado = 2
    elif 21 <= porcentaje <= 50:
        grado = 3
    else:  # 51–100
        grado = 4

    # --------------------------------------
    # 6. VISUALIZACIÓN
    # --------------------------------------
    cv2.putText(frame,
                f"Indice de Plaga: {porcentaje:.2f}%",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 255), 2)

    cv2.putText(frame,
                f"Grado: {grado}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (225, 0, 255), 2)


    cv2.imshow("Hoja + YOLO + Analisis", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
