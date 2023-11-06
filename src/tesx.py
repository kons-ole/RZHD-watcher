import cv2

# Открываем видеофайл для чтения
cap = cv2.VideoCapture('../data/Archery/v_Archery_g01_c01.avi')  # Замените 'video.avi' на путь к вашему видеофайлу

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Здесь вы можете выполнять обработку каждого кадра (например, распознавание объектов, фильтрацию и т. д.)
    
    # Вывод кадра
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
