import cv2

while True:
    for i in range(10):  # กำหนด loop จำนวน 10 รอบ
        if True:
            x = True
            Z = True
            if x:
                print("S")
            elif Z:
                print("A")
        for j in range(10):  # กำหนด loop จำนวน 10 รอบภายใน loop ของ i
            if x:
                print("S2")
            elif Z:
                print("A1")

    # ตรวจสอบการกดปุ่ม 'q' เพื่อหยุดโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดหน้าต่างทั้งหมดเมื่อหยุดโปรแกรม
cv2.destroyAllWindows()
