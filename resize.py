import cv2

# Load Haar cascades ---
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

# smoothing counters ---
smile_frames = 0
neutral_frames = 0
label = "Not Smiling"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # helps in uneven lighting

    # detect faces ---
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(120, 120))

    smiling_now = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # detect eyes (optional) ---
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=15, minSize=(25, 25))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # detect smile ONLY in lower half of the face ---
        mouth_roi_gray = roi_gray[int(h*0.5):h, 0:w]
        smiles = smile_cascade.detectMultiScale(
            mouth_roi_gray,
            scaleFactor=1.7,   # try 1.5–1.9
            minNeighbors=22,   # try 15–30 (higher = stricter)
            minSize=(25, 25)
        )

        if len(smiles) > 0:
            smiling_now = True

        # draw mouth ROI (debug): uncomment if needed
        # cv2.rectangle(roi_color, (0, int(h*0.5)), (w, h), (255, 255, 0), 1)

    # ---- debounce / smoothing over frames ----
    if smiling_now:
        smile_frames += 1
        neutral_frames = 0
    else:
        neutral_frames += 1
        smile_frames = 0

    if smile_frames >= 3:        # needs 3 consecutive frames to switch ON
        label = "Smiling"
    if neutral_frames >= 7:      # needs 7 consecutive frames to switch OFF
        label = "Not Smiling"

    # Put label for each detected face (or top-left if none)
    color = (0, 200, 0) if label == "Smiling" else (0, 0, 255)
    for (x, y, w, h) in faces:
        cv2.putText(frame, label, (x, max(30, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    cv2.imshow("Face / Eyes / Smile", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
