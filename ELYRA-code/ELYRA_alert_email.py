import os
import smtplib
import cv2
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# ===== CONFIG =====
BASE_PATH = "/mnt/nfs/recordings/"
TODAY = datetime.now().strftime("%Y-%m-%d")
TODAY_FOLDER = os.path.join(BASE_PATH, TODAY)
CACHE_FILE = f"/home/lylim/AIoTCam/notified_{TODAY}.txt"
EMAIL_SENDER = "lylim0371@gmail.com"
EMAIL_PASSWORD = "---"  # Use Gmail App Password
EMAIL_RECEIVER = "tayjunshengmauduit@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_SUBJECT = f"🎥 New Lab Recording Detected {TODAY}"
# ===================

def load_notified():
    if not os.path.exists(CACHE_FILE):
        return set()
    with open(CACHE_FILE, "r") as f:
        return set(line.strip() for line in f)

def save_notified(filepaths):
    with open(CACHE_FILE, "a") as f:
        for path in filepaths:
            f.write(path + "\n")

def extract_first_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
    cap.release()

def send_email(video_path):
    filename = os.path.basename(video_path)
    frame_path = f"/tmp/{filename}_frame.jpg"

    try:
        print(f"[EXTRACTING FRAME] {filename}")
        extract_first_frame(video_path, frame_path)

        msg = MIMEMultipart()
        msg["Subject"] = EMAIL_SUBJECT
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        body = MIMEText(f"""
Dear Lab Executive,

A new lab recording has been detected for {TODAY}.

📁 File Name: {filename}
📍 File Path: {video_path}

Attached is the first frame of the video for your reference.

Best Regards,  
Sunway ELYRA Developer Team.
""")
        msg.attach(body)

        with open(frame_path, "rb") as f:
            img = MIMEApplication(f.read(), _subtype="jpeg")
            img.add_header("Content-Disposition", "attachment", filename=f"{filename}_frame.jpg")
            msg.attach(img)
        print(f"[SENDING EMAIL] To: {EMAIL_RECEIVER}")

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"[EMAIL SENT] {video_path}")
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")
    finally:
        if os.path.exists(frame_path):
            os.remove(frame_path)

def main():
    print("=========== [🕒 SCRIPT STARTED] ===========")
    print(f"[🔍 SCANNING] {TODAY_FOLDER} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(TODAY_FOLDER):
        print(f"[📁 SKIPPED] Today's folder does not exist: {TODAY_FOLDER}")
        print("=========== [✅ SCRIPT ENDED] ===========\n")
        return

    notified = load_notified()
    new_files = []

    for file in os.listdir(TODAY_FOLDER):
        if file.endswith(".mp4"):
            full_path = os.path.join(TODAY_FOLDER, file)
            if full_path in notified:
                print(f"[⏩ SKIPPED] Already notified: {file}")
                continue
            if os.path.getsize(full_path) < 1024:
                print(f"[⚠️ SKIPPED] File too small: {file}")
                continue
            print(f"[🎞️ NEW VIDEO] {file}")
            send_email(full_path)
            new_files.append(full_path)

    if new_files:
        save_notified(new_files)
        print(f"[🧠 CACHE UPDATED] {len(new_files)} new files added.")
    else:
        print("[📭 NO NEW FILES] Nothing to notify.")

    print("=========== [✅ SCRIPT ENDED] ===========\n")

if __name__ == "__main__":
    main()

