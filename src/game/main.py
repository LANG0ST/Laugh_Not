import customtkinter as ctk
import tkinter as tk
import socket
import threading
import pyaudio
import wave
import os
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model

WIN_SCORE = 5

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

PORT = 5000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(os.path.dirname(BASE_DIR), "AI")
MODEL_PATH = os.path.join(AI_DIR, "emotion_model.h5")
CASCADE_PATH = os.path.join(AI_DIR, "haarcascade_frontalface_default.xml")

BG_COLOR = "#FFF8E1"
PANEL_COLOR = "#FFE0B2"
ACCENT_COLOR = "#FFB74D"
ACCENT_DARK = "#FB8C00"
TEXT_DARK = "#4E342E"


def record_audio(filename="joke.wav", duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    return filename


def play_audio(filename="received.wav"):
    wf = wave.open(filename, "rb")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    data = wf.readframes(1024)

    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()


def send_audio(conn, filename):
    if conn is None:
        return

    with open(filename, "rb") as f:
        audio_data = f.read()

    conn.sendall(b"AUDIO_START")
    conn.sendall(audio_data)
    conn.sendall(b"AUDIO_END")


class MainMenu(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LAUGHN'T AI – Main Menu")
        self.geometry("900x550")
        self.resizable(False, False)
        self.configure(fg_color=BG_COLOR)

        self.main_frame = ctk.CTkFrame(self, corner_radius=30, fg_color=PANEL_COLOR)
        self.main_frame.pack(expand=True, fill="both", padx=40, pady=40)

        title_label = ctk.CTkLabel(
            self.main_frame,
            text="LAUGHN'T AI",
            font=ctk.CTkFont(size=38, weight="bold"),
            text_color=TEXT_DARK
        )
        title_label.pack(pady=(25, 10))

        subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="Two players. One tells a joke. The other tries not to laugh.",
            font=ctk.CTkFont(size=16),
            text_color=TEXT_DARK
        )
        subtitle_label.pack(pady=(0, 25))

        btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        btn_frame.pack(pady=15)

        host_button = ctk.CTkButton(
            btn_frame,
            text="Host Game",
            width=230,
            height=50,
            fg_color=ACCENT_COLOR,
            hover_color=ACCENT_DARK,
            text_color="white",
            font=ctk.CTkFont(size=20, weight="bold"),
            command=self.open_host_screen
        )
        host_button.pack(pady=10)

        join_button = ctk.CTkButton(
            btn_frame,
            text="Join Game",
            width=230,
            height=50,
            fg_color="#FFCC80",
            hover_color="#FFB74D",
            text_color=TEXT_DARK,
            font=ctk.CTkFont(size=20, weight="bold"),
            command=self.open_join_screen
        )
        join_button.pack(pady=10)

        quit_button = ctk.CTkButton(
            self.main_frame,
            text="Exit",
            width=140,
            fg_color="#D7CCC8",
            hover_color="#BCAAA4",
            text_color=TEXT_DARK,
            font=ctk.CTkFont(size=14),
            command=self.destroy
        )
        quit_button.pack(pady=(30, 5))

        footer = ctk.CTkLabel(
            self.main_frame,
            text=f"First to {WIN_SCORE} points wins the match",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_DARK
        )
        footer.pack(side="bottom", pady=10)

    def open_host_screen(self):
        HostWindow(self)

    def open_join_screen(self):
        JoinWindow(self)


class HostWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)

        self.title("LAUGHN'T AI – Host")
        self.geometry("900x550")
        self.resizable(False, False)
        self.configure(fg_color=BG_COLOR)

        self.current_turn = "host"
        self.host_score = 0
        self.guest_score = 0
        self.game_over = False

        self.server_socket = None
        self.client_conn = None
        self.client_addr = None
        self.server_thread = None
        self.listening_thread = None

        self.stop_detection_flag = False
        self.laughed_result = False

        try:
            self.model = load_model(MODEL_PATH)
            self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        except Exception:
            self.model = None
            self.face_cascade = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.main_frame = ctk.CTkFrame(self, corner_radius=25, fg_color=PANEL_COLOR)
        self.main_frame.pack(expand=True, fill="both", padx=30, pady=30)

        top_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        top_frame.pack(fill="x", padx=20, pady=(10, 5))

        title = ctk.CTkLabel(
            top_frame,
            text="Host",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_DARK
        )
        title.pack(side="left")

        self.score_label = ctk.CTkLabel(
            top_frame,
            text=self.score_text(),
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color=TEXT_DARK
        )
        self.score_label.pack(side="right")

        center_frame = ctk.CTkFrame(self.main_frame, fg_color="#FFE9C6", corner_radius=25)
        center_frame.pack(fill="x", padx=20, pady=(10, 15))

        self.status_label = ctk.CTkLabel(
            center_frame,
            text="Server not started",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT_DARK
        )
        self.status_label.pack(pady=(18, 5))

        self.info_label = ctk.CTkLabel(
            center_frame,
            text="Press Start Server and wait for the guest, then record a joke.",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_DARK,
            justify="center"
        )
        self.info_label.pack(pady=(0, 18))

        bottom_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        bottom_frame.pack(fill="x", padx=20, pady=(5, 10))

        ip_frame = ctk.CTkFrame(bottom_frame, fg_color="#FFE0B2", corner_radius=20)
        ip_frame.pack(side="left", padx=(0, 10), pady=5)

        label_ip = ctk.CTkLabel(
            ip_frame,
            text="Your IP:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_DARK
        )
        label_ip.pack(side="left", padx=(15, 5), pady=10)

        self.ip_value_label = ctk.CTkLabel(
            ip_frame,
            text=self.get_local_ip(),
            font=ctk.CTkFont(size=14),
            text_color=TEXT_DARK
        )
        self.ip_value_label.pack(side="left", padx=(0, 15), pady=10)

        btn_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        btn_frame.pack(side="right", pady=5)

        self.start_button = ctk.CTkButton(
            btn_frame,
            text="Start Server",
            fg_color=ACCENT_COLOR,
            hover_color=ACCENT_DARK,
            text_color="white",
            width=170,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.start_server
        )
        self.start_button.pack(pady=6)

        self.record_button = ctk.CTkButton(
            btn_frame,
            text="Record Joke (5s)",
            fg_color="#FFCC80",
            hover_color="#FFB74D",
            text_color=TEXT_DARK,
            width=190,
            height=42,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled",
            command=self.on_record_joke_clicked
        )
        self.record_button.pack(pady=6)

        footer = ctk.CTkLabel(
            self.main_frame,
            text=f"First to {WIN_SCORE} points wins the match",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_DARK
        )
        footer.pack(side="bottom", pady=8)

    def score_text(self):
        return f"Host {self.host_score}  -  {self.guest_score} Guest"

    def update_score_label(self):
        self.score_label.configure(text=self.score_text())

    def get_local_ip(self):
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

    def start_server(self):
        if self.game_over:
            self.set_status("Game is over. Restart for a new match.")
            return

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind(("", PORT))
            self.server_socket.listen(1)

            self.set_status(f"Server started on port {PORT}. Waiting for guest...")
            self.info_label.configure(text="Share your IP with the guest.")
            self.start_button.configure(state="disabled")

            self.server_thread = threading.Thread(target=self.accept_client, daemon=True)
            self.server_thread.start()
        except Exception as e:
            self.set_status(f"Error: {e}")

    def accept_client(self):
        conn, addr = self.server_socket.accept()
        self.client_conn = conn
        self.client_addr = addr

        def ui():
            self.set_status(f"Guest connected from {addr[0]}:{addr[1]}")
            self.info_label.configure(text="Guest connected. Record a joke to start the round.")
            self.record_button.configure(state="normal")

        self.after(0, ui)

        self.listening_thread = threading.Thread(target=self.listen_to_client, daemon=True)
        self.listening_thread.start()

    def listen_to_client(self):
        json_buffer = b""
        audio_buffer = b""
        receiving_audio = False

        while True:
            try:
                data = self.client_conn.recv(4096)
                if not data:
                    break

                if b"AUDIO_START" in data:
                    receiving_audio = True
                    audio_buffer += data
                    continue

                if receiving_audio:
                    audio_buffer += data
                    if b"AUDIO_END" in audio_buffer:
                        try:
                            raw = audio_buffer.split(b"AUDIO_START")[1].split(b"AUDIO_END")[0]
                        except Exception:
                            self.set_status("Audio decode error.")
                            audio_buffer = b""
                            receiving_audio = False
                            continue

                        self.process_client_audio(raw)
                        audio_buffer = b""
                        receiving_audio = False

                    continue

                json_buffer += data

                while b"\n" in json_buffer:
                    line, json_buffer = json_buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode())
                    except Exception:
                        continue
                    self.handle_client_message(msg)

            except Exception as e:
                self.set_status(f"Receive error: {e}")
                break

    def handle_client_message(self, msg):
        if msg.get("type") == "result":
            laughed = bool(msg.get("laugh"))

            if laughed:
                self.host_score += 1
                txt = "Guest laughed at your joke. Host gains 1 point."
            else:
                self.guest_score += 1
                txt = "Guest stayed serious. Guest gains 1 point."

            def ui():
                self.update_score_label()
                self.info_label.configure(text=txt)

            self.after(0, ui)

            if self.check_game_over():
                return

            self.send_turn_update("guest")

        elif msg.get("type") == "game_over":
            winner = msg.get("winner", "Unknown")
            h = msg.get("host_score", 0)
            g = msg.get("guest_score", 0)
            text = f"Game over. Winner: {winner}. Final score: Host {h} - {g} Guest."
            self.after(0, lambda: self.info_label.configure(text=text))

    def check_game_over(self):
        if self.host_score >= WIN_SCORE or self.guest_score >= WIN_SCORE:
            self.game_over = True
            winner = "Host" if self.host_score > self.guest_score else "Guest"

            def ui():
                self.update_score_label()
                self.set_status("Game over.")
                self.info_label.configure(
                    text=f"Winner: {winner}. Final score: Host {self.host_score} - {self.guest_score} Guest."
                )
                self.record_button.configure(state="disabled")

            self.after(0, ui)

            try:
                msg = {
                    "type": "game_over",
                    "winner": winner,
                    "host_score": self.host_score,
                    "guest_score": self.guest_score
                }
                line = (json.dumps(msg) + "\n").encode("utf-8")
                self.client_conn.sendall(line)
            except Exception:
                pass

            return True
        return False

    def process_client_audio(self, raw_audio):
        if self.game_over:
            return

        filename = "client_joke.wav"
        with open(filename, "wb") as f:
            f.write(raw_audio)

        self.after(0, lambda: self.info_label.configure(text="Playing guest's joke. Try not to laugh."))
        self.after(0, lambda: self.set_status("Host is listening to the guest."))

        self.start_realtime_detection()
        play_audio(filename)

    def start_realtime_detection(self):
        if self.model is None or self.face_cascade is None:
            return

        self.stop_detection_flag = False
        self.laughed_result = False

        t = threading.Thread(target=self.realtime_detection_loop, daemon=True)
        t.start()

        timer = threading.Timer(5.0, self.stop_realtime_detection)
        timer.start()

    def stop_realtime_detection(self):
        self.stop_detection_flag = True

    def realtime_detection_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        while not self.stop_detection_flag:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                roi = roi.astype("float32") / 255.0
                roi = np.expand_dims(np.expand_dims(roi, -1), 0)
                preds = self.model.predict(roi, verbose=0)
                if int(np.argmax(preds)) == 3:
                    self.laughed_result = True

            cv2.imshow("Host Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.finish_host_detection()

    def finish_host_detection(self):
        if self.game_over:
            return

        laughed = self.laughed_result

        if laughed:
            self.guest_score += 1
            txt = "Host laughed at the guest's joke. Guest gains 1 point."
        else:
            self.host_score += 1
            txt = "Host stayed serious. Host gains 1 point."

        def ui():
            self.update_score_label()
            self.info_label.configure(text=txt)
            self.set_status("Host turn. Record a new joke.")
            self.record_button.configure(state="normal")

        self.after(0, ui)

        try:
            msg_res = {
                "type": "result",
                "laugh": bool(laughed),
                "host_score": self.host_score,
                "guest_score": self.guest_score
            }
            line_res = (json.dumps(msg_res) + "\n").encode("utf-8")
            self.client_conn.sendall(line_res)
        except Exception:
            pass

        if self.check_game_over():
            return

        self.send_turn_update("host")

    def send_turn_update(self, turn):
        try:
            msg = {
                "type": "turn",
                "turn": turn,
                "host_score": self.host_score,
                "guest_score": self.guest_score
            }
            line = (json.dumps(msg) + "\n").encode("utf-8")
            self.client_conn.sendall(line)
        except Exception:
            pass

    def on_record_joke_clicked(self):
        if self.game_over:
            self.set_status("Game is over. Restart the app.")
            return
        self.record_button.configure(state="disabled")
        threading.Thread(target=self.record_and_send_thread, daemon=True).start()

    def record_and_send_thread(self):
        try:
            filename = record_audio(filename="host_joke.wav", duration=5)
            send_audio(self.client_conn, filename)
            self.set_status("Guest is listening to your joke.")
            self.info_label.configure(text="Waiting for the guest reaction.")
        except Exception as e:
            self.set_status(f"Error: {e}")
            self.record_button.configure(state="normal")

    def set_status(self, txt):
        try:
            self.status_label.configure(text=txt)
        except Exception:
            pass

    def on_close(self):
        try:
            if self.client_conn:
                self.client_conn.close()
            if self.server_socket:
                self.server_socket.close()
        except Exception:
            pass
        self.destroy()


class JoinWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)

        self.title("LAUGHN'T AI – Guest")
        self.geometry("900x550")
        self.resizable(False, False)
        self.configure(fg_color=BG_COLOR)

        self.current_turn = "host"
        self.client_socket = None
        self.listen_thread = None

        self.stop_detection_flag = False
        self.laughed_result = False

        self.host_score = 0
        self.guest_score = 0

        try:
            self.model = load_model(MODEL_PATH)
            self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        except Exception:
            self.model = None
            self.face_cascade = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.main_frame = ctk.CTkFrame(self, corner_radius=25, fg_color=PANEL_COLOR)
        self.main_frame.pack(expand=True, fill="both", padx=30, pady=30)

        top_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        top_frame.pack(fill="x", padx=20, pady=(10, 5))

        title = ctk.CTkLabel(
            top_frame,
            text="Guest",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_DARK
        )
        title.pack(side="left")

        self.score_label = ctk.CTkLabel(
            top_frame,
            text=self.score_text(),
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color=TEXT_DARK
        )
        self.score_label.pack(side="right")

        ip_frame = ctk.CTkFrame(self.main_frame, fg_color="#FFECB3", corner_radius=20)
        ip_frame.pack(fill="x", padx=20, pady=(5, 10))

        label_host = ctk.CTkLabel(
            ip_frame,
            text="Host IP:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_DARK
        )
        label_host.pack(pady=(10, 0))

        self.ip_entry = ctk.CTkEntry(ip_frame, width=230)
        self.ip_entry.insert(0, "192.168.")
        self.ip_entry.pack(pady=5)

        connect_btn = ctk.CTkButton(
            ip_frame,
            text="Connect",
            fg_color=ACCENT_COLOR,
            hover_color=ACCENT_DARK,
            text_color="white",
            width=150,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.connect_to_host
        )
        connect_btn.pack(pady=(5, 10))

        self.status_label = ctk.CTkLabel(
            ip_frame,
            text="Not connected.",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_DARK
        )
        self.status_label.pack(pady=(0, 10))

        center_frame = ctk.CTkFrame(self.main_frame, fg_color="#FFE9C6", corner_radius=25)
        center_frame.pack(fill="x", padx=20, pady=10)

        self.audio_label = ctk.CTkLabel(
            center_frame,
            text="Waiting for host...",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT_DARK
        )
        self.audio_label.pack(pady=(16, 4))

        self.info_label = ctk.CTkLabel(
            center_frame,
            text="When the host tells a joke, try to stay serious.",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_DARK
        )
        self.info_label.pack(pady=(0, 16))

        self.record_button = ctk.CTkButton(
            self.main_frame,
            text="Record Joke",
            state="disabled",
            fg_color="#FFCC80",
            hover_color="#FFB74D",
            text_color=TEXT_DARK,
            width=200,
            height=44,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.record_and_send_joke
        )
        self.record_button.pack(pady=18)

        footer = ctk.CTkLabel(
            self.main_frame,
            text=f"First to {WIN_SCORE} points wins the match",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_DARK
        )
        footer.pack(side="bottom", pady=8)

    def score_text(self):
        return f"Host {self.host_score}  -  {self.guest_score} Guest"

    def update_score_label(self):
        self.score_label.configure(text=self.score_text())

    def connect_to_host(self):
        try:
            ip = self.ip_entry.get().strip()
            self.client_socket = socket.socket()
            self.client_socket.connect((ip, PORT))

            self.set_status("Connected to host.")
            self.info_label.configure(text="Waiting for the host to record a joke.")

            self.listen_thread = threading.Thread(
                target=self.listen_from_host,
                daemon=True
            )
            self.listen_thread.start()

        except Exception as e:
            self.set_status(f"Connection failed: {e}")

    def listen_from_host(self):
        json_buffer = b""
        audio_buffer = b""
        receiving_audio = False

        while True:
            try:
                data = self.client_socket.recv(4096)
                if not data:
                    break

                if b"AUDIO_START" in data:
                    receiving_audio = True
                    audio_buffer += data
                    continue

                if receiving_audio:
                    audio_buffer += data
                    if b"AUDIO_END" in audio_buffer:
                        try:
                            audio_bytes = audio_buffer.split(b"AUDIO_START")[1].split(b"AUDIO_END")[0]
                        except Exception:
                            audio_buffer = b""
                            receiving_audio = False
                            continue

                        filename = "received.wav"
                        with open(filename, "wb") as f:
                            f.write(audio_bytes)

                        self.after(0, lambda: self.audio_label.configure(text="Listening to host's joke..."))
                        self.after(0, lambda: self.info_label.configure(
                            text="Stay serious while the joke is playing."
                        ))

                        self.start_realtime_detection()
                        play_audio(filename)
                        self.after(5000, lambda: setattr(self, "stop_detection_flag", True))

                        audio_buffer = b""
                        receiving_audio = False

                    continue

                json_buffer += data

                while b"\n" in json_buffer:
                    line, json_buffer = json_buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode())
                    except Exception:
                        continue
                    self.handle_json_message(msg)

            except Exception as e:
                self.set_status(f"Receive error: {e}")
                break

    def handle_json_message(self, msg):
        if msg.get("type") == "turn":
            turn = msg.get("turn")
            self.current_turn = turn

            host_score = msg.get("host_score", None)
            guest_score = msg.get("guest_score", None)
            if host_score is not None and guest_score is not None:
                self.host_score = host_score
                self.guest_score = guest_score
                self.update_score_label()

            if turn == "guest":
                def enable():
                    self.record_button.configure(state="normal")
                    self.audio_label.configure(text="Your turn to tell a joke.")
                    self.info_label.configure(text="Record a joke and try to make the host laugh.")
                self.after(0, enable)

            elif turn == "host":
                def disable():
                    self.record_button.configure(state="disabled")
                    self.audio_label.configure(text="Waiting for host's joke...")
                    self.info_label.configure(text="Get ready. Stay serious while the host speaks.")
                self.after(0, disable)

        elif msg.get("type") == "result":
            laughed = bool(msg.get("laugh"))
            host_score = msg.get("host_score", None)
            guest_score = msg.get("guest_score", None)
            if host_score is not None and guest_score is not None:
                self.host_score = host_score
                self.guest_score = guest_score
                self.update_score_label()

            if laughed:
                txt = "Host laughed at your joke. You gained 1 point."
            else:
                txt = "Host stayed serious. No point this round."
            self.after(0, lambda: self.info_label.configure(text=txt))

        elif msg.get("type") == "game_over":
            winner = msg.get("winner", "Unknown")
            h = msg.get("host_score", 0)
            g = msg.get("guest_score", 0)

            def ui():
                self.audio_label.configure(text="Game over.")
                self.info_label.configure(
                    text=f"Winner: {winner}. Final score: Host {h} - {g} Guest."
                )
                self.record_button.configure(state="disabled")

            self.after(0, ui)

    def start_realtime_detection(self):
        if self.model is None or self.face_cascade is None:
            return

        self.stop_detection_flag = False
        self.laughed_result = False
        threading.Thread(target=self.realtime_detection_loop, daemon=True).start()

    def realtime_detection_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        while not self.stop_detection_flag:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                roi = roi.astype("float32") / 255.0
                roi = np.expand_dims(np.expand_dims(roi, -1), 0)
                preds = self.model.predict(roi, verbose=0)
                if int(np.argmax(preds)) == 3:
                    self.laughed_result = True

            cv2.imshow("Guest Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        result_msg = {"type": "result", "laugh": bool(self.laughed_result)}
        try:
            line = (json.dumps(result_msg) + "\n").encode("utf-8")
            self.client_socket.sendall(line)
        except Exception:
            pass

        txt = "You laughed during the joke." if self.laughed_result else "You stayed serious during the joke."
        self.after(0, lambda: self.audio_label.configure(text=txt))

    def record_and_send_joke(self):
        self.record_button.configure(state="disabled")
        self.audio_label.configure(text="Recording your joke...")
        self.info_label.configure(text="Speak into the microphone.")

        threading.Thread(target=self.record_and_send_thread, daemon=True).start()

    def record_and_send_thread(self):
        try:
            filename = record_audio(filename="client_joke.wav", duration=5)
            self.audio_label.configure(text="Sending joke to host...")
            send_audio(self.client_socket, filename)
            self.info_label.configure(text="Host is listening to your joke.")
        except Exception:
            self.set_status("Error sending audio.")

    def set_status(self, txt):
        try:
            self.status_label.configure(text=txt)
        except Exception:
            pass

    def on_close(self):
        try:
            if self.client_socket:
                self.client_socket.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = MainMenu()
    app.mainloop()
