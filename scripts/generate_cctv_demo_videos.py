"""
Generates high-definition, realistic CCTV surveillance demo video loops in H.264 (avc1)
for the IBVAP Command Center dashboard.
"""

import cv2
import numpy as np
import os
import math

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "ibvap-platform", "public", "videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 640, 360
FPS = 25
NUM_FRAMES = 150  # 6-second seamless loop


def create_thermal_perimeter_video():
    """Generates a Thermal Night-Vision perimeter intrusion video."""
    video_path = os.path.join(OUTPUT_DIR, "cctv_thermal_perimeter.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), FPS, (WIDTH, HEIGHT))

    for f in range(NUM_FRAMES):
        # Dark thermal background with gradient
        frame = np.full((HEIGHT, WIDTH, 3), 15, dtype=np.uint8)
        frame[:, :, 0] = 25  # slight blue-gray cold tone
        frame[:, :, 1] = 20

        # Terrain background (mountain contours)
        for x in range(WIDTH):
            ground_y = int(240 + 20 * math.sin(x * 0.015) + 10 * math.cos(x * 0.03))
            frame[ground_y:, x] = [35, 45, 30]  # Cold ground

        # Fence wire lines
        cv2.line(frame, (0, 200), (WIDTH, 220), (50, 60, 50), 1)
        cv2.line(frame, (0, 230), (WIDTH, 250), (50, 60, 50), 1)
        for px in range(20, WIDTH, 70):
            cv2.line(frame, (px, 180), (px, 270), (60, 70, 60), 2)

        # Virtual Fence tripwire line (glowing cyan / red)
        fence_x = int(WIDTH * 0.45)
        pulse = int(180 + 70 * math.sin(f * 0.2))
        cv2.line(frame, (fence_x, 80), (fence_x, 300), (0, pulse, 255), 2)
        cv2.putText(frame, "VIRTUAL FENCE BOUNDARY", (fence_x + 5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, pulse, 255), 1)

        # Moving Thermal Intruder (Humanoid heat signature)
        # Walks from x = 520 to x = 200
        progress = (f % NUM_FRAMES) / NUM_FRAMES
        target_x = int(520 - progress * 320)
        target_y = int(200 + 10 * math.sin(f * 0.3))  # subtle walk bobbing

        # Draw glowing thermal heat signature (white/yellow/red glow)
        # Head
        cv2.circle(frame, (target_x, target_y - 25), 10, (255, 255, 255), -1)
        cv2.circle(frame, (target_x, target_y - 25), 15, (0, 220, 255), 2)  # Heat halo
        # Torso
        cv2.ellipse(frame, (target_x, target_y), (14, 22), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (target_x, target_y), (18, 26), 0, 0, 360, (0, 180, 255), 2)
        # Legs
        leg_offset = int(6 * math.sin(f * 0.5))
        cv2.line(frame, (target_x - 5, target_y + 20), (target_x - 8 + leg_offset, target_y + 55), (255, 255, 255), 4)
        cv2.line(frame, (target_x + 5, target_y + 20), (target_x + 8 - leg_offset, target_y + 55), (255, 255, 255), 4)

        # Tactical HUD Overlays
        sec = f // FPS
        msec = int((f % FPS) * (1000 / FPS))
        cv2.putText(frame, f"CAM-07 • NORTH GATE [THERMAL IR] 23:41:{sec:02d}.{msec:03d}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, "STATUS: ACTIVE REC • 30 FPS • AGC ON", (WIDTH - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 255, 50), 1)
        cv2.putText(frame, "LAT: 34.1284° N  LNG: 74.8012° E  ALT: 2140M", (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Crosshair in center
        cx, cy = WIDTH // 2, HEIGHT // 2
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

        # Slight sensor scanline noise
        noise = np.random.randint(-5, 5, (HEIGHT, WIDTH, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        out.write(frame)

    out.release()
    print(f"[✓] Created: {video_path} ({os.path.getsize(video_path)} bytes)")


def create_anpr_checkpoint_video():
    """Generates an ANPR Highway / Checkpoint vehicle tracking video."""
    video_path = os.path.join(OUTPUT_DIR, "cctv_anpr_checkpoint.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), FPS, (WIDTH, HEIGHT))

    for f in range(NUM_FRAMES):
        # Road perspective background
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)

        # Asphalt Road (Trapezoid perspective)
        road_pts = np.array([[int(WIDTH * 0.35), 100], [int(WIDTH * 0.65), 100], [WIDTH - 20, HEIGHT], [20, HEIGHT]], np.int32)
        cv2.fillPoly(frame, [road_pts], (40, 42, 45))

        # Lane markings
        cv2.line(frame, (WIDTH // 2, 100), (WIDTH // 2, HEIGHT), (200, 200, 200), 2)
        # Shoulder guardrails
        cv2.line(frame, (int(WIDTH * 0.35), 100), (20, HEIGHT), (255, 255, 0), 2)
        cv2.line(frame, (int(WIDTH * 0.65), 100), (WIDTH - 20, HEIGHT), (255, 255, 0), 2)

        # Security Checkpoint Gate Barrier (left side)
        cv2.rectangle(frame, (40, 160), (70, 280), (100, 100, 100), -1)
        # Red-white striped barrier boom
        cv2.line(frame, (70, 180), (int(WIDTH * 0.55), 210), (0, 0, 255), 6)

        # Approaching Vehicle with ANPR Plate
        # Scales from small (y=120) to close (y=270)
        progress = (f % NUM_FRAMES) / NUM_FRAMES
        scale = 0.4 + 0.9 * progress
        v_w = int(120 * scale)
        v_h = int(80 * scale)
        vx = int(WIDTH * 0.52 - v_w / 2)
        vy = int(110 + progress * 150)

        # Vehicle Body (Dark Military Utility Truck)
        cv2.rectangle(frame, (vx, vy), (vx + v_w, vy + v_h), (45, 55, 45), -1)
        cv2.rectangle(frame, (vx + 5, vy - int(25 * scale)), (vx + v_w - 5, vy), (35, 45, 35), -1)  # Cabin
        # Windshield
        cv2.rectangle(frame, (vx + 10, vy - int(22 * scale)), (vx + v_w - 10, vy - 4), (100, 140, 160), -1)

        # Headlights with light cones
        hl_y = vy + int(v_h * 0.45)
        cv2.circle(frame, (vx + int(v_w * 0.18), hl_y), int(6 * scale), (255, 255, 220), -1)
        cv2.circle(frame, (vx + int(v_w * 0.82), hl_y), int(6 * scale), (255, 255, 220), -1)

        # License Plate Banner on front bumper
        pw = int(65 * scale)
        ph = int(18 * scale)
        px = vx + (v_w - pw) // 2
        py = vy + int(v_h * 0.72)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), 1)

        # Text on plate if large enough
        if scale > 0.6:
            cv2.putText(frame, "DL01AB1234", (px + 3, py + ph - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28 * scale, (0, 0, 0), 1)

        # ANPR OCR Active Scanning Bounding Box (Green/Amber reticle)
        cv2.rectangle(frame, (px - 4, py - 4), (px + pw + 4, py + ph + 4), (0, 255, 0), 1)
        cv2.putText(frame, "ANPR LOCK: DL01AB1234 (CONF: 97.4%)", (px - 20, max(20, py - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Tactical HUD
        sec = f // FPS
        msec = int((f % FPS) * (1000 / FPS))
        cv2.putText(frame, f"CAM-04 • VALLEY ACCESS ROAD [ANPR OPTICAL] 23:42:{sec:02d}.{msec:03d}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, "BOP BRAVO CHECKPOINT • SPEED: 34 KM/H", (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(frame, "ANPR ENGINE: ACTIVE (WATCHLIST INDEXED)", (WIDTH - 270, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

        out.write(frame)

    out.release()
    print(f"[✓] Created: {video_path} ({os.path.getsize(video_path)} bytes)")


def create_fence_sector_video():
    """Generates a Sector 3 boundary fence patrol CCTV video."""
    video_path = os.path.join(OUTPUT_DIR, "cctv_fence_sector3.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), FPS, (WIDTH, HEIGHT))

    for f in range(NUM_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)
        frame[:, :, 1] = 28  # Night-vision phosphor green tint
        frame[:, :, 0] = 18

        # Fence perspective
        for y_wire in range(120, 290, 25):
            cv2.line(frame, (0, y_wire - 40), (WIDTH, y_wire + 30), (40, 75, 40), 1)
            # Barbed wire knots
            for bx in range(20, WIDTH, 35):
                by = int((y_wire - 40) + (bx / WIDTH) * 70)
                cv2.circle(frame, (bx, by), 2, (70, 110, 70), -1)

        # Fence vertical steel poles
        for px in range(50, WIDTH, 90):
            cv2.line(frame, (px, 70), (px, 320), (60, 100, 60), 3)

        # Sweeping military searchlight cone across the fence
        sweep_x = int(WIDTH * 0.5 + WIDTH * 0.35 * math.sin(f * 0.08))
        overlay = frame.copy()
        cv2.ellipse(overlay, (sweep_x, 220), (110, 50), 0, 0, 360, (140, 220, 140), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Human figure stealthily moving in shadows
        p_x = int(120 + 300 * (f / NUM_FRAMES))
        p_y = 195
        # Silhouette
        cv2.circle(frame, (p_x, p_y - 20), 8, (10, 15, 10), -1)
        cv2.ellipse(frame, (p_x, p_y), (10, 18), 0, 0, 360, (10, 15, 10), -1)
        cv2.line(frame, (p_x - 3, p_y + 16), (p_x - 6, p_y + 42), (10, 15, 10), 3)
        cv2.line(frame, (p_x + 3, p_y + 16), (p_x + 6, p_y + 42), (10, 15, 10), 3)

        # AI Computer Vision Target Bounding Box
        cv2.rectangle(frame, (p_x - 16, p_y - 32), (p_x + 16, p_y + 44), (0, 0, 255), 2)
        cv2.rectangle(frame, (p_x - 16, p_y - 48), (p_x + 85, p_y - 32), (0, 0, 255), -1)
        cv2.putText(frame, "INTRUDER 98.2%", (p_x - 13, p_y - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        # Tactical HUD
        sec = f // FPS
        msec = int((f % FPS) * (1000 / FPS))
        cv2.putText(frame, f"CAM-08 • SECTOR 3 FENCE LINE [NIGHT VISION] 23:41:{sec:02d}.{msec:03d}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, "BREACH DETECTION: ARMED • SECTOR 3 NORTH", (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        cv2.putText(frame, "ZOOM: 2.4X • GAIN: +14dB", (WIDTH - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

        out.write(frame)

    out.release()
    print(f"[✓] Created: {video_path} ({os.path.getsize(video_path)} bytes)")


def create_gorge_recon_video():
    """Generates a River Gorge Watchtower Recon pan-tilt-zoom video."""
    video_path = os.path.join(OUTPUT_DIR, "cctv_gorge_recon.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), FPS, (WIDTH, HEIGHT))

    for f in range(NUM_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 18, dtype=np.uint8)

        # River gorge depth curves (water reflection and valley cliffs)
        for y in range(140, HEIGHT):
            curve_left = int(180 - (y - 140) * 0.4 + 15 * math.sin(f * 0.05 + y * 0.03))
            curve_right = int(460 + (y - 140) * 0.4 - 15 * math.cos(f * 0.05 + y * 0.03))
            # River water
            frame[y, curve_left:curve_right] = [50, 45, 25]

        # Mountain ridgelines
        for x in range(WIDTH):
            ridge1 = int(120 + 30 * math.sin(x * 0.01))
            frame[ridge1:160, x] = [30, 28, 26]

        # Recon Radar Sweep Reticle
        cx, cy = int(WIDTH * 0.48), int(HEIGHT * 0.52)
        angle = (f * 4.0) % 360
        rad = math.radians(angle)
        sweep_end_x = int(cx + 80 * math.cos(rad))
        sweep_end_y = int(cy + 80 * math.sin(rad))
        cv2.circle(frame, (cx, cy), 80, (0, 200, 255), 1)
        cv2.circle(frame, (cx, cy), 40, (0, 150, 200), 1)
        cv2.line(frame, (cx, cy), (sweep_end_x, sweep_end_y), (0, 255, 255), 1)

        # Tactical HUD
        sec = f // FPS
        msec = int((f % FPS) * (1000 / FPS))
        cv2.putText(frame, f"CAM-11 • RIVER GORGE TRAIL [PTZ INFRARED] 23:42:{sec:02d}.{msec:03d}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, "BOP CHARLIE • PTZ BEARING: 312° NW • TILT: -14°", (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(frame, "EDGE LOCAL AI: ONLINE", (WIDTH - 190, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)

        out.write(frame)

    out.release()
    print(f"[✓] Created: {video_path} ({os.path.getsize(video_path)} bytes)")


if __name__ == "__main__":
    print("Generating H.264 CCTV demo surveillance videos...")
    create_thermal_perimeter_video()
    create_anpr_checkpoint_video()
    create_fence_sector_video()
    create_gorge_recon_video()
    print("All 4 live CCTV demo video loops created successfully!")
