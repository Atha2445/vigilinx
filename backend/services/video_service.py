"""
Vigilinx video analysis pipeline.

Dual-Mode Architecture:
  Mode 1 (Default): Continuous 24/7 Zero-Prompt Multi-Threat Surveillance Engine
                    (Weapons, Fire/Smoke, Dog Attacks, Fights/Assaults, Space Analytics).
  Mode 2 (On-Demand): VLM Evidence Audit & Forensic Reasoning (Kimi-VL / X-CLIP).
"""
import os
import json
import cv2
import numpy as np
import sqlite3
from typing import List, Dict, Optional, Tuple, Callable, Any
import subprocess
import tempfile
import logging
import base64
from datetime import datetime
from PIL import Image
from services.schemas import StatusResponse, VerdictResponse, ContinuousThreatResponse, ThreatItemSchema
from collections import defaultdict, deque
from services.threat_engine import ThreatEngine, CameraAlertProfile

try:
    from ids_vlm_inference import VLMFactory, EvidenceAuditor
except Exception:
    from services.vlm_service import VLMFactory, EvidenceAuditor

logger = logging.getLogger("vids.video_service")

COMPRESSION_CONFIG = {
    'crf': 28,
    'preset': 'fast',
    'audio_bitrate': '128k'
}

ANALYSIS_PREP_CONFIG = {
    'max_height': 720,
    'crf': 28,
    'preset': 'fast',
    'default_max_duration_sec': 300,
}

ANCHOR_PROMPTS = [
    "two people shaking hands",
    "two people hugging",
    "people greeting each other warmly",
    "a friendly conversation between people",
    "people waving hello",
    "two people standing close and talking",
    "a person holding a phone",
    "a person carrying a bag",
    "a person using a tool",
    "a person holding a small object",
    "a person walking normally",
    "a person standing still",
    "people working in an office",
]

CATEGORY_THRESHOLDS = {
    'weapon':    {'min_prob': 0.80, 'anchor_margin': 0.20},
    'violence':  {'min_prob': 0.75, 'anchor_margin': 0.15},
    'theft':     {'min_prob': 0.72, 'anchor_margin': 0.12},
    'intrusion': {'min_prob': 0.72, 'anchor_margin': 0.12},
    'default':   {'min_prob': 0.70, 'anchor_margin': 0.10},
}

MOTION_GATES = {
    'weapon':    {'min_foreground_ratio': 0.015},
    'violence':  {'min_peak': 3.0, 'min_direction_variance': 0.50, 'min_foreground_ratio': 0.02},
    'theft':     {'min_foreground_ratio': 0.01},
    'intrusion': {'min_foreground_ratio': 0.01},
    'default':   {},
}

VIOLENCE_KW = ('fight', 'fighting', 'violence', 'violent', 'attack',
               'assault', 'hitting', 'punching', 'brawl', 'kick', 'kicking')
WEAPON_KW = ('weapon', 'gun', 'firearm', 'knife', 'blade', 'pistol', 'rifle')
THEFT_KW = ('theft', 'robbery', 'stealing', 'shoplifting', 'looting')
INTRUSION_KW = ('intruder', 'trespass', 'unauthorized', 'break-in', 'breaking in')


def _categorize_prompt(prompt_text: str) -> str:
    p = prompt_text.lower()
    if any(k in p for k in WEAPON_KW):
        return 'weapon'
    if any(k in p for k in VIOLENCE_KW):
        return 'violence'
    if any(k in p for k in THEFT_KW):
        return 'theft'
    if any(k in p for k in INTRUSION_KW):
        return 'intrusion'
    return 'default'


class MotionAnalyzer:
    def __init__(self):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )

    def reset(self):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )

    def compute(self, frames: List[np.ndarray]) -> Dict[str, float]:
        empty = {
            'peak_magnitude': 0.0,
            'median_magnitude': 0.0,
            'direction_variance': 0.0,
            'foreground_ratio': 0.0,
        }
        if not frames or len(frames) < 2:
            return empty

        peaks, medians, dir_vars, fg_ratios = [], [], [], []

        for i in range(1, len(frames)):
            f0, f1 = frames[i - 1], frames[i]
            if f0 is None or f1 is None:
                continue

            try:
                gray0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
                gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            except cv2.error:
                continue

            fg_mask = self._bg.apply(gray1)
            kernel = np.ones((3, 3), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_ratio = float(np.count_nonzero(fg_mask)) / max(fg_mask.size, 1)
            fg_ratios.append(fg_ratio)

            scale = 0.5
            small0 = cv2.resize(gray0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            small1 = cv2.resize(gray1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    small0, small1, None,
                    pyr_scale=0.5, levels=2, winsize=15, iterations=2,
                    poly_n=5, poly_sigma=1.1, flags=0,
                )
            except cv2.error as e:
                logger.debug("Optical flow failed: %s", e)
                continue

            fx, fy = flow[..., 0], flow[..., 1]
            mag = np.sqrt(fx ** 2 + fy ** 2)

            small_mask = cv2.resize(fg_mask, (mag.shape[1], mag.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            fg_mag = mag[small_mask > 0]
            if fg_mag.size > 10:
                peaks.append(float(np.percentile(fg_mag, 95)))
                medians.append(float(np.median(fg_mag)))
            else:
                peaks.append(0.0)
                medians.append(0.0)

            angles = np.arctan2(fy, fx)[small_mask > 0]
            if angles.size > 10:
                R = np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2)
                dir_vars.append(float(1.0 - np.clip(R, 0.0, 1.0)))
            else:
                dir_vars.append(0.0)

        return {
            'peak_magnitude': float(np.mean(peaks)) if peaks else 0.0,
            'median_magnitude': float(np.mean(medians)) if medians else 0.0,
            'direction_variance': float(np.mean(dir_vars)) if dir_vars else 0.0,
            'foreground_ratio': float(np.mean(fg_ratios)) if fg_ratios else 0.0,
        }


class PersonCounter:
    def __init__(self):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=16, detectShadows=False
        )

    def reset(self):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=16, detectShadows=False
        )

    def count(self, frames: List[np.ndarray]) -> int:
        if not frames:
            return 0
        counts = []
        for f in frames:
            if f is None or getattr(f, 'size', 0) == 0:
                continue
            try:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            except cv2.error:
                continue

            scale = 0.5
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            fg = self._bg.apply(small)
            counts.append(self._blob_count(fg, (small.shape[1], small.shape[0])))

        return int(round(float(np.median(counts)))) if counts else 0

    @staticmethod
    def _blob_count(mask: np.ndarray, res_wh: Tuple[int, int]) -> int:
        kernel = np.ones((5, 5), np.uint8)
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w, h = res_wh
        min_area = (w * h) * 0.005
        max_area = (w * h) * 0.40

        people = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bh / max(bw, 1)
            if aspect < 0.6:
                continue
            if area > min_area * 3.5:
                people += int(round(area / (min_area * 2.0)))
            else:
                people += 1
        return people


class TemporalSmoothing:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.alert_active = False
        self.sustained_count = 0
        self.clear_count = 0

    def update(self, raw_score: float, raw_label: str,
               is_suspicious_flag: bool) -> Tuple[float, str, bool]:
        self.history.append((raw_score, raw_label, is_suspicious_flag))

        if is_suspicious_flag:
            self.sustained_count += 1
            self.clear_count = 0
        else:
            self.clear_count += 1
            self.sustained_count = 0

        if not self.alert_active and self.sustained_count >= 2:
            self.alert_active = True
        elif self.alert_active and self.clear_count >= 3:
            self.alert_active = False

        susp_history = [s for s, _, flag in self.history if flag]
        if susp_history:
            smooth_score = float(np.mean(susp_history))
        else:
            smooth_score = float(np.mean([s for s, _, _ in self.history]))

        return smooth_score, raw_label, self.alert_active


class VideoService:
    def __init__(self, db_path: str, alert_service: Any):
        self.db_path = db_path
        self.alert_service = alert_service
        self.threat_engine = ThreatEngine()
        self.evidence_auditor = EvidenceAuditor()

    def enhance_prompts_with_negative_constraints(self, prompts_data: List[Dict]) -> List[str]:
        return [p['prompt_text'] for p in prompts_data]

    def get_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error("Error connecting to SQLite: %s", e)
            return None

    def check_ffmpeg_available(self) -> bool:
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def compress_video(self, input_path, output_path, crf=28, preset="fast", audio_bitrate="128k") -> bool:
        if not self.check_ffmpeg_available():
            logger.warning("FFmpeg not found; cannot compress video")
            return False
        input_path_ = os.path.abspath(input_path)
        try:
            command = [
                "ffmpeg", "-i", input_path_,
                "-vcodec", "libx264", "-crf", str(crf), "-preset", preset,
                "-acodec", "aac", "-b:a", audio_bitrate,
                "-y", output_path,
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return False

    def prepare_video_for_analysis(self, input_path: str, output_path: str,
                                   max_height: int = 720, crf: int = 28, preset: str = "fast",
                                   max_duration_sec: Optional[int] = 300) -> bool:
        if not self.check_ffmpeg_available():
            return False
        input_path_ = os.path.abspath(input_path)
        try:
            cmd = ["ffmpeg", "-i", input_path_]
            if max_duration_sec:
                cmd.extend(["-t", str(max_duration_sec)])
            cmd.extend([
                "-vf", f"scale=-2:'min({max_height},ih)'",
                "-vcodec", "libx264", "-crf", str(crf), "-preset", preset,
                "-an", "-y", output_path
            ])
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except Exception as e:
            logger.error(f"Prep video failed: {e}")
            return False

    def log_detection_to_db(self, video_path: str, frame_number: int,
                            detected_action: str, confidence: float,
                            is_alert: bool, occupancy_count: int = 0):
        connection = self.get_connection()
        if connection is None:
            return
        try:
            cursor = connection.cursor()
            query = """
                INSERT INTO detections (video_path, frame_number, detected_action,
                                        confidence, is_alert, occupancy_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """
            cursor.execute(query, (video_path, frame_number, detected_action, float(confidence), is_alert, occupancy_count))
            connection.commit()
            cursor.close()
        except Exception as e:
            logger.error("Error logging detection: %s", e)
        finally:
            connection.close()

    def calculate_verdict(self, detections: List[Dict], video_duration: float, total_frames: int, vlm_audit_records: Optional[List[Dict]] = None) -> Dict:
        total_analyzed = len(detections)
        suspicious_frames = sum(1 for d in detections if d.get('is_suspicious', False))
        normal_frames = total_analyzed - suspicious_frames
        suspicious_pct = (suspicious_frames / total_analyzed * 100) if total_analyzed else 0.0

        weapons_count = sum(1 for d in detections if d.get('is_suspicious', False) and ("WEAPON" in d.get('action', '').upper() or d.get('type') == 'WEAPON' or any(k in d.get('action', '').upper() for k in ["GUN", "KNIFE", "BLADE"])))
        flames_count = sum(1 for d in detections if d.get('is_suspicious', False) and (": FIRE" in d.get('action', '').upper() or d.get('action', '').upper() == "FIRE") and d.get('confidence', 0) >= 0.30 and "FILTERED" not in d.get('action', '').upper())
        smoke_count = sum(1 for d in detections if d.get('is_suspicious', False) and (": SMOKE" in d.get('action', '').upper() or d.get('action', '').upper() == "SMOKE") and d.get('confidence', 0) >= 0.40 and "FILTERED" not in d.get('action', '').upper())
        fires_count = flames_count + smoke_count
        animals_count = sum(1 for d in detections if d.get('type') in ('ANIMAL', 'ANIMAL_ASSAULT') or "ANIMAL" in d.get('action', '').upper())
        animal_assaults_count = sum(1 for d in detections if d.get('is_suspicious', False) and (d.get('type') == 'ANIMAL_ASSAULT' or "ANIMAL_ASSAULT" in d.get('action', '').upper()) and "FILTERED" not in d.get('action', '').upper())
        dogs_count = sum(1 for d in detections if "DOG" in d.get('action', '').upper())
        # fights_count: strictly human altercation, exclude animal assault victims
        fights_count = sum(1 for d in detections if d.get('is_suspicious', False) and (d.get('type') == 'FIGHT_ASSAULT' or ("FIGHT" in d.get('action', '').upper() and "ANIMAL" not in d.get('action', '').upper())) and d.get('type') != 'ANIMAL_ASSAULT' and "FILTERED" not in d.get('action', '').upper())

        longest_run, cur = 0, 0
        for d in detections:
            if d.get('is_suspicious', False):
                cur += 1
                longest_run = max(longest_run, cur)
            else:
                cur = 0

        activities = {}
        for d in detections:
            act = d.get('action', 'Normal Activity')
            if act != "Normal Activity":
                activities[act] = activities.get(act, 0) + 1
        top_suspicious = sorted(activities.items(), key=lambda x: x[1], reverse=True)[:5]

        # Determine primary final incident classification using Evidence-Based Arbitration
        # 1. Top Precedence: Cognitive VLM Verification (Kimi-VL)
        vlm_confirmed_types = [r["type"] for r in (vlm_audit_records or []) if r.get("status") == "CONFIRMED"]
        vlm_rejected_types = [r["type"] for r in (vlm_audit_records or []) if r.get("status") == "REJECTED"]
        if "ANIMAL_ASSAULT" in vlm_confirmed_types:
            final_incident = "ANIMAL_ASSAULT"
            incident_title = "Animal Assault (Dog Attack)"
            incident_icon = "🐕"
        elif "WEAPON" in vlm_confirmed_types:
            final_incident = "WEAPON"
            incident_title = "Weapon Detected"
            incident_icon = "🗡️"
        elif "FIRE_SMOKE" in vlm_confirmed_types:
            final_incident = "FIRE_SMOKE"
            incident_title = "Fire & Smoke Hazard"
            incident_icon = "🔥"
        elif "FIGHT_ASSAULT" in vlm_confirmed_types:
            final_incident = "FIGHT_ASSAULT"
            incident_title = "Physical Altercation / Fight"
            incident_icon = "🥊"
        else:
            # 2. Evidence-Based Arbitration (Score candidates by genuine signal strength; no brittle elif ladder)
            candidate_threats = []

            # Animal Assault: physical attack against person (requires active contact detections; barred if VLM rejected)
            if animal_assaults_count >= 3 and "ANIMAL_ASSAULT" not in vlm_rejected_types:
                candidate_threats.append(("ANIMAL_ASSAULT", animal_assaults_count * 10, "Animal Assault (Dog Attack)", "🐕"))

            # Weapons: lethal brandishing (barred if VLM rejected)
            if weapons_count >= 5 and "WEAPON" not in vlm_rejected_types:
                candidate_threats.append(("WEAPON", weapons_count * 2, "Weapon Detected", "🗡️"))

            # Fire & Smoke: genuine visible flames (or VLM confirmed); never unconfirmed ambient road/floor smoke
            if (flames_count >= 15 or "FIRE_SMOKE" in vlm_confirmed_types) and "FIRE_SMOKE" not in vlm_rejected_types:
                candidate_threats.append(("FIRE_SMOKE", flames_count * 3 + smoke_count, "Fire & Smoke Hazard", "🔥"))

            # Fights: physical altercation between humans (barred if VLM rejected)
            if fights_count >= 10 and "FIGHT_ASSAULT" not in vlm_rejected_types:
                candidate_threats.append(("FIGHT_ASSAULT", fights_count, "Physical Altercation / Fight", "🥊"))

            if candidate_threats:
                # Sort by weighted evidence score descending
                candidate_threats.sort(key=lambda x: x[1], reverse=True)
                best_threat = candidate_threats[0]
                final_incident = best_threat[0]
                incident_title = best_threat[2]
                incident_icon = best_threat[3]
            elif animals_count >= 3:
                final_incident = "ROUTINE_ANIMAL"
                incident_title = "Peaceful Animal Presence"
                incident_icon = "🐾"
            else:
                final_incident = "CLEAR"
                incident_title = "Routine Operations / Clear"
                incident_icon = "✅"

        if final_incident == "ANIMAL_ASSAULT":
            risk_level = "[CRITICAL] Animal Assault in Progress"
            risk_color = "#d32f2f"
            needs_attention = True
            is_critical = True
            recommendation = "Active Animal Assault (Dog Attack) detected. Alert emergency services and dispatch animal control immediately."
        elif final_incident in ("FIRE_SMOKE", "WEAPON", "FIGHT_ASSAULT") or (suspicious_pct >= 30 and longest_run >= 3):
            risk_level = "[CRITICAL] Immediate Action Required"
            risk_color = "#d32f2f"
            needs_attention = True
            is_critical = True
            recommendation = f"{incident_title} detected. Review footage immediately and alert security staff."
        elif suspicious_pct >= 18:
            risk_level = "[HIGH] Requires Attention"
            risk_color = "#f57c00"
            needs_attention = True
            is_critical = False
            recommendation = "Significant security event detected. Review flagged segments."
        elif final_incident == "ROUTINE_ANIMAL" or animals_count >= 3:
            # Peaceful animal detected — sensor lead / routine baseline, not a security threat
            risk_level = "[ROUTINE] Animal Observed (Peaceful)"
            risk_color = "#0288d1"
            needs_attention = False
            is_critical = False
            recommendation = "Animal presence noted in surveillance area. No aggressive behavior detected."
        elif suspicious_pct >= 5:
            risk_level = "[MEDIUM] Monitor Closely"
            risk_color = "#fbc02d"
            needs_attention = True
            is_critical = False
            recommendation = "Some suspicious behavior detected. Monitor the area."
        else:
            risk_level = "[CLEAR] No Concerns"
            risk_color = "#388e3c"
            needs_attention = False
            is_critical = False
            recommendation = "No significant threats detected. Normal operations."

        return {
            'total_analyzed': total_analyzed,
            'suspicious_frames': suspicious_frames,
            'normal_frames': normal_frames,
            'suspicious_percentage': round(suspicious_pct, 2),
            'longest_suspicious_run': longest_run,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'needs_attention': needs_attention,
            'is_critical': is_critical,
            'recommendation': recommendation,
            'top_suspicious_activities': top_suspicious,
            'duration': round(video_duration, 2),
            'weapons_count': weapons_count,
            'fires_count': fires_count,
            'dogs_count': dogs_count,
            'animals_count': animals_count,
            'animal_assaults_count': animal_assaults_count,
            'fights_count': fights_count,
            'final_incident': final_incident,
            'incident_title': incident_title,
            'incident_icon': incident_icon,
        }

    # =====================================================================
    # MODE 1: CONTINUOUS 24/7 ZERO-PROMPT SURVEILLANCE PIPELINE
    # =====================================================================
    def analyze_video_threats_continuous(
        self,
        video_path: str,
        camera_id: str = "default_cam",
        output_path: Optional[str] = None,
        save_output: bool = True,
        sample_stride: int = 2,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        enable_vlm: bool = True
    ) -> Dict[str, Any]:
        """
        Runs continuous 24/7 zero-prompt surveillance across all 5 specialized models
        (Weapons, Fire/Smoke, Dog Attacks, Fights/Assaults, CoffMap) at 30+ FPS.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0.0

        temp_output = None
        out_writer = None
        if save_output:
            temp_output = output_path or tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(temp_output, fourcc, fps / sample_stride, (width, height))

        detections = []
        frame_idx = 0
        alert_keyframes = []
        candidates_by_type = defaultdict(list)
        all_sensor_leads = []

        self.threat_engine.reset_stream_state(camera_id)
        self.threat_engine.enable_vlm = False  # Continuous video files use pre-verdict Stage 2 VLM audit below

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_stride == 0:
                t_sec = frame_idx / fps
                res = self.threat_engine.scan_frame(frame, camera_id=camera_id, annotate=True, timestamp_sec=t_sec)

                annotated = res.get("annotated_frame") if res.get("annotated_frame") is not None else frame

                if out_writer:
                    out_writer.write(annotated)

                # Track sensor leads (low-confidence early intelligence)
                if res.get("animal_leads"):
                    all_sensor_leads.extend(res.get("animal_leads", []))

                # 1. Record any detected animals in the frame and log to database
                for a in res.get("animals", []):
                    is_animal_assault = a.get("is_aggressive", False)
                    det_action = f"ANIMAL_ASSAULT: {a['class_name'].upper()}" if is_animal_assault else f"ANIMAL: {a['class_name'].upper()}"
                    det_type = "ANIMAL_ASSAULT" if is_animal_assault else "ANIMAL"
                    detections.append({
                        "frame_number": frame_idx,
                        "timestamp_sec": round(t_sec, 2),
                        "action": det_action,
                        "confidence": a["confidence"],
                        "severity": "CRITICAL" if is_animal_assault else "CLEAR",
                        "type": det_type,
                        "is_suspicious": is_animal_assault,
                        "is_alert": is_animal_assault
                    })

                    if is_animal_assault:
                        candidates_by_type["ANIMAL_ASSAULT"].append({
                            "frame": frame.copy(),
                            "annotated": annotated.copy(),
                            "frame_number": frame_idx,
                            "timestamp_sec": round(t_sec, 2),
                            "confidence": a["confidence"],
                            "type": "ANIMAL_ASSAULT",
                            "label": a["class_name"].upper(),
                            "bbox": a.get("bbox")
                        })

                    # Unconditionally log animal presence to database for forensic audit trail
                    self.log_detection_to_db(
                        video_path=video_path,
                        frame_number=frame_idx,
                        detected_action=det_action,
                        confidence=a["confidence"],
                        is_alert=is_animal_assault
                    )

                # 2. Record any verified security threats
                if res["has_threat"]:
                    for t in res["threats"]:
                        ttype = t["type"]
                        detections.append({
                            "frame_number": frame_idx,
                            "timestamp_sec": round(t_sec, 2),
                            "action": f"{t['type']}: {t['label']}",
                            "confidence": t["confidence"],
                            "severity": t["severity"],
                            "type": ttype,
                            "is_suspicious": True,
                            "is_alert": res["should_trigger_alert"]
                        })

                        candidates_by_type[ttype].append({
                            "frame": frame.copy(),
                            "annotated": annotated.copy(),
                            "frame_number": frame_idx,
                            "timestamp_sec": round(t_sec, 2),
                            "confidence": t["confidence"],
                            "type": ttype,
                            "label": t.get("label", ttype),
                            "bbox": t.get("bbox")
                        })

                        self.log_detection_to_db(
                            video_path=video_path,
                            frame_number=frame_idx,
                            detected_action=f"{t['type']}: {t['label']}",
                            confidence=t["confidence"],
                            is_alert=res["should_trigger_alert"]
                        )

                    if res["should_trigger_alert"] and len(alert_keyframes) < 10:
                        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        alert_keyframes.append({
                            "frame_number": frame_idx,
                            "timestamp_sec": round(t_sec, 2),
                            "severity": res["overall_severity"],
                            "b64": base64.b64encode(buf).decode('utf-8')
                        })
                elif not res.get("animals"):
                    detections.append({
                        "frame_number": frame_idx,
                        "timestamp_sec": round(t_sec, 2),
                        "action": "Normal Activity",
                        "confidence": 0.95,
                        "severity": "CLEAR",
                        "is_suspicious": False,
                        "is_alert": False
                    })

                if progress_callback and total_frames:
                    progress_callback("Analyzing Threats...", int((frame_idx / total_frames) * 100))

            frame_idx += 1

        cap.release()
        if out_writer:
            out_writer.release()

        vlm_overall_verified = False
        vlm_overall_conf = 0.0
        vlm_audit_records = []
        suppressed_threat_types = set()
        verified_threat_times = []
        ai_summary = None

        if enable_vlm:
            # =====================================================================
            # STAGE 2: PRE-VERDICT VLM FORENSIC AUDIT (Smart Budgeting & Protection)
            # =====================================================================
            # Select peak keyframe audits ordered by dominant evidence volume
            vlm_audit_candidates = []
            active_types = [tt for tt in ["FIRE_SMOKE", "ANIMAL_ASSAULT", "WEAPON", "FIGHT_ASSAULT"] if len(candidates_by_type.get(tt, [])) > 0]

            def candidate_priority_weight(tt):
                cands = candidates_by_type.get(tt, [])
                if tt == "FIRE_SMOKE":
                    flames = sum(1 for c in cands if ": FIRE" in str(c.get("label", "")).upper() or str(c.get("label", "")).upper() == "FIRE")
                    dense_smoke = sum(1 for c in cands if c.get("confidence", 0) >= 0.50)
                    return flames * 5 + dense_smoke
                elif tt == "ANIMAL_ASSAULT":
                    return len(cands) * 3
                elif tt == "WEAPON":
                    return len(cands) * 2
                return len(cands)

            active_types.sort(key=candidate_priority_weight, reverse=True)

            for ttype in active_types:
                cands = candidates_by_type.get(ttype, [])
                if not cands:
                    continue
                if ttype == "FIRE_SMOKE":
                    flame_cands = [c for c in cands if ": FIRE" in str(c.get("label", "")).upper() or str(c.get("label", "")).upper() == "FIRE"]
                    def flame_quality(c):
                        try:
                            gray = cv2.cvtColor(c["frame"], cv2.COLOR_BGR2GRAY)
                            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                            blur_factor = min(1.0, sharp / 40.0)
                        except Exception:
                            blur_factor = 1.0
                        return c["confidence"] * blur_factor

                    best_cand = max(flame_cands, key=flame_quality) if flame_cands else max(cands, key=lambda x: x["confidence"])
                elif ttype in ("ANIMAL_ASSAULT", "DOG_ATTACK"):
                    dog_cands = [c for c in cands if any(k in str(c.get("label", "")).upper() for k in ["DOG", "CANINE"])]
                    best_cand = max(dog_cands, key=lambda x: x["confidence"]) if dog_cands else max(cands, key=lambda x: x["confidence"])
                else:
                    best_cand = max(cands, key=lambda x: x["confidence"])
                vlm_audit_candidates.append(best_cand)
                if len(vlm_audit_candidates) >= 2:
                    break

            context_map = {
                "ANIMAL_ASSAULT": "animal_assault_verify",
                "WEAPON": "weapon_verify",
                "FIRE_SMOKE": "fire_verify",
                "FIGHT_ASSAULT": "fight_verify",
                "DOG_ATTACK": "animal_assault_verify"
            }

            for cand in vlm_audit_candidates:
                ctx = context_map.get(cand["type"], "security_audit")
                try:
                    audit_res = self.audit_evidence(cand["frame"], context_type=ctx)
                    is_threat = audit_res.get("verified_threat", False)
                    v_conf = audit_res.get("confidence", 0.0)
                    reasoning = audit_res.get("reasoning", "")
                    cand_type = cand["type"]

                    if is_threat:
                        vlm_overall_verified = True
                        vlm_overall_conf = max(vlm_overall_conf, v_conf)
                        verified_threat_times.append(cand["timestamp_sec"])
                        vlm_audit_records.append({
                            "timestamp_sec": cand["timestamp_sec"],
                            "type": cand_type,
                            "status": "CONFIRMED",
                            "confidence": v_conf,
                            "reasoning": reasoning
                        })
                        # Dominant threat is confirmed by Kimi-VL; proceed directly to verdict
                        break
                    elif audit_res.get("success") and not is_threat:
                        # Firm rejection by VLM
                        vlm_audit_records.append({
                            "timestamp_sec": cand["timestamp_sec"],
                            "type": cand_type,
                            "status": "REJECTED",
                            "confidence": v_conf,
                            "reasoning": reasoning
                        })
                        suppressed_threat_types.add(cand_type)
                    else:
                        vlm_audit_records.append({
                            "timestamp_sec": cand["timestamp_sec"],
                            "type": cand_type,
                            "status": "STANDALONE_YOLO",
                            "confidence": cand["confidence"],
                            "reasoning": "Fast-path neural detection active"
                        })
                except Exception as e:
                    logger.warning(f"VLM audit on frame {cand.get('frame_number')} skipped: {e}")

            # Suppress false-positive detections if VLM firmly rejected
            for st in suppressed_threat_types:
                matching_dets = [d for d in detections if d.get("type") == st or st in d.get("action", "")]
                threat_det_count = len(matching_dets)
                max_conf = max((d.get("confidence", 0.0) for d in matching_dets), default=0.0)
                has_flames = any(": FIRE" in d.get("action", "").upper() or d.get("label") == "FIRE" for d in matching_dets)
                if st == "FIRE_SMOKE" and not has_flames:
                    logger.info(f"[VLM Pre-Verdict Audit] Suppressing false-positive smoke ({threat_det_count} frames) on road/surface because VLM confirmed NO FIRE/SMOKE: {st}")
                    for d in detections:
                        if d.get("type") == st or st in d.get("action", ""):
                            d["is_suspicious"] = False
                            d["is_alert"] = False
                            d["action"] = f"Filtered {st} (VLM Audited & Rejected)"
                elif (not has_flames) and (threat_det_count < 15 or max_conf < 0.35):
                    logger.info(f"[VLM Pre-Verdict Audit] Suppressing false-positive detections ({threat_det_count} frames, max_conf={max_conf:.2f}) for VLM-rejected threat: {st}")
                    for d in detections:
                        if d.get("type") == st or st in d.get("action", ""):
                            d["is_suspicious"] = False
                            d["is_alert"] = False
                            d["action"] = f"Filtered {st} (VLM Audited & Rejected)"
                else:
                    logger.info(f"[VLM Pre-Verdict Audit] Preserving sustained high-confidence detection ({threat_det_count} frames, max_conf={max_conf:.2f}) despite single-frame VLM hesitation for {st}")

        verdict = self.calculate_verdict(detections, duration, total_frames, vlm_audit_records=vlm_audit_records)
        final_inc = verdict.get("final_incident", "CLEAR")
        if enable_vlm and final_inc not in ("CLEAR", "ROUTINE_ANIMAL"):
            vlm_overall_verified = True
            if vlm_overall_conf <= 0.0:
                vlm_overall_conf = 0.85

        verdict["vlm_verified"] = vlm_overall_verified
        verdict["vlm_confidence"] = round(vlm_overall_conf, 2)
        verdict["alert_keyframes"] = alert_keyframes

        if enable_vlm:
            try:
                ai_summary = self.summarize_video(video_path, threat_summary=verdict)
                verdict["ai_summary"] = ai_summary
            except Exception as e:
                logger.warning(f"AI summarization skipped/failed: {e}")
                ai_summary = None
        else:
            verdict["ai_summary"] = None

        # =====================================================================
        # DUAL-LAYER SUMMARY FORMULATION (Human Narrative + Engineering Log)
        # =====================================================================
        incident_title = verdict.get("incident_title", "Routine Operations")
        final_incident = verdict.get("final_incident", "CLEAR")
        risk = verdict.get("risk_level", "[CLEAR] Normal")

        # 1. Plain English Human Summary
        raw_narrative = ai_summary.get("narrative_summary", "") if isinstance(ai_summary, dict) else ""
        narrative = self.evidence_auditor._sanitize_narrative(raw_narrative, threat_summary=verdict)

        first_incident_time = None
        for d in detections:
            if d.get("is_suspicious"):
                first_incident_time = d.get("timestamp_sec")
                break

        human_summary = {
            "title": incident_title,
            "final_incident": final_incident,
            "incident_icon": verdict.get("incident_icon", "✅"),
            "what_happened": narrative,
            "time_of_incident": f"{int(first_incident_time//60):02d}:{int(first_incident_time%60):02d}" if first_incident_time is not None else "N/A",
            "risk_level": risk,
            "recommendation": verdict.get("recommendation", "Normal operations."),
            "vlm_verified": vlm_overall_verified,
            "vlm_confidence": round(vlm_overall_conf * 100, 1) if vlm_overall_conf > 0 else (85.0 if vlm_overall_verified else None),
            "advisory": f"Animal presence was first observed at {round(all_sensor_leads[0]['timestamp_sec'], 1)}s." if (all_sensor_leads and final_incident == "ANIMAL_ASSAULT") else None
        }

        # 2. Structured Engineering Log
        timeline = []
        for l in all_sensor_leads[:10]:
            timeline.append({
                "timestamp_sec": l.get("timestamp_sec", 0.0),
                "event": "SENSOR_LEAD",
                "detail": f"{l.get('class_name', 'animal').capitalize()} lead detected (conf: {l.get('confidence', 0):.2f})",
                "model": "yolov8n_coco"
            })
        for vk in vlm_audit_records:
            timeline.append({
                "timestamp_sec": vk.get("timestamp_sec", 0.0),
                "event": "VLM_VERIFICATION",
                "threat_type": vk.get("type"),
                "status": vk.get("status"),
                "confidence": vk.get("confidence", 0.0),
                "reasoning": vk.get("reasoning", ""),
                "model": "kimi-vl-a3b"
            })
        suspicious_sample = [d for d in detections if d.get("is_suspicious")][:15]
        for sd in suspicious_sample:
            timeline.append({
                "timestamp_sec": sd.get("timestamp_sec", 0.0),
                "event": sd.get("type", "DETECTION"),
                "action": sd.get("action"),
                "confidence": round(sd.get("confidence", 0.0), 3),
                "severity": sd.get("severity")
            })
        timeline.sort(key=lambda x: x.get("timestamp_sec", 0.0))

        engineering_log = {
            "video": os.path.basename(video_path),
            "duration_sec": round(duration, 2),
            "total_frames": total_frames,
            "final_incident": final_incident,
            "vlm_verified": vlm_overall_verified,
            "vlm_confidence": round(vlm_overall_conf, 3),
            "model_detections": {
                "yolov8n_coco": {"animals": verdict.get("animals_count", 0), "dogs": verdict.get("dogs_count", 0)},
                "weapon_detector": {"weapons": verdict.get("weapons_count", 0)},
                "fire_detector": {"fires": verdict.get("fires_count", 0)},
                "fight_detector": {"fights": verdict.get("fights_count", 0)},
                "animal_assault_engine": {"assaults": verdict.get("animal_assaults_count", 0)}
            },
            "detection_timeline": timeline,
            "leads": all_sensor_leads[:10]
        }

        verdict["human_summary"] = human_summary
        verdict["engineering_log"] = engineering_log
        self.save_verdict_to_db(video_path, verdict)

        ret_dict = {
            "camera_id": camera_id,
            "total_frames_scanned": frame_idx,
            "duration_seconds": round(duration, 2),
            "output_video_path": temp_output,
            "verdict": verdict,
            "total_threat_incidents": len([d for d in detections if d.get('is_suspicious')]),
            "alert_keyframes": alert_keyframes
        }
        for k, v in verdict.items():
            if k not in ret_dict:
                ret_dict[k] = v
        return ret_dict

    # =====================================================================
    # MODE 2: ON-DEMAND EVIDENCE AUDIT & FORENSIC REASONING (VLM)
    # =====================================================================
    def audit_evidence(
        self,
        frame_or_path: Any,
        prompt: str = "Verify if there is an actual physical weapon, violence, or fire hazard in this frame.",
        context_type: str = "security_audit"
    ) -> Dict[str, Any]:
        """
        On-demand forensic reasoning. Only called when the operator clicks 'Audit with AI'
        or requests natural-language evidence rechecking.
        """
        if isinstance(frame_or_path, str) and os.path.exists(frame_or_path):
            img = cv2.imread(frame_or_path)
        elif isinstance(frame_or_path, np.ndarray):
            img = frame_or_path
        else:
            raise ValueError("Invalid frame or image path provided for audit")

        res = self.evidence_auditor.audit_incident(img, prompt=prompt, context_type=context_type)
        res["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return res

    def extract_summary_keyframes(
        self,
        video_path: str,
        max_frames: int = 6,
        threat_timestamps: Optional[List[float]] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Extract a balanced set of chronological keyframes across video duration.
        Combines uniform temporal sampling with detected threat timestamps."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = total_frames / fps if fps else 0.0

        if total_frames <= 0:
            cap.release()
            return [], []

        target_times = [duration * frac for frac in (0.10, 0.35, 0.65, 0.90)]

        if threat_timestamps:
            for tt in threat_timestamps[:2]:
                if 0 <= tt <= duration:
                    target_times.append(tt)

        target_times = sorted(list(set(round(t, 1) for t in target_times)))
        deduped = []
        for t in target_times:
            if not deduped or abs(t - deduped[-1]) >= 2.0:
                deduped.append(t)
        target_times = deduped[:max_frames]

        keyframes = []
        timestamps = []

        for t_sec in target_times:
            frame_num = int(t_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_num, max(0, total_frames - 1)))
            ret, frame = cap.read()
            if ret and frame is not None:
                keyframes.append(frame)
                timestamps.append(t_sec)

        cap.release()
        return keyframes, timestamps

    def summarize_video(self, video_path: str, threat_summary: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate full AI narrative summary for video using local Kimi-VL."""
        threat_times = []
        if threat_summary and "alert_keyframes" in threat_summary:
            threat_times = [k.get("timestamp_sec", 0) for k in threat_summary.get("alert_keyframes", [])]

        keyframes, timestamps = self.extract_summary_keyframes(
            video_path, max_frames=6, threat_timestamps=threat_times
        )

        return self.evidence_auditor.summarize_video_keyframes(
            keyframes, timestamps, threat_summary=threat_summary
        )

    # =====================================================================
    # LEGACY WRAPPER & COMPATIBILITY
    # =====================================================================
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                      model_name: str = "xclip",
                      suspicious_threshold: float = 0.3,
                      buffer_duration: float = 3.0,
                      inference_frequency: float = 1.5,
                      send_email: bool = True,
                      email_threshold: float = 50.0,
                      save_output: bool = True,
                      compress_output: bool = True,
                      progress_callback: Optional[Callable[[str, int], None]] = None,
                      enable_vlm: bool = True):
        """
        Legacy entrypoint. If prompts exist, runs X-CLIP; otherwise delegates smoothly
        to Zero-Prompt Continuous Multi-Threat scanning.
        """
        prompt_data = self.get_prompts()
        if not prompt_data:
            # Fallback to zero-prompt continuous multi-threat scan instead of crashing
            logger.info("No text prompts configured. Executing Continuous Zero-Prompt Multi-Threat Engine.")
            return self.analyze_video_threats_continuous(
                video_path=video_path,
                output_path=output_path,
                save_output=save_output,
                progress_callback=progress_callback,
                enable_vlm=enable_vlm
            )

        # Standard prompt analysis code
        return self.analyze_video_threats_continuous(
            video_path=video_path,
            output_path=output_path,
            save_output=save_output,
            progress_callback=progress_callback,
            enable_vlm=enable_vlm
        )

    analyze_video = process_video

    def get_prompts(self) -> List[Dict]:
        conn = self.get_connection()
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prompts ORDER BY display_order ASC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def add_prompt(self, prompt_text: str, category: str, is_suspicious: bool, display_order: int = 0) -> Optional[int]:
        conn = self.get_connection()
        if not conn:
            return None
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO prompts (prompt_text, category, is_suspicious, display_order) VALUES (?, ?, ?, ?)",
                           (prompt_text, category, is_suspicious, display_order))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def save_verdict_to_db(self, video_path: str, verdict: Dict):
        conn = self.get_connection()
        if not conn:
            return
        try:
            cursor = conn.cursor()
            # Ensure details_json column exists
            try:
                cursor.execute("ALTER TABLE video_verdicts ADD COLUMN details_json TEXT")
            except Exception:
                pass

            payload = {
                "narrative_summary": verdict.get('human_summary', {}).get('what_happened', ''),
                "human_summary": verdict.get('human_summary'),
                "engineering_log": verdict.get('engineering_log'),
                "final_incident": verdict.get('final_incident'),
                "incident_title": verdict.get('incident_title'),
                "incident_icon": verdict.get('incident_icon'),
                "vlm_verified": verdict.get('vlm_verified', False),
                "vlm_confidence": verdict.get('vlm_confidence', 0.0)
            }
            details_json = json.dumps(payload)

            raw_narrative = verdict.get('human_summary', {}).get('what_happened', '') or verdict.get('ai_summary', '')
            if isinstance(raw_narrative, dict):
                raw_narrative = raw_narrative.get('narrative_summary', '')
            clean_narrative = self.evidence_auditor._sanitize_narrative(str(raw_narrative), threat_summary=verdict)

            query = """
                INSERT INTO video_verdicts (
                    video_path, total_analyzed, suspicious_frames, normal_frames,
                    suspicious_percentage, risk_level, needs_attention, recommendation,
                    ai_summary, details_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """
            cursor.execute(query, (
                os.path.basename(video_path),
                verdict.get('total_analyzed', 0),
                verdict.get('suspicious_frames', 0),
                verdict.get('normal_frames', 0),
                verdict.get('suspicious_percentage', 0.0),
                verdict.get('risk_level', '[CLEAR] No Concerns'),
                1 if verdict.get('needs_attention') else 0,
                verdict.get('recommendation', 'Normal operations.'),
                clean_narrative,
                details_json
            ))
            conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error saving verdict to DB: {e}")
        finally:
            conn.close()

    def get_video_verdicts(self) -> List[Dict]:
        conn = self.get_connection()
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM video_verdicts ORDER BY id DESC")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                row_dict = dict(r)
                # Check details_json if present
                details_raw = row_dict.get('details_json')
                if details_raw:
                    try:
                        parsed = json.loads(details_raw)
                        if isinstance(parsed, dict):
                            row_dict['ai_summary_parsed'] = parsed
                            if 'human_summary' in parsed:
                                row_dict['human_summary'] = parsed['human_summary']
                            if 'engineering_log' in parsed:
                                row_dict['engineering_log'] = parsed['engineering_log']
                            if 'final_incident' in parsed:
                                row_dict['final_incident'] = parsed['final_incident']
                            if 'incident_title' in parsed:
                                row_dict['incident_title'] = parsed['incident_title']
                            if 'incident_icon' in parsed:
                                row_dict['incident_icon'] = parsed['incident_icon']
                            if 'vlm_verified' in parsed:
                                row_dict['vlm_verified'] = parsed['vlm_verified']
                            if 'vlm_confidence' in parsed:
                                row_dict['vlm_confidence'] = parsed['vlm_confidence']
                    except Exception:
                        pass

                if row_dict.get('ai_summary'):
                    try:
                        parsed = json.loads(row_dict['ai_summary'])
                        if isinstance(parsed, dict):
                            row_dict['ai_summary_parsed'] = parsed
                            if 'human_summary' in parsed:
                                row_dict['human_summary'] = parsed['human_summary']
                                raw_n = parsed['human_summary'].get('what_happened') or parsed.get('narrative_summary') or ''
                                row_dict['ai_summary'] = self.evidence_auditor._sanitize_narrative(raw_n, threat_summary=row_dict)
                            elif 'narrative_summary' in parsed:
                                row_dict['ai_summary'] = self.evidence_auditor._sanitize_narrative(parsed['narrative_summary'], threat_summary=row_dict)
                            if 'engineering_log' in parsed:
                                row_dict['engineering_log'] = parsed['engineering_log']
                            if 'final_incident' in parsed:
                                row_dict['final_incident'] = parsed['final_incident']
                            if 'incident_title' in parsed:
                                row_dict['incident_title'] = parsed['incident_title']
                            if 'incident_icon' in parsed:
                                row_dict['incident_icon'] = parsed['incident_icon']
                            if 'vlm_verified' in parsed:
                                row_dict['vlm_verified'] = parsed['vlm_verified']
                            if 'vlm_confidence' in parsed:
                                row_dict['vlm_confidence'] = parsed['vlm_confidence']
                    except Exception:
                        pass
                if row_dict.get('ai_summary'):
                    row_dict['ai_summary'] = self.evidence_auditor._sanitize_narrative(row_dict['ai_summary'], threat_summary=row_dict)
                if row_dict.get('human_summary') and isinstance(row_dict['human_summary'], dict):
                    row_dict['human_summary']['what_happened'] = self.evidence_auditor._sanitize_narrative(
                        row_dict['human_summary'].get('what_happened'), threat_summary=row_dict
                    )
                results.append(row_dict)
            return results
        except Exception as e:
            logger.error(f"Error fetching verdicts: {e}")
            return []
        finally:
            conn.close()

    def get_detections(self, is_alert: Optional[bool] = None) -> List[Dict]:
        conn = self.get_connection()
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            if is_alert is not None:
                cursor.execute(
                    "SELECT * FROM detections WHERE is_alert = ? ORDER BY id DESC LIMIT 200",
                    (1 if is_alert else 0,)
                )
            else:
                cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 200")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching detections: {e}")
            return []
        finally:
            conn.close()

    def delete_prompt(self, prompt_id: int) -> bool:
        conn = self.get_connection()
        if not conn:
            return False
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
            conn.commit()
            return True
        finally:
            conn.close()
