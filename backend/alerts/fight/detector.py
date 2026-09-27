"""
alerts/fight/detector.py — Fight/Assault Detection (Proximity + Motion Gated).

Replaces the previous .pkl Random Forest classifier approach with the
proximity-gating architecture from the Fight workspace:

  1. YOLOv8-Pose extracts person skeletons and bounding boxes
  2. Proximity gate: are two people within arm's reach?
  3. Motion analysis: are arms moving rapidly? (wrist velocity over time)
  4. Posture check: are arms raised aggressively above shoulders?

No .pkl files required. Works out of the box with just yolov8n-pose.pt.

Severity levels:
  NORMAL           -> No concerning activity
  AGGRESSIVE_STANCE -> Close proximity with moderate motion or raised arms
  ACTIVE_FIGHT     -> Close proximity with rapid arm movement (punching)
"""
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class FightDetector:
    SEVERITY_LEVELS = {
        "NORMAL": {"level": 0, "color": (0, 200, 0), "label": "NORMAL ACTIVITY"},
        "AGGRESSIVE_STANCE": {"level": 1, "color": (0, 140, 255), "label": "AGGRESSIVE STANCE DETECTED"},
        "ACTIVE_FIGHT": {"level": 2, "color": (0, 0, 255), "label": "FIGHT IN PROGRESS"},
    }

    # COCO 17-keypoint indices
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    # --- Tunable thresholds ---------------------------------------------------

    # Proximity: two persons whose center-to-center pixel distance is less than
    # this fraction of the average bounding-box height are "within arm's reach".
    PROXIMITY_RATIO = 0.95

    # Normalized wrist velocity per frame (keypoints are 0-1 normalized).
    # Calibrated for real physical altercations (punching, tackling, brawling).
    WRIST_VEL_AGGRESSIVE = 0.025   # moderate grappling / aggressive posture
    WRIST_VEL_FIGHTING = 0.040     # rapid punching / physical combat

    # Minimum frames of keypoint history needed before motion can be computed.
    MIN_HISTORY_FRAMES = 3

    # Maximum keypoint buffer length per person (prevents unbounded memory).
    MAX_BUFFER_LEN = 30

    def __init__(
        self,
        pose_model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",
        **kwargs,  # Accept and ignore classifier_path/scaler_path for backward compat
    ):
        # Resolve pose model path from models/ directory
        if not Path(pose_model_path).exists():
            candidates = []
            if getattr(sys, "frozen", False):
                candidates.append(Path(sys.executable).parent / "models" / Path(pose_model_path).name)
                if hasattr(sys, "_MEIPASS"):
                    candidates.append(Path(sys._MEIPASS) / "models" / Path(pose_model_path).name)
            base1 = Path(__file__).resolve().parent.parent.parent / "models"
            base2 = Path(__file__).resolve().parent.parent / "models"
            candidates.extend([base1 / Path(pose_model_path).name, base2 / Path(pose_model_path).name, Path.cwd() / "models" / Path(pose_model_path).name])
            for candidate in candidates:
                if candidate.exists():
                    pose_model_path = str(candidate)
                    break

        # Load pose model
        if YOLO is not None:
            self.pose_model = YOLO(pose_model_path)
        else:
            self.pose_model = None
            print("[Fight] WARNING: ultralytics not available.")

        self.device = device
        # Per-person keypoint history: track_id → list of (17, 2) normalized arrays
        self.keypoint_buffers: Dict[int, List[np.ndarray]] = {}
        # Spatial IoU tracking to eliminate person ID-swapping artifacts
        self._prev_tracks: List[Dict] = []
        self._next_track_id: int = 0

    def reset(self):
        """Reset temporal state between video streams."""
        self.keypoint_buffers.clear()
        self._prev_tracks.clear()
        self._next_track_id = 0

    def reset_buffers(self):
        """Compatibility alias for ThreatEngine."""
        self.reset()

    def _assign_tracks(self, bboxes: List[Tuple[float, float, float, float]]) -> List[int]:
        """Assign persistent track IDs based on spatial bounding box IoU."""
        if not bboxes:
            self._prev_tracks = []
            return []

        assigned_ids = []
        used_prev = set()

        for b in bboxes:
            best_iou = 0.0
            best_idx = -1
            for idx, prev in enumerate(self._prev_tracks):
                if idx in used_prev:
                    continue
                pb = prev["bbox"]
                ix1, iy1 = max(b[0], pb[0]), max(b[1], pb[1])
                ix2, iy2 = min(b[2], pb[2]), min(b[3], pb[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
                union = area_b + area_p - inter
                iou = inter / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= 0.20 and best_idx >= 0:
                tid = self._prev_tracks[best_idx]["track_id"]
                used_prev.add(best_idx)
                assigned_ids.append(tid)
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                assigned_ids.append(tid)

        self._prev_tracks = [{"track_id": tid, "bbox": bboxes[i]} for i, tid in enumerate(assigned_ids)]
        return assigned_ids

    def detect(self, image: np.ndarray, annotate: bool = True) -> Dict:
        """
        Run fight detection on a single frame.

        Pipeline:
          1. YOLOv8-Pose → person bounding boxes + 17 keypoints
          2. Spatial IoU tracking to eliminate frame-to-frame index swapping
          3. Buffer keypoints per person for temporal motion analysis
          4. Compute per-person arm (wrist) velocity
          5. Check all person-pairs for proximity
          6. Combine proximity + motion + posture → severity
        """
        if self.pose_model is None:
            return {
                "persons_detected": 0,
                "severity": "NORMAL",
                "severity_info": self.SEVERITY_LEVELS["NORMAL"],
                "predictions": [],
                "model_loaded": False,
            }

        h, w = image.shape[:2]
        t0 = time.time()

        # --- Step 1: Pose estimation ---
        results = self.pose_model.predict(
            image, device=self.device, verbose=False, conf=0.3
        )
        inference_ms = round((time.time() - t0) * 1000, 2)

        persons = []

        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kps_data = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = results[0].boxes

            raw_bboxes = []
            valid_indices = []
            for i, kps in enumerate(kps_data):
                if kps is None or len(kps) < 17:
                    continue
                confs = kps[:, 2]
                if boxes is not None and i < len(boxes):
                    bbox = tuple(boxes[i].xyxy[0].tolist())
                else:
                    valid_mask = confs > 0.3
                    if valid_mask.any():
                        bbox = (
                            float(kps[valid_mask, 0].min()),
                            float(kps[valid_mask, 1].min()),
                            float(kps[valid_mask, 0].max()),
                            float(kps[valid_mask, 1].max()),
                        )
                    else:
                        continue
                raw_bboxes.append(bbox)
                valid_indices.append(i)

            # Spatial tracking: match to previous frame by IoU to avoid jumping indices
            track_ids = self._assign_tracks(raw_bboxes)

            for idx, i in enumerate(valid_indices):
                kps = kps_data[i]
                bbox = raw_bboxes[idx]
                track_id = track_ids[idx]

                xy = kps[:, :2]       # (17, 2) pixel coords
                confs = kps[:, 2]     # (17,) per-keypoint confidence

                # Normalize to 0-1 for frame-size-independent velocity
                xy_norm = xy.copy()
                xy_norm[:, 0] /= max(w, 1)
                xy_norm[:, 1] /= max(h, 1)

                # --- Step 2: Buffer keypoints ---
                if track_id not in self.keypoint_buffers:
                    self.keypoint_buffers[track_id] = []
                self.keypoint_buffers[track_id].append(xy_norm)
                if len(self.keypoint_buffers[track_id]) > self.MAX_BUFFER_LEN:
                    self.keypoint_buffers[track_id] = \
                        self.keypoint_buffers[track_id][-self.MAX_BUFFER_LEN:]

                # --- Step 3: Per-person motion + posture ---
                motion_score = self._compute_wrist_velocity(track_id)
                raised_arms = self._check_raised_arms(xy_norm, confs)

                persons.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "keypoints": xy.tolist(),
                    "avg_confidence": round(float(np.mean(confs)), 3),
                    "motion_score": motion_score,
                    "raised_arms": raised_arms,
                })

        # --- Step 4: Proximity check between all pairs ---
        close_pairs = self._find_close_pairs(persons)

        # --- Step 5: Combine proximity + motion → severity & predictions ---
        predictions = []
        max_severity = "NORMAL"
        seen_pairs = set()

        for (idx_a, idx_b, pixel_dist, threshold) in close_pairs:
            pa, pb = persons[idx_a], persons[idx_b]
            max_motion = max(pa["motion_score"], pb["motion_score"])
            either_raised = pa["raised_arms"] or pb["raised_arms"]

            if max_motion >= self.WRIST_VEL_FIGHTING:
                # High arm velocity + close proximity = active fight
                pred_label = "fighting"
                pred_id = 2
                confidence = min(0.95, 0.70 + max_motion * 5)
            elif max_motion >= self.WRIST_VEL_AGGRESSIVE or (either_raised and max_motion >= 0.038):
                # Moderate motion or raised fists with movement + close proximity = aggressive
                pred_label = "aggressive_stance"
                pred_id = 1
                confidence = min(0.85, 0.50 + max_motion * 5)
            else:
                # Close together but no significant motion — just standing near
                continue

            # Update max severity
            if pred_label == "fighting":
                max_severity = "ACTIVE_FIGHT"
            elif pred_label == "aggressive_stance" and max_severity != "ACTIVE_FIGHT":
                max_severity = "AGGRESSIVE_STANCE"

            # Emit one prediction per involved person (avoid duplicates)
            for person in (pa, pb):
                tid = person["track_id"]
                if tid in seen_pairs:
                    continue
                seen_pairs.add(tid)

                predictions.append({
                    "track_id": tid,
                    "prediction": pred_label,
                    "prediction_id": pred_id,
                    "confidence": round(confidence, 3),
                    "proximity_px": round(pixel_dist, 1),
                    "threshold_px": round(threshold, 1),
                    "motion_score": round(max_motion, 4),
                })

        # Mark combatant status on persons
        combatant_tids = {p["track_id"] for p in predictions if p.get("prediction") in ["fighting", "aggressive_stance"]}
        for p in persons:
            p["is_combatant"] = p["track_id"] in combatant_tids

        result = {
            "persons_detected": len(persons),
            "persons": persons,
            "severity": max_severity,
            "severity_info": self.SEVERITY_LEVELS[max_severity],
            "predictions": predictions,
            "inference_ms": inference_ms,
            "model_loaded": True,
        }

        if annotate:
            result["annotated_image"] = self._annotate(
                image, results, predictions, max_severity
            )

        return result

    # ------------------------------------------------------------------
    #  Internal analysis helpers
    # ------------------------------------------------------------------

    def _compute_wrist_velocity(self, track_id: int) -> float:
        """Average wrist velocity over recent frames (normalized 0-1 coords).

        Uses the maximum of left/right wrist displacement per frame pair,
        averaged over the last MIN_HISTORY_FRAMES frames.
        """
        buf = self.keypoint_buffers.get(track_id, [])
        if len(buf) < self.MIN_HISTORY_FRAMES:
            return 0.0

        recent = buf[-self.MIN_HISTORY_FRAMES:]
        velocities = []
        for i in range(1, len(recent)):
            prev, curr = recent[i - 1], recent[i]
            lw_vel = float(np.linalg.norm(curr[self.L_WRIST] - prev[self.L_WRIST]))
            rw_vel = float(np.linalg.norm(curr[self.R_WRIST] - prev[self.R_WRIST]))
            velocities.append(max(lw_vel, rw_vel))

        return float(np.mean(velocities)) if velocities else 0.0

    def _check_raised_arms(self, kps_norm: np.ndarray, confs: np.ndarray) -> bool:
        """Check if either wrist is above the corresponding shoulder.

        In image coordinates, "above" means a smaller y value.
        This is a strong indicator of an aggressive/fighting posture
        (raised fists, overhead swings).
        """
        if confs[self.L_WRIST] < 0.3 and confs[self.R_WRIST] < 0.3:
            return False

        left_raised = (
            confs[self.L_WRIST] > 0.3
            and confs[self.L_SHOULDER] > 0.3
            and kps_norm[self.L_WRIST][1] < kps_norm[self.L_SHOULDER][1]
        )
        right_raised = (
            confs[self.R_WRIST] > 0.3
            and confs[self.R_SHOULDER] > 0.3
            and kps_norm[self.R_WRIST][1] < kps_norm[self.R_SHOULDER][1]
        )
        return left_raised or right_raised

    def _find_close_pairs(self, persons: list) -> list:
        """Find all pairs of persons within arm's reach of each other.

        Uses the average bounding-box height of each pair as a scale
        reference: if two people's centers are closer than
        PROXIMITY_RATIO × avg_height, they are "within arm's reach".
        """
        close_pairs = []
        for i, j in itertools.combinations(range(len(persons)), 2):
            a_bbox = persons[i]["bbox"]
            b_bbox = persons[j]["bbox"]

            # Center of each bounding box
            a_cx = (a_bbox[0] + a_bbox[2]) / 2
            a_cy = (a_bbox[1] + a_bbox[3]) / 2
            b_cx = (b_bbox[0] + b_bbox[2]) / 2
            b_cy = (b_bbox[1] + b_bbox[3]) / 2

            pixel_dist = ((a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2) ** 0.5

            # Use average bbox height as a proxy for real-world scale
            a_h = a_bbox[3] - a_bbox[1]
            b_h = b_bbox[3] - b_bbox[1]
            avg_h = (a_h + b_h) / 2
            if avg_h <= 0:
                continue

            threshold_px = avg_h * self.PROXIMITY_RATIO

            # Physical contact check via bounding box overlap
            inter_w = max(0.0, min(a_bbox[2], b_bbox[2]) - max(a_bbox[0], b_bbox[0]))
            inter_h = max(0.0, min(a_bbox[3], b_bbox[3]) - max(a_bbox[1], b_bbox[1]))
            inter_area = inter_w * inter_h
            area_a = (a_bbox[2] - a_bbox[0]) * (a_bbox[3] - a_bbox[1])
            area_b = (b_bbox[2] - b_bbox[0]) * (b_bbox[3] - b_bbox[1])
            union = area_a + area_b - inter_area
            contact_iou = (inter_area / union) if union > 0 else 0.0

            if pixel_dist < threshold_px or contact_iou > 0.05:
                close_pairs.append((i, j, pixel_dist, threshold_px))

        return close_pairs

    # ------------------------------------------------------------------
    #  Annotation / drawing
    # ------------------------------------------------------------------

    def _annotate(self, image, pose_results, predictions, severity):
        """Draw skeletons, prediction labels, and severity banner."""
        annotated = image.copy()
        info = self.SEVERITY_LEVELS[severity]

        # Draw skeletons
        if pose_results[0].keypoints is not None:
            kps_data = pose_results[0].keypoints.data.cpu().numpy()
            skeleton_pairs = [
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
                (12, 14), (14, 16),
            ]
            for person_kps in kps_data:
                for (a, b) in skeleton_pairs:
                    if person_kps[a][2] > 0.3 and person_kps[b][2] > 0.3:
                        pt1 = (int(person_kps[a][0]), int(person_kps[a][1]))
                        pt2 = (int(person_kps[b][0]), int(person_kps[b][1]))
                        cv2.line(annotated, pt1, pt2, info["color"], 2)

                for kp in person_kps:
                    if kp[2] > 0.3:
                        cv2.circle(
                            annotated, (int(kp[0]), int(kp[1])), 3,
                            (255, 255, 255), -1,
                        )

        # Draw prediction labels
        for pred in predictions:
            label = (
                f"Person {pred['track_id']}: "
                f"{pred['prediction'].upper()} "
                f"({pred['confidence'] * 100:.0f}%)"
            )
            y_pos = 50 + pred["track_id"] * 25
            color = (0, 0, 255) if pred["prediction_id"] == 2 else (0, 140, 255)
            cv2.putText(
                annotated, label, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        # Severity banner
        cv2.rectangle(
            annotated, (0, 0), (annotated.shape[1], 30), info["color"], -1,
        )
        cv2.putText(
            annotated, f"FIGHT ALERT: {info['label']}", (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        return annotated

    def reset_buffers(self):
        """Clear keypoint history (call between videos/clips)."""
        self.keypoint_buffers.clear()
