"""
services/ollama_verifier.py — Confirms or rejects a flagged incident with a local
vision-language model served by Ollama (default: qwen3-vl:4b).

Unlike the old single-frame check, it looks at a short sequence of frames,
because fights and dog bites are actions: one still frame of two people close
together looks the same whether they are hugging or fighting.

It never pretends: if Ollama is unreachable or answers with something that
isn't the expected JSON, the result has success=False so callers keep the
detector's verdict instead of treating it as a rejection.
"""
import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger("vids.ollama_verifier")

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3-vl:4b"

_SEQUENCE_NOTE = (
    "These {n} images are consecutive frames, in time order, from one residential "
    "CCTV camera covering about {seconds:.0f} seconds. "
)

# One question per threat type. Each asks for the specific action and names the
# harmless look-alikes that caused false alarms in testing.
PROMPTS: Dict[str, str] = {
    "fight_verify": (
        "Is a physical fight or assault happening: someone punching, kicking, "
        "shoving, grappling or beating another person? Answer false for hugging, "
        "handshakes, play, sports, dancing, people standing or talking close together."
    ),
    "weapon_verify": (
        "Is a person clearly holding a weapon such as a knife, gun, machete, rod "
        "or bat? Answer false for phones, keys, umbrellas, bags, tools being used "
        "for work, or when you cannot actually see the object in a hand."
    ),
    "animal_assault_verify": (
        "Is a dog attacking a person: biting, lunging at, jumping on aggressively "
        "or chasing someone who is fleeing? Answer false for a dog walking, "
        "sniffing, being walked on a leash, being petted, or playing calmly."
    ),
    "fire_verify": (
        "Is there real fire or smoke from something burning? Answer false for "
        "lamps, sunlight, reflections, steam, fog, dust or orange objects."
    ),
    "security_audit": (
        "Is anything dangerous happening: a fight, a weapon being held, an animal "
        "attacking a person, or fire? Answer false for normal activity."
    ),
}

_ANSWER_RULES = (
    " Respond only with JSON matching the schema. Set confirmed to true only if "
    "the images clearly show it; if unsure, set confirmed to false and explain "
    "in description. confidence is 0 to 1."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "confirmed": {"type": "boolean"},
        "confidence": {"type": "number"},
        "description": {"type": "string"},
    },
    "required": ["confirmed", "confidence", "description"],
}


def _encode(frame: np.ndarray, max_side: int = 768) -> str:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse(text: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of the reply (tolerates <think> blocks or fences)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("confirmed"), bool):
        return None
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "confirmed": data["confirmed"],
        "confidence": max(0.0, min(1.0, conf)),
        "description": str(data.get("description", "")).strip(),
    }


class OllamaVerifier:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[float] = None, image_size: Optional[int] = None):
        self.base_url = (base_url or os.getenv("OLLAMA_URL", DEFAULT_URL)).rstrip("/")
        self.model = model or os.getenv("OLLAMA_VISION_MODEL", DEFAULT_MODEL)
        # On a CPU-only PC a check can take minutes: raise OLLAMA_TIMEOUT and
        # lower OLLAMA_IMAGE_SIZE (see deploy/frigate/windows/README.md).
        self.timeout = timeout if timeout is not None else float(os.getenv("OLLAMA_TIMEOUT", "90"))
        self.image_size = image_size if image_size is not None else int(os.getenv("OLLAMA_IMAGE_SIZE", "768"))
        self._available: Optional[bool] = None
        self._checked_at = 0.0

    def is_available(self) -> bool:
        """True if Ollama is up and the model is pulled. Cached for 60s."""
        if self._available is not None and time.time() - self._checked_at < 60:
            return self._available
        self._checked_at = time.time()
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            r.raise_for_status()
            names = {m.get("name", "") for m in r.json().get("models", [])}
            wanted = self.model if ":" in self.model else f"{self.model}:latest"
            self._available = wanted in names
            if not self._available:
                logger.warning("Ollama is running but model %s is not pulled. Run: ollama pull %s "
                               "(if that is blocked: deploy/frigate/scripts/import_qwen_from_dockerhub.sh)",
                               self.model, self.model)
        except Exception as e:
            logger.warning("Ollama not reachable at %s: %s", self.base_url, e)
            self._available = False
        return self._available

    def verify(self, frames: List[np.ndarray], context_type: str = "security_audit",
               clip_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Ask the model whether the frames show the threat. See module docstring."""
        t0 = time.time()
        base = {"model_used": f"ollama:{self.model}", "verified_threat": False,
                "confidence": 0.0, "reasoning": ""}
        frames = [f for f in frames if f is not None and getattr(f, "size", 0)]
        if not frames:
            return {**base, "success": False, "reasoning": "no frames to check"}
        if not self.is_available():
            return {**base, "success": False, "unavailable": True,
                    "reasoning": f"Ollama model {self.model} unavailable"}

        seconds = clip_seconds if clip_seconds is not None else max(1.0, len(frames) * 0.5)
        prompt = ""
        if len(frames) > 1:
            prompt = _SEQUENCE_NOTE.format(n=len(frames), seconds=seconds)
        prompt += PROMPTS.get(context_type, PROMPTS["security_audit"]) + _ANSWER_RULES

        body = {
            "model": self.model,
            "stream": False,
            "format": _SCHEMA,
            "options": {"temperature": 0},
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [_encode(f, self.image_size) for f in frames],
            }],
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            reply = r.json().get("message", {}).get("content", "")
        except Exception as e:
            logger.error("Ollama verification failed: %s", e)
            self._available = None  # re-check next time
            return {**base, "success": False, "reasoning": f"ollama error: {e}",
                    "latency_seconds": round(time.time() - t0, 1)}

        parsed = _parse(reply)
        if parsed is None:
            logger.warning("Ollama gave an unusable answer: %.200s", reply)
            return {**base, "success": False, "reasoning": "unparseable model answer",
                    "raw": reply[:500], "latency_seconds": round(time.time() - t0, 1)}

        return {
            **base,
            "success": True,
            "verified_threat": parsed["confirmed"],
            "confidence": parsed["confidence"],
            "reasoning": parsed["description"],
            "frames_checked": len(frames),
            "latency_seconds": round(time.time() - t0, 1),
        }


_instance: Optional[OllamaVerifier] = None


def get_ollama_verifier() -> OllamaVerifier:
    global _instance
    if _instance is None:
        _instance = OllamaVerifier()
    return _instance
