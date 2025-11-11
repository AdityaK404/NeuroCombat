"""
NeuroCombat v5 - Pydantic Data Schemas
Core message types for Redis Streams communication.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Literal, Optional
from datetime import datetime
import numpy as np

__all__ = [
    "Joint",
    "PoseFrame",
    "MotionVector",
    "SequenceWindow",
    "PredictionPacket",
    "CommentaryEvent",
    "SystemMetrics",
]


# ========================================
# 1. Pose Extraction Messages
# ========================================

class Joint(BaseModel):
    """
    Single 3D joint with visibility confidence.
    Coordinates normalized to [-1, 1] or [0, 1] depending on model.
    """
    x: float = Field(..., ge=-1.0, le=1.0, description="Normalized x coordinate")
    y: float = Field(..., ge=-1.0, le=1.0, description="Normalized y coordinate")
    z: float = Field(..., ge=-1.0, le=1.0, description="Depth (relative to camera)")
    visibility: float = Field(..., ge=0.0, le=1.0, description="Detection confidence [0-1]")
    
    model_config = ConfigDict(frozen=True)  # Immutable for hashing/caching


class PoseFrame(BaseModel):
    """
    Atomic pose detection message for a single fighter in a single frame.
    
    Redis Stream: pose_frames:{fighter_id}
    Retention: 1800 messages (~60s @ 30fps)
    Consumer Groups: motion_workers, replay_workers
    
    Example:
        XADD pose_frames:fighter_1 * payload '{"frame_id": 12345, ...}'
    """
    frame_id: int = Field(..., ge=0, description="Monotonic frame counter (0-indexed)")
    timestamp_ms: int = Field(..., description="Unix epoch timestamp in milliseconds")
    fighter_id: str = Field(..., pattern=r"^fighter_[12]$", description="fighter_1 or fighter_2")
    
    # Joint data (33 for MediaPipe, 17 for MoveNet)
    joints: list[Joint] = Field(..., min_length=17, max_length=33)
    
    # Metadata
    model_source: Literal["mediapipe", "movenet"] = "mediapipe"
    bbox: tuple[float, float, float, float] = Field(
        ..., 
        description="Bounding box (x1, y1, x2, y2) in normalized coords [0-1]"
    )
    detection_confidence: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0,
        description="Overall pose detection confidence"
    )
    
    model_config = ConfigDict(frozen=True)
    
    @field_validator('bbox')
    @classmethod
    def validate_bbox(cls, v):
        """Ensure bbox is valid (x1 < x2, y1 < y2)."""
        x1, y1, x2, y2 = v
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Invalid bbox: {v}")
        return v
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert to NumPy array for downstream processing.
        
        Returns:
            np.ndarray: Shape (N_joints, 4) where columns are [x, y, z, visibility]
        """
        return np.array([[j.x, j.y, j.z, j.visibility] for j in self.joints], dtype=np.float32)
    
    @property
    def num_joints(self) -> int:
        """Number of detected joints."""
        return len(self.joints)


# ========================================
# 2. Motion Feature Messages
# ========================================

class MotionVector(BaseModel):
    """
    Derived motion features from pose temporal sequence.
    Computed by differencing consecutive PoseFrame messages.
    
    Redis Stream: motion_vectors:{fighter_id}
    Retention: 1800 messages
    Consumer Groups: seq_workers
    
    Feature Composition:
        - Velocity: 33 joints × 3D = 99 features
        - Acceleration: 33 joints × 3D = 99 features
        - Total: 198D vector
    """
    frame_id: int = Field(..., description="Reference frame ID (current frame)")
    timestamp_ms: int
    fighter_id: str
    
    # Motion features (flattened arrays)
    velocity: list[float] = Field(
        ..., 
        min_length=99, 
        max_length=99, 
        description="33 joints × 3D velocity (m/s normalized)"
    )
    acceleration: list[float] = Field(
        ..., 
        min_length=99, 
        max_length=99,
        description="33 joints × 3D acceleration (m/s²)"
    )
    
    # Optional: Joint angles (e.g., elbow, knee flexion)
    joint_angles: Optional[list[float]] = Field(
        None, 
        max_length=20,
        description="Pre-computed joint angles in radians"
    )
    
    # Normalization metadata
    velocity_norm: float = Field(..., ge=0.0, description="L2 norm of velocity vector")
    acceleration_norm: float = Field(default=0.0, ge=0.0)
    
    # Quality flags
    is_stable: bool = Field(
        default=True,
        description="False if velocity norm exceeds threshold (likely tracking error)"
    )
    
    model_config = ConfigDict(frozen=True)
    
    def to_array(self) -> np.ndarray:
        """
        Concatenate velocity and acceleration into single 198D vector.
        
        Returns:
            np.ndarray: Shape (198,)
        """
        return np.concatenate([self.velocity, self.acceleration], dtype=np.float32)


# ========================================
# 3. Sequence Model Input
# ========================================

class SequenceWindow(BaseModel):
    """
    Sliding window of motion vectors for sequence model input.
    
    NOT directly streamed via Redis—constructed in-memory by workers.
    
    Structure:
        - Window size: 30 frames (1 second @ 30fps)
        - Stride: 15 frames (50% overlap)
        - Shape: (30, 198) tensor
    """
    fighter_id: str
    start_frame: int = Field(..., description="First frame ID in window")
    end_frame: int = Field(..., description="Last frame ID in window")
    window_size: int = Field(default=30, ge=10, le=60)
    
    # Sequence data
    vectors: list[MotionVector] = Field(..., min_length=10, max_length=60)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('vectors')
    @classmethod
    def validate_window_size(cls, v, info):
        """Ensure vectors length matches window_size."""
        window_size = info.data.get('window_size', 30)
        if len(v) != window_size:
            raise ValueError(f"Expected {window_size} vectors, got {len(v)}")
        return v
    
    def to_tensor(self) -> np.ndarray:
        """
        Stack motion vectors into (T, D) tensor.
        
        Returns:
            np.ndarray: Shape (window_size, 198)
        """
        return np.vstack([v.to_array() for v in self.vectors])
    
    @property
    def duration_ms(self) -> int:
        """Duration of window in milliseconds."""
        if not self.vectors:
            return 0
        return self.vectors[-1].timestamp_ms - self.vectors[0].timestamp_ms


# ========================================
# 4. Prediction Messages
# ========================================

class PredictionPacket(BaseModel):
    """
    Classification result from sequence model or ensemble.
    
    Redis Stream: predictions:{fighter_id}
    Retention: 300 messages (~10s @ 30fps)
    Consumer Groups: ensemble_workers, commentary_workers, dashboard_workers
    
    Lifecycle:
        1. Sequence model → raw logits → predictions stream
        2. Ensemble fusion → weighted average → predictions stream (overwrite)
        3. Commentary/Dashboard consume final predictions
    """
    prediction_id: str = Field(..., description="UUID for distributed tracing")
    frame_id: int = Field(..., description="Reference frame (center of sequence window)")
    timestamp_ms: int
    fighter_id: str
    
    # Model output
    predicted_label: str = Field(..., description="Most likely class (e.g., 'jab', 'hook')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability")
    
    # Full distribution over all classes
    class_probabilities: dict[str, float] = Field(
        ..., 
        description="All class scores (must sum to ~1.0)"
    )
    
    # Provenance
    model_name: Literal["bilstm", "transformer", "cnn_lstm", "ensemble"] = "ensemble"
    model_version: str = Field(default="1.0.0", description="Model checkpoint version")
    
    # Ensemble metadata (only if model_name == 'ensemble')
    ensemble_weights: Optional[dict[str, float]] = Field(
        None,
        description="Bayesian weights used for fusion"
    )
    
    # Performance tracking
    inference_time_ms: float = Field(..., gt=0.0, description="Model forward pass time")
    total_latency_ms: float = Field(
        default=0.0,
        description="End-to-end latency from frame capture to prediction"
    )
    
    model_config = ConfigDict(frozen=True)
    
    @field_validator('class_probabilities')
    @classmethod
    def validate_probabilities(cls, v):
        """Ensure probabilities sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Class probabilities sum to {total}, expected ~1.0")
        return v
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if prediction exceeds confidence threshold."""
        return self.confidence >= 0.7


# ========================================
# 5. Commentary Messages
# ========================================

class CommentaryEvent(BaseModel):
    """
    Generated natural-language commentary snippet.
    
    Redis Stream: commentary_events
    Retention: 100 messages
    Consumer Groups: audio_mixer, dashboard_workers
    
    Triggered when:
        - High-confidence prediction detected
        - Significant move performed
        - Fight state changes (e.g., takedown, clinch)
    """
    event_id: str = Field(..., description="UUID for event tracking")
    timestamp_ms: int
    fighter_id: str
    
    # Commentary content
    text: str = Field(..., min_length=5, max_length=200, description="Human-readable text")
    audio_url: Optional[str] = Field(
        None,
        description="URL to pre-generated TTS audio file (if async generation)"
    )
    
    # Context
    trigger_prediction: str = Field(..., description="Move that triggered commentary (e.g., 'uppercut')")
    trigger_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Metadata
    excitement_level: Literal["low", "medium", "high", "knockout"] = "medium"
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    
    model_config = ConfigDict(frozen=True)


# ========================================
# 6. System Metrics
# ========================================

class SystemMetrics(BaseModel):
    """
    Performance telemetry for observability.
    
    Redis Stream: system_metrics
    Retention: 1000 messages
    Consumer Groups: prometheus_exporter
    
    Published by all components every N seconds or per-message.
    """
    component: str = Field(..., description="Module name (e.g., 'pose_extraction')")
    timestamp_ms: int
    
    # Latency metrics (milliseconds)
    processing_time: float = Field(..., ge=0.0, description="Time spent processing message")
    queue_wait_time: float = Field(default=0.0, ge=0.0, description="Time in Redis queue")
    
    # Throughput
    messages_processed: int = Field(default=1, ge=0)
    fps: float = Field(default=0.0, ge=0.0, description="Processing rate (frames/sec)")
    
    # Resource usage
    gpu_utilization: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU usage %")
    cpu_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    memory_mb: Optional[float] = Field(None, ge=0.0, description="RSS memory in MB")
    
    # Queue health
    redis_stream_length: Optional[int] = Field(None, ge=0, description="Pending messages")
    
    model_config = ConfigDict(frozen=True)


# ========================================
# Stream Configuration Constants
# ========================================

REDIS_STREAMS = {
    "pose_frames": {
        "pattern": "pose_frames:{fighter_id}",
        "maxlen": 1800,
        "consumer_groups": ["motion_workers", "replay_workers"],
    },
    "motion_vectors": {
        "pattern": "motion_vectors:{fighter_id}",
        "maxlen": 1800,
        "consumer_groups": ["seq_workers"],
    },
    "predictions": {
        "pattern": "predictions:{fighter_id}",
        "maxlen": 300,
        "consumer_groups": ["ensemble_workers", "commentary_workers", "dashboard_workers"],
    },
    "commentary_events": {
        "pattern": "commentary_events",
        "maxlen": 100,
        "consumer_groups": ["audio_mixer", "dashboard_workers"],
    },
    "system_metrics": {
        "pattern": "system_metrics",
        "maxlen": 1000,
        "consumer_groups": ["prometheus_exporter"],
    },
}
