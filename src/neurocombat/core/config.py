"""
NeuroCombat v5 - Configuration Management
Pydantic Settings for type-safe configuration loading.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional, List, Dict
from pathlib import Path
import yaml


class RedisSettings(BaseSettings):
    """Redis connection and stream configuration."""
    url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    db: int = Field(default=0, ge=0, le=15)
    max_connections: int = Field(default=50, ge=10, le=200)
    socket_timeout: int = Field(default=5, ge=1, le=60)
    socket_connect_timeout: int = Field(default=5, ge=1, le=60)


class VideoSettings(BaseSettings):
    """Video input configuration."""
    input_source: str | int = Field(default=0, description="Webcam ID or video file path")
    target_fps: int = Field(default=30, ge=1, le=120)
    width: int = Field(default=1920, ge=640, le=3840, alias="resolution.width")
    height: int = Field(default=1080, ge=480, le=2160, alias="resolution.height")
    frame_skip: int = Field(default=0, ge=0, le=10)


class PoseExtractionSettings(BaseSettings):
    """Pose extraction model configuration."""
    model_type: Literal["mediapipe", "movenet"] = "mediapipe"
    use_gpu: bool = True
    batch_size: int = Field(default=1, ge=1, le=8)
    inference_device: str = "cuda:0"
    
    # MediaPipe specific
    model_complexity: int = Field(default=2, ge=0, le=2)
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SequenceModelSettings(BaseSettings):
    """Sequence model architecture and inference settings."""
    model_type: Literal["bilstm", "transformer", "hybrid"] = "bilstm"
    
    # Architecture
    input_dim: int = 198
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = Field(default=0.3, ge=0.0, le=0.9)
    bidirectional: bool = True
    
    # Windowing
    sequence_length: int = Field(default=30, ge=10, le=120)
    stride: int = Field(default=15, ge=1, le=60)
    
    # Inference
    batch_size: int = Field(default=4, ge=1, le=32)
    use_gpu: bool = True
    device: str = "cuda:0"
    fp16: bool = True
    
    # Model paths
    checkpoint_path: str = "models/bilstm_v1.pth"
    onnx_path: Optional[str] = None


class EnsembleSettings(BaseSettings):
    """Ensemble fusion configuration."""
    method: Literal["bayesian", "voting", "stacking"] = "bayesian"
    adaptive_weights: bool = True
    update_interval: int = Field(default=100, ge=10, le=1000)
    min_ensemble_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    smoothing_window: int = Field(default=3, ge=1, le=10)


class CommentarySettings(BaseSettings):
    """Commentary engine configuration."""
    engine: Literal["pyttsx3", "elevenlabs", "google_tts"] = "pyttsx3"
    max_events_per_second: float = Field(default=2.0, ge=0.1, le=10.0)
    min_time_between_events: float = Field(default=0.5, ge=0.1, le=5.0)
    
    # TTS settings
    rate: int = Field(default=150, ge=50, le=300, description="Words per minute")
    volume: float = Field(default=0.9, ge=0.0, le=1.0)


class DashboardSettings(BaseSettings):
    """Dashboard API configuration."""
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    reload: bool = False
    websocket_path: str = "/ws"
    max_connections: int = Field(default=100, ge=10, le=1000)
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    enable_replay: bool = True
    replay_buffer_seconds: int = Field(default=300, ge=60, le=3600)


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    format: Literal["json", "text"] = "json"
    output: Literal["stdout", "file"] = "stdout"
    file_path: Optional[str] = "logs/neurocombat.log"
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class MetricsSettings(BaseSettings):
    """Prometheus metrics configuration."""
    enabled: bool = True
    port: int = Field(default=9090, ge=1024, le=65535)
    path: str = "/metrics"
    collect_system_metrics: bool = True
    collect_redis_metrics: bool = True
    collect_gpu_metrics: bool = True


class PerformanceSettings(BaseSettings):
    """Performance tuning configuration."""
    max_workers: int = Field(default=4, ge=1, le=32)
    worker_queue_size: int = Field(default=100, ge=10, le=1000)
    batch_timeout_ms: int = Field(default=100, ge=10, le=1000)
    max_batch_size: int = Field(default=10, ge=1, le=100)
    max_pending_messages: int = Field(default=500, ge=100, le=10000)
    drop_frames_on_overload: bool = False


class NeuroCombatConfig(BaseSettings):
    """
    Root configuration for NeuroCombat v5.
    
    Loads from:
        1. config/base.yaml (defaults)
        2. config/{environment}.yaml (overrides)
        3. Environment variables (highest priority)
    
    Example:
        config = NeuroCombatConfig()
        print(config.redis.url)
    """
    
    model_config = SettingsConfigDict(
        env_prefix="NEUROCOMBAT_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    
    # System
    project_name: str = "NeuroCombat v5"
    version: str = "5.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    
    # Component settings
    redis: RedisSettings = Field(default_factory=RedisSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    pose_extraction: PoseExtractionSettings = Field(default_factory=PoseExtractionSettings)
    sequence_model: SequenceModelSettings = Field(default_factory=SequenceModelSettings)
    ensemble: EnsembleSettings = Field(default_factory=EnsembleSettings)
    commentary: CommentarySettings = Field(default_factory=CommentarySettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    # Move classes
    class_labels: List[str] = Field(
        default_factory=lambda: [
            "idle", "jab", "cross", "hook", "uppercut",
            "kick_low", "kick_mid", "kick_high",
            "knee", "elbow", "block", "dodge",
            "clinch", "takedown", "ground"
        ]
    )
    
    @classmethod
    def load_from_yaml(cls, config_path: str | Path) -> "NeuroCombatConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            NeuroCombatConfig instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    @classmethod
    def load_with_overrides(
        cls,
        base_config: str | Path = "config/base.yaml",
        override_config: Optional[str | Path] = None,
    ) -> "NeuroCombatConfig":
        """
        Load base config with optional environment overrides.
        
        Args:
            base_config: Base configuration file
            override_config: Environment-specific overrides
        
        Returns:
            Merged configuration
        """
        # Load base config
        base = cls.load_from_yaml(base_config)
        
        # Apply overrides if provided
        if override_config:
            override_path = Path(override_config)
            if override_path.exists():
                with open(override_path, "r") as f:
                    override_data = yaml.safe_load(f)
                
                # Deep merge (simplified - use more robust merge for production)
                for key, value in override_data.items():
                    if hasattr(base, key):
                        setattr(base, key, value)
        
        return base


# Singleton instance
_config: Optional[NeuroCombatConfig] = None


def get_config() -> NeuroCombatConfig:
    """
    Get global configuration singleton.
    
    Returns:
        NeuroCombatConfig instance
    """
    global _config
    if _config is None:
        _config = NeuroCombatConfig()
    return _config


def load_config(
    base_path: str = "config/base.yaml",
    environment: Optional[str] = None,
) -> NeuroCombatConfig:
    """
    Load and cache configuration.
    
    Args:
        base_path: Base config file path
        environment: Environment name (development, staging, production)
    
    Returns:
        NeuroCombatConfig instance
    """
    global _config
    
    override_path = None
    if environment:
        override_path = f"config/{environment}.yaml"
    
    _config = NeuroCombatConfig.load_with_overrides(base_path, override_path)
    return _config
