# NeuroCombat v5 - Implementation Roadmap

## 🎯 Executive Summary

This document provides a **step-by-step implementation guide** for building NeuroCombat v5 from the architectural blueprint. Each phase includes specific tasks, file paths, acceptance criteria, and testing requirements.

**Total Estimated Timeline:** 8 weeks (1 developer)

---

## Phase 1: Foundation & Core Infrastructure (Week 1-2)

### 1.1 Project Setup
**Duration:** 2 days

- [x] Initialize Poetry project (`pyproject.toml`)
- [x] Create directory structure
- [x] Setup Git repository
- [ ] Configure pre-commit hooks
- [ ] Setup CI/CD pipeline (GitHub Actions)

**Files to Create:**
```
.pre-commit-config.yaml
.github/workflows/ci.yml
.gitignore
```

**Acceptance Criteria:**
- Poetry install completes without errors
- Pre-commit hooks run on commit
- CI pipeline runs tests on push

---

### 1.2 Pydantic Schemas & Data Contracts
**Duration:** 3 days

- [x] Implement `Joint` and `PoseFrame` models
- [x] Implement `MotionVector` model
- [x] Implement `SequenceWindow` model
- [x] Implement `PredictionPacket` model
- [x] Implement `CommentaryEvent` model
- [x] Implement `SystemMetrics` model
- [ ] Write unit tests for all schemas
- [ ] Add JSON schema export utility

**Files:**
```
src/neurocombat/schemas/__init__.py  ✅ Created
src/neurocombat/tests/unit/test_schemas.py
scripts/export_json_schemas.py
```

**Tests:**
```python
# src/neurocombat/tests/unit/test_schemas.py
def test_pose_frame_validation():
    pose = PoseFrame(
        frame_id=1,
        timestamp_ms=1699700000000,
        fighter_id="fighter_1",
        joints=[Joint(x=0.5, y=0.3, z=-0.1, visibility=0.95)] * 33,
        model_source="mediapipe",
        bbox=(0.2, 0.1, 0.8, 0.9)
    )
    assert pose.num_joints == 33
    assert pose.to_numpy().shape == (33, 4)

def test_motion_vector_to_array():
    motion = MotionVector(
        frame_id=1,
        timestamp_ms=1699700000000,
        fighter_id="fighter_1",
        velocity=[0.01] * 99,
        acceleration=[0.001] * 99,
        velocity_norm=0.45
    )
    arr = motion.to_array()
    assert arr.shape == (198,)
```

---

### 1.3 Redis Stream Client
**Duration:** 3 days

- [x] Implement `RedisStreamClient` class
- [ ] Add connection pooling
- [ ] Implement consumer group management
- [ ] Add health check methods
- [ ] Write integration tests
- [ ] Add reconnection logic

**Files:**
```
src/neurocombat/core/redis_client.py  ✅ Created
src/neurocombat/tests/integration/test_redis.py
```

**Tests:**
```python
# Integration test with real Redis instance
@pytest.mark.asyncio
async def test_publish_consume_cycle(redis_client):
    stream = "test_stream"
    group = "test_group"
    
    # Publish
    msg_id = await redis_client.publish(
        stream,
        {"payload": '{"test": "data"}'}
    )
    assert msg_id is not None
    
    # Create consumer group
    await redis_client.create_consumer_group(stream, group)
    
    # Consume
    async for msg_id, data in redis_client.consume(stream, group, "worker1", count=1):
        assert data[b"payload"] == b'{"test": "data"}'
        await redis_client.ack(stream, group, msg_id)
        break
```

---

### 1.4 Configuration Management
**Duration:** 2 days

- [x] Implement Pydantic Settings models
- [x] Create YAML config loader
- [x] Add environment variable support
- [ ] Write config validation tests
- [ ] Document all configuration options

**Files:**
```
src/neurocombat/core/config.py  ✅ Created
config/base.yaml  ✅ Created
config/production.yaml  ✅ Created
src/neurocombat/tests/unit/test_config.py
```

---

### 1.5 Logging & Metrics
**Duration:** 2 days

- [ ] Setup structlog configuration
- [ ] Implement Prometheus metrics
- [ ] Create `SystemMetrics` publisher
- [ ] Add OpenTelemetry tracing
- [ ] Write logging tests

**Files:**
```
src/neurocombat/core/logging.py
src/neurocombat/core/metrics.py
src/neurocombat/core/tracing.py
```

**Example:**
```python
# src/neurocombat/core/logging.py
import structlog

def configure_logging(level: str = "INFO", format: str = "json"):
    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

---

## Phase 2: Pose Extraction & Tracking (Week 3)

### 2.1 MediaPipe Pose Extractor
**Duration:** 3 days

- [ ] Implement `MediaPipeExtractor` class
- [ ] Add GPU acceleration (TFLite GPU delegate)
- [ ] Implement frame preprocessing
- [ ] Add bounding box extraction
- [ ] Write unit tests with mock frames

**Files:**
```
src/neurocombat/modules/pose_extraction/mediapipe_extractor.py
src/neurocombat/modules/pose_extraction/__init__.py
src/neurocombat/tests/unit/test_mediapipe.py
```

**Implementation:**
```python
# src/neurocombat/modules/pose_extraction/mediapipe_extractor.py
import mediapipe as mp
import numpy as np
from neurocombat.schemas import PoseFrame, Joint

class MediaPipeExtractor:
    def __init__(self, model_complexity: int = 2, use_gpu: bool = True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract(self, frame: np.ndarray) -> PoseFrame | None:
        results = self.pose.process(frame)
        
        if not results.pose_landmarks:
            return None
        
        # Convert landmarks to Joint objects
        joints = []
        for lm in results.pose_landmarks.landmark:
            joints.append(Joint(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))
        
        # Compute bounding box
        bbox = self._compute_bbox(joints)
        
        return PoseFrame(
            frame_id=0,  # Set by worker
            timestamp_ms=0,
            fighter_id="fighter_1",
            joints=joints,
            model_source="mediapipe",
            bbox=bbox
        )
    
    def _compute_bbox(self, joints: list[Joint]) -> tuple[float, float, float, float]:
        xs = [j.x for j in joints if j.visibility > 0.5]
        ys = [j.y for j in joints if j.visibility > 0.5]
        return (min(xs), min(ys), max(xs), max(ys))
```

---

### 2.2 ByteTrack Dual Fighter Tracker
**Duration:** 2 days

- [ ] Implement `ByteTrackTracker` class
- [ ] Add fighter ID assignment logic
- [ ] Implement centroid-based fallback
- [ ] Write tracking tests with synthetic data

**Files:**
```
src/neurocombat/modules/tracker/bytetrack.py
src/neurocombat/modules/tracker/__init__.py
src/neurocombat/tests/unit/test_tracker.py
```

---

### 2.3 Pose Extraction Worker
**Duration:** 2 days

- [ ] Implement async video capture
- [ ] Create `PoseExtractionWorker` class
- [ ] Add Redis Stream publishing
- [ ] Implement FPS throttling
- [ ] Write integration test

**Files:**
```
src/neurocombat/modules/pose_extraction/worker.py
```

**Implementation:**
```python
# src/neurocombat/modules/pose_extraction/worker.py
import asyncio
import cv2
import time
from neurocombat.core.redis_client import RedisStreamClient
from neurocombat.core.config import get_config
from neurocombat.modules.pose_extraction.mediapipe_extractor import MediaPipeExtractor
from neurocombat.modules.tracker.bytetrack import ByteTrackTracker

class PoseExtractionWorker:
    def __init__(self, config):
        self.config = config
        self.redis = RedisStreamClient(config.redis.url)
        self.extractor = MediaPipeExtractor(
            model_complexity=config.pose_extraction.model_complexity,
            use_gpu=config.pose_extraction.use_gpu
        )
        self.tracker = ByteTrackTracker()
        self.frame_id = 0
    
    async def run(self):
        await self.redis.connect()
        
        # Open video source
        cap = cv2.VideoCapture(self.config.video.input_source)
        target_fps = self.config.video.target_fps
        frame_interval = 1.0 / target_fps
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract poses
                poses = self.extractor.extract(frame)
                if poses:
                    # Track fighters
                    tracked_poses = self.tracker.update([poses])
                    
                    # Publish to Redis
                    for pose in tracked_poses:
                        pose.frame_id = self.frame_id
                        pose.timestamp_ms = int(time.time() * 1000)
                        
                        await self.redis.publish(
                            f"pose_frames:{pose.fighter_id}",
                            {"payload": pose.model_dump_json()},
                            maxlen=1800
                        )
                
                self.frame_id += 1
                
                # FPS throttling
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)
        
        finally:
            cap.release()
            await self.redis.close()
```

---

## Phase 3: Motion Features & Sequence Model (Week 4)

### 3.1 Motion Feature Computation
**Duration:** 2 days

- [ ] Implement velocity computation (finite differences)
- [ ] Implement acceleration computation
- [ ] Add joint angle calculation
- [ ] Implement normalization
- [ ] Write numerical tests

**Files:**
```
src/neurocombat/modules/motion_features/compute.py
src/neurocombat/modules/motion_features/worker.py
src/neurocombat/tests/unit/test_motion_features.py
```

**Implementation:**
```python
# src/neurocombat/modules/motion_features/compute.py
import numpy as np
from neurocombat.schemas import PoseFrame, MotionVector

def compute_motion_features(
    current: PoseFrame,
    previous: PoseFrame
) -> MotionVector:
    curr_joints = current.to_numpy()[:, :3]  # (33, 3) xyz only
    prev_joints = previous.to_numpy()[:, :3]
    
    # Time delta
    dt = (current.timestamp_ms - previous.timestamp_ms) / 1000.0
    if dt == 0:
        dt = 1.0 / 30.0  # Default 30 FPS
    
    # Velocity
    velocity = (curr_joints - prev_joints) / dt
    velocity_flat = velocity.flatten().tolist()
    velocity_norm = float(np.linalg.norm(velocity))
    
    # Acceleration (requires 3 frames, stub for now)
    acceleration = np.zeros_like(velocity)
    acceleration_flat = acceleration.flatten().tolist()
    
    return MotionVector(
        frame_id=current.frame_id,
        timestamp_ms=current.timestamp_ms,
        fighter_id=current.fighter_id,
        velocity=velocity_flat,
        acceleration=acceleration_flat,
        velocity_norm=velocity_norm
    )
```

---

### 3.2 Bi-LSTM Model
**Duration:** 3 days

- [ ] Implement PyTorch Bi-LSTM architecture
- [ ] Add batch inference support
- [ ] Implement model loading/saving
- [ ] Export to ONNX
- [ ] Write model tests

**Files:**
```
src/neurocombat/modules/sequence_model/bilstm.py
src/neurocombat/modules/sequence_model/inference.py
src/neurocombat/tests/unit/test_bilstm.py
```

**Implementation:**
```python
# src/neurocombat/modules/sequence_model/bilstm.py
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 198,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 15,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        
        # Classify
        logits = self.classifier(last_output)
        return logits
```

---

### 3.3 Sequence Model Worker
**Duration:** 2 days

- [ ] Implement windowing logic
- [ ] Create batched inference engine
- [ ] Add GPU memory management
- [ ] Implement Redis consumer
- [ ] Write integration test

**Files:**
```
src/neurocombat/modules/sequence_model/worker.py
```

---

## Phase 4: Ensemble & Commentary (Week 5)

### 4.1 Bayesian Ensemble Fusion
**Duration:** 2 days

- [ ] Implement Bayesian averaging
- [ ] Add adaptive weight update
- [ ] Implement confidence gating
- [ ] Write fusion tests

**Files:**
```
src/neurocombat/modules/ensemble/bayesian_fusion.py
src/neurocombat/modules/ensemble/worker.py
```

---

### 4.2 Commentary Engine
**Duration:** 3 days

- [ ] Implement rule-based NLG templates
- [ ] Add pyttsx3 TTS integration
- [ ] Implement rate limiting
- [ ] Add excitement level logic
- [ ] Write commentary tests

**Files:**
```
src/neurocombat/modules/commentary/nlg_templates.py
src/neurocombat/modules/commentary/tts_engine.py
src/neurocombat/modules/commentary/worker.py
```

---

## Phase 5: Dashboard & Deployment (Week 6)

### 5.1 FastAPI Backend
**Duration:** 3 days

- [ ] Implement REST API endpoints
- [ ] Add WebSocket live feed
- [ ] Create health check endpoint
- [ ] Add CORS middleware
- [ ] Write API tests

**Files:**
```
src/neurocombat/modules/dashboard/api.py
src/neurocombat/modules/dashboard/websocket.py
src/neurocombat/tests/integration/test_api.py
```

---

### 5.2 Docker & Kubernetes
**Duration:** 2 days

- [x] Create Dockerfiles (pose, inference, dashboard)
- [x] Create docker-compose.yml
- [ ] Create Kubernetes manifests
- [ ] Test container builds
- [ ] Write deployment docs

**Files:**
```
docker/Dockerfile.pose  ✅ Created
docker/Dockerfile.base  ✅ Created
docker/Dockerfile.inference  ✅ Created
docker/Dockerfile.dashboard  ✅ Created
docker-compose.yml  ✅ Created
kubernetes/*.yaml
```

---

## Phase 6: Testing & Optimization (Week 7)

### 6.1 End-to-End Testing
**Duration:** 3 days

- [ ] Write full pipeline integration test
- [ ] Add performance benchmarks
- [ ] Measure latency at each stage
- [ ] Profile memory usage
- [ ] Optimize bottlenecks

**Files:**
```
src/neurocombat/tests/integration/test_pipeline.py
scripts/benchmark_latency.py
```

---

### 6.2 Load Testing
**Duration:** 2 days

- [ ] Create Locust test scenarios
- [ ] Test with synthetic video streams
- [ ] Measure throughput limits
- [ ] Test Redis back-pressure
- [ ] Document performance results

**Files:**
```
src/neurocombat/tests/load/test_pipeline.py
```

---

## Phase 7: Dual-Fighter Extension (Week 8)

### 7.1 Interaction Features
**Duration:** 2 days

- [ ] Implement fighter-fighter distance metrics
- [ ] Add relative velocity computation
- [ ] Create `InteractionVector` schema
- [ ] Write interaction tests

**Files:**
```
src/neurocombat/schemas/interaction.py
src/neurocombat/modules/motion_features/interaction.py
```

---

### 7.2 Dual-Fighter Commentary
**Duration:** 2 days

- [ ] Add context-aware templates
- [ ] Implement fighter name resolution
- [ ] Add interaction-based commentary
- [ ] Test dual-fighter scenarios

---

## Testing Strategy

### Unit Tests
- Cover >80% of code
- Mock external dependencies (Redis, GPU)
- Run in CI on every commit

### Integration Tests
- Test with real Redis instance
- Use Docker Compose for services
- Measure end-to-end latency

### Load Tests
- Simulate 30-60 FPS video input
- Test with 2-4 concurrent workers
- Measure Redis queue depth under load

---

## Acceptance Criteria

### Phase 1
- ✅ All schemas pass validation
- ✅ Redis client publishes and consumes messages
- ✅ Configuration loads from YAML and env vars

### Phase 2
- [ ] MediaPipe extracts 33 joints @ >25 FPS
- [ ] ByteTrack assigns fighter IDs correctly
- [ ] Pose frames published to Redis

### Phase 3
- [ ] Motion features computed at <5ms/frame
- [ ] Bi-LSTM achieves >80% validation accuracy
- [ ] Predictions published to Redis

### Phase 4
- [ ] Ensemble fusion runs at <3ms/prediction
- [ ] Commentary generates 1-2 events/second
- [ ] TTS audio plays without delay

### Phase 5
- [ ] Dashboard displays live predictions
- [ ] Docker Compose starts all services
- [ ] Prometheus collects metrics

### Phase 6
- [ ] End-to-end latency <50ms
- [ ] Throughput ≥30 FPS
- [ ] Memory usage <2GB

### Phase 7
- [ ] Dual-fighter tracking maintains IDs
- [ ] Interaction features computed correctly
- [ ] Commentary switches context smoothly

---

## Next Steps

1. **Start Phase 1**: Implement unit tests for schemas
2. **Setup CI/CD**: Create GitHub Actions workflow
3. **Redis Testing**: Write integration tests for Redis client
4. **Model Training**: Train Bi-LSTM on synthetic data
5. **Documentation**: Complete API reference docs

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-11  
**Status:** Ready for Implementation ✅
