"""
NeuroCombat v5 - Redis Stream Client
Async wrapper for Redis Streams with consumer groups and error handling.
"""

import asyncio
import logging
from typing import Optional, Dict, List, Callable, Any
from contextlib import asynccontextmanager

from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError, ResponseError

logger = logging.getLogger(__name__)


class RedisStreamClient:
    """
    Async Redis Streams client with built-in consumer group management.
    
    Features:
        - Automatic consumer group creation
        - Connection pooling
        - Health checks
        - Graceful shutdown
    
    Example:
        async with RedisStreamClient("redis://localhost:6379") as client:
            await client.publish("pose_frames:fighter_1", {"frame_id": 123, ...})
            
            async for msg_id, data in client.consume("pose_frames:fighter_1", "workers"):
                process(data)
                await client.ack("pose_frames:fighter_1", "workers", msg_id)
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 50,
        socket_timeout: int = 5,
        decode_responses: bool = False,  # Keep binary for msgpack
    ):
        self.redis_url = redis_url
        self.pool = ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=decode_responses,
        )
        self.redis: Optional[Redis] = None
        self._closed = False
    
    async def connect(self):
        """Initialize Redis connection."""
        if self.redis is None:
            self.redis = Redis(connection_pool=self.pool)
            logger.info(f"Connected to Redis at {self.redis_url}")
    
    async def close(self):
        """Close Redis connection and pool."""
        if not self._closed and self.redis:
            await self.redis.close()
            await self.pool.disconnect()
            self._closed = True
            logger.info("Redis connection closed")
    
    @asynccontextmanager
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    # ========================================
    # Producer Methods
    # ========================================
    
    async def publish(
        self,
        stream_name: str,
        data: Dict[str, Any],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        """
        Publish message to Redis Stream.
        
        Args:
            stream_name: Stream key (e.g., "pose_frames:fighter_1")
            data: Message payload (dict of field:value)
            maxlen: Max stream length (None = unlimited)
            approximate: Use ~ for faster trimming
        
        Returns:
            Message ID (e.g., "1699700000000-0")
        """
        try:
            msg_id = await self.redis.xadd(
                stream_name,
                data,
                maxlen=maxlen,
                approximate=approximate,
            )
            logger.debug(f"Published to {stream_name}: {msg_id}")
            return msg_id.decode() if isinstance(msg_id, bytes) else msg_id
        
        except RedisError as e:
            logger.error(f"Failed to publish to {stream_name}: {e}")
            raise
    
    # ========================================
    # Consumer Methods
    # ========================================
    
    async def create_consumer_group(
        self,
        stream_name: str,
        group_name: str,
        start_id: str = "0",
        mkstream: bool = True,
    ) -> bool:
        """
        Create consumer group (idempotent).
        
        Args:
            stream_name: Stream key
            group_name: Consumer group name
            start_id: Starting position ("0" = beginning, "$" = latest)
            mkstream: Create stream if doesn't exist
        
        Returns:
            True if created, False if already exists
        """
        try:
            await self.redis.xgroup_create(
                stream_name,
                group_name,
                id=start_id,
                mkstream=mkstream,
            )
            logger.info(f"Created consumer group '{group_name}' on {stream_name}")
            return True
        
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group '{group_name}' already exists")
                return False
            raise
    
    async def consume(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 100,
    ):
        """
        Consume messages from stream (blocking iterator).
        
        Args:
            stream_name: Stream to consume from
            group_name: Consumer group
            consumer_name: Unique consumer ID
            count: Max messages per read
            block_ms: Block timeout (0 = non-blocking)
        
        Yields:
            Tuple of (message_id, data_dict)
        
        Example:
            async for msg_id, data in client.consume("stream", "group", "worker1"):
                print(data[b"payload"])
                await client.ack("stream", "group", msg_id)
        """
        while not self._closed:
            try:
                # XREADGROUP GROUP group consumer BLOCK ms COUNT n STREAMS stream >
                messages = await self.redis.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={stream_name: ">"},
                    count=count,
                    block=block_ms,
                )
                
                if not messages:
                    await asyncio.sleep(0.01)  # Yield to event loop
                    continue
                
                # Messages format: [(stream_name, [(msg_id, {field: value})])]
                for stream, msg_list in messages:
                    for msg_id, data in msg_list:
                        yield msg_id, data
            
            except ConnectionError as e:
                logger.error(f"Redis connection lost: {e}")
                await asyncio.sleep(1)  # Reconnect backoff
            
            except Exception as e:
                logger.error(f"Consume error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def ack(self, stream_name: str, group_name: str, *msg_ids: str):
        """
        Acknowledge message(s) as processed.
        
        Args:
            stream_name: Stream name
            group_name: Consumer group
            msg_ids: One or more message IDs to ACK
        """
        try:
            count = await self.redis.xack(stream_name, group_name, *msg_ids)
            logger.debug(f"ACKed {count} messages on {stream_name}")
        except RedisError as e:
            logger.error(f"ACK failed: {e}")
    
    async def claim_pending(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        min_idle_time: int = 60000,  # 1 minute
        count: int = 10,
    ) -> List[tuple]:
        """
        Claim pending messages from dead consumers (failure recovery).
        
        Args:
            stream_name: Stream name
            group_name: Consumer group
            consumer_name: This consumer's name
            min_idle_time: Min idle time (ms) to claim
            count: Max messages to claim
        
        Returns:
            List of (msg_id, data) tuples
        """
        try:
            messages = await self.redis.xautoclaim(
                stream_name,
                group_name,
                consumer_name,
                min_idle_time=min_idle_time,
                count=count,
            )
            if messages:
                logger.info(f"Claimed {len(messages)} pending messages")
            return messages
        except RedisError as e:
            logger.error(f"Claim failed: {e}")
            return []
    
    # ========================================
    # Monitoring
    # ========================================
    
    async def get_stream_info(self, stream_name: str) -> Dict:
        """Get stream metadata (length, groups, etc.)."""
        try:
            return await self.redis.xinfo_stream(stream_name)
        except RedisError as e:
            logger.error(f"Failed to get info for {stream_name}: {e}")
            return {}
    
    async def get_stream_length(self, stream_name: str) -> int:
        """Get number of messages in stream."""
        try:
            return await self.redis.xlen(stream_name)
        except RedisError:
            return 0
    
    async def get_pending_count(self, stream_name: str, group_name: str) -> int:
        """Get number of pending (unacknowledged) messages."""
        try:
            pending = await self.redis.xpending(stream_name, group_name)
            # pending = {count, min_id, max_id, consumers}
            return pending.get("pending", 0) if isinstance(pending, dict) else 0
        except RedisError:
            return 0
    
    async def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            await self.redis.ping()
            return True
        except RedisError:
            return False


# ========================================
# Helper Functions
# ========================================

async def setup_streams(
    client: RedisStreamClient,
    stream_configs: Dict[str, Dict],
):
    """
    Initialize all Redis Streams and consumer groups.
    
    Args:
        client: RedisStreamClient instance
        stream_configs: Dict mapping stream patterns to config
    
    Example:
        await setup_streams(client, REDIS_STREAMS)
    """
    for stream_key, config in stream_configs.items():
        pattern = config["pattern"]
        
        # For patterns with {fighter_id}, create for both fighters
        if "{fighter_id}" in pattern:
            stream_names = [
                pattern.replace("{fighter_id}", "fighter_1"),
                pattern.replace("{fighter_id}", "fighter_2"),
            ]
        else:
            stream_names = [pattern]
        
        # Create streams and consumer groups
        for stream_name in stream_names:
            for group_name in config["consumer_groups"]:
                try:
                    await client.create_consumer_group(
                        stream_name, 
                        group_name,
                        start_id="$",  # Start from latest (ignore old messages)
                        mkstream=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create {group_name} on {stream_name}: {e}")
    
    logger.info("All Redis Streams initialized")
