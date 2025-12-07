"""User history management with all-MiniLM-L6-v2 embeddings and optional Redis support"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import json

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("mem0_manager")


class Mem0Manager:
    """Manager for user history with semantic search using all-MiniLM-L6-v2"""

    def __init__(self):
        """Initialize memory storage with embeddings and optional Redis"""
        try:
            self.storage_file = Config.DATA_DIR / "user_memories.json"
            self.embeddings_file = Config.DATA_DIR / "user_embeddings.npy"

            # Initialize Redis if enabled and available
            self.redis_client = None
            self.use_redis = Config.USE_REDIS and REDIS_AVAILABLE

            if self.use_redis:
                try:
                    logger.info("Attempting to connect to Redis...")
                    self.redis_client = redis.Redis(
                        host=Config.REDIS_HOST,
                        port=Config.REDIS_PORT,
                        db=Config.REDIS_DB,
                        password=Config.REDIS_PASSWORD,
                        decode_responses=True,
                        socket_connect_timeout=2,
                    )
                    # Test connection
                    self.redis_client.ping()
                    logger.info(
                        f"Redis connected successfully at {Config.REDIS_HOST}:{Config.REDIS_PORT}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Redis connection failed: {e}. Falling back to file storage."
                    )
                    self.redis_client = None
                    self.use_redis = False
            else:
                if Config.USE_REDIS and not REDIS_AVAILABLE:
                    logger.warning(
                        "Redis enabled in config but 'redis' package not installed. Using file storage."
                    )
                    logger.warning("Install with: pip install redis")

            self.memories = self._load_memories()

            # Load the all-MiniLM-L6-v2 model for embeddings
            logger.info("Loading all-MiniLM-L6-v2 embedding model...")
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

            # Load or initialize embeddings cache
            self.embeddings_cache = self._load_embeddings()

            storage_type = "Redis" if self.use_redis else "JSON file"
            logger.info(
                f"Memory manager initialized with all-MiniLM-L6-v2 embeddings ({storage_type} storage)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            self.memories = defaultdict(list)
            self.embeddings_cache = {}
            self.redis_client = None
            self.use_redis = False

    def _load_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load memories from Redis or file"""
        if self.use_redis and self.redis_client:
            try:
                # Load all user memories from Redis
                memories = defaultdict(list)
                for key in self.redis_client.scan_iter("mem0:*"):
                    user_id = key.split(":", 1)[1]
                    data = self.redis_client.get(key)
                    if data:
                        memories[user_id] = json.loads(data)
                logger.info(f"Loaded memories for {len(memories)} users from Redis")
                return memories
            except Exception as e:
                logger.warning(f"Could not load memories from Redis: {e}")

        # Fallback to file storage
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r") as f:
                    return defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load memories from file: {e}")
        return defaultdict(list)

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load embeddings cache from file"""
        if self.embeddings_file.exists():
            try:
                embeddings_dict = np.load(
                    self.embeddings_file, allow_pickle=True
                ).item()
                logger.info(f"Loaded {len(embeddings_dict)} cached embeddings")
                return embeddings_dict
            except Exception as e:
                logger.warning(f"Could not load embeddings cache: {e}")
        return {}

    def _save_memories(self) -> None:
        """Save memories to Redis or file"""
        try:
            if self.use_redis and self.redis_client:
                # Save to Redis with expiry (24 hours)
                for user_id, user_memories in self.memories.items():
                    key = f"mem0:{user_id}"
                    self.redis_client.setex(
                        key,
                        86400,  # 24 hours TTL
                        json.dumps(user_memories),
                    )
                logger.debug(f"Saved memories to Redis for {len(self.memories)} users")
            else:
                # Save to file
                with open(self.storage_file, "w") as f:
                    json.dump(dict(self.memories), f, indent=2)
                logger.debug("Saved memories to file")
        except Exception as e:
            logger.error(f"Error saving memories: {e}")

    def _save_embeddings(self) -> None:
        """Save embeddings cache to file"""
        try:
            np.save(self.embeddings_file, self.embeddings_cache)
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using all-MiniLM-L6-v2"""
        try:
            # Check cache first
            if text in self.embeddings_cache:
                return self.embeddings_cache[text]

            # Generate new embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(384)  # all-MiniLM-L6-v2 has 384 dimensions

    def add_memory(
        self, user_id: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a memory to user history with embedding

        Args:
            user_id: User identifier
            message: Message to store
            metadata: Additional metadata
        """
        try:
            # Generate embedding for the message (stored in cache)
            self._get_embedding(message)

            memory_entry = {
                "memory": message,
                "content": message,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "id": f"{user_id}_{len(self.memories[user_id])}",
            }
            self.memories[user_id].append(memory_entry)
            self._save_memories()
            self._save_embeddings()
            logger.info(f"Added memory for user {user_id} with embedding")
        except Exception as e:
            logger.error(f"Error adding memory for user {user_id}: {e}")
            # Don't raise - just log the error so the app continues working

    def get_memories(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve user memories

        Args:
            user_id: User identifier
            limit: Maximum number of memories to retrieve

        Returns:
            List of memories
        """
        try:
            user_memories = self.memories.get(user_id, [])
            logger.info(f"Retrieved {len(user_memories)} memories for user {user_id}")
            return user_memories[-limit:] if user_memories else []
        except Exception as e:
            logger.error(f"Error retrieving memories for user {user_id}: {e}")
            return []

    def search_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search user memories using semantic similarity (all-MiniLM-L6-v2)

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results

        Returns:
            List of relevant memories ranked by similarity
        """
        try:
            user_memories = self.memories.get(user_id, [])
            if not user_memories:
                return []

            # Get query embedding
            query_embedding = self._get_embedding(query)

            # Calculate similarity scores for all memories
            similarities = []
            for mem in user_memories:
                mem_text = mem.get("memory", mem.get("content", ""))
                mem_embedding = self._get_embedding(mem_text)

                # Cosine similarity
                similarity = np.dot(query_embedding, mem_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(mem_embedding)
                    + 1e-8
                )
                similarities.append((similarity, mem))

            # Sort by similarity (highest first)
            similarities.sort(reverse=True, key=lambda x: x[0])

            # Return top matches
            top_matches = [mem for _, mem in similarities[:limit]]
            logger.info(
                f"Found {len(top_matches)} semantically similar memories for query: {query}"
            )
            return top_matches
        except Exception as e:
            logger.error(f"Error searching memories for user {user_id}: {e}")
            # Fallback to simple text search
            return self._simple_text_search(user_id, query, limit)

    def _simple_text_search(
        self, user_id: str, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Fallback simple text search"""
        user_memories = self.memories.get(user_id, [])
        query_lower = query.lower()
        matching = [
            mem
            for mem in user_memories
            if query_lower in mem.get("memory", "").lower()
            or query_lower in mem.get("content", "").lower()
        ]
        return matching[-limit:] if matching else []

    def delete_user_memories(self, user_id: str) -> None:
        """
        Delete all memories for a user

        Args:
            user_id: User identifier
        """
        try:
            if user_id in self.memories:
                del self.memories[user_id]
                self._save_memories()
            logger.info(f"Deleted all memories for user {user_id}")
        except Exception as e:
            logger.error(f"Error deleting memories for user {user_id}: {e}")
