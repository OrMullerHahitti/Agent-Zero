"""Communication protocols and message routing."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
import asyncio
import logging
from datetime import datetime
import json

from ..core.message import Message, MessageType
from ..core.registry import AgentRegistry


class MessageBus(ABC):
    """Abstract message bus for agent communication."""
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """Send a message to the target recipient."""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: Message) -> bool:
        """Broadcast a message to all agents."""
        pass
    
    @abstractmethod
    async def subscribe(self, agent_id: str, handler: Callable[[Message], None]) -> bool:
        """Subscribe an agent to receive messages."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, agent_id: str) -> bool:
        """Unsubscribe an agent from receiving messages."""
        pass


class InMemoryMessageBus(MessageBus):
    """In-memory message bus for development and testing."""
    
    def __init__(self):
        self._subscribers: Dict[str, Callable[[Message], None]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger("message_bus")
    
    async def start(self) -> None:
        """Start the message bus."""
        self._running = True
        self._worker_task = asyncio.create_task(self._message_worker())
        self.logger.info("In-memory message bus started")
    
    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self.logger.info("In-memory message bus stopped")
    
    async def send_message(self, message: Message) -> bool:
        """Send a message to the target recipient."""
        try:
            await self._message_queue.put(message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def broadcast_message(self, message: Message) -> bool:
        """Broadcast a message to all agents."""
        message.recipient_id = None  # Mark as broadcast
        return await self.send_message(message)
    
    async def subscribe(self, agent_id: str, handler: Callable[[Message], None]) -> bool:
        """Subscribe an agent to receive messages."""
        self._subscribers[agent_id] = handler
        self.logger.info(f"Agent {agent_id} subscribed to message bus")
        return True
    
    async def unsubscribe(self, agent_id: str) -> bool:
        """Unsubscribe an agent from receiving messages."""
        if agent_id in self._subscribers:
            del self._subscribers[agent_id]
            self.logger.info(f"Agent {agent_id} unsubscribed from message bus")
            return True
        return False
    
    async def _message_worker(self) -> None:
        """Worker that processes messages from the queue."""
        while self._running:
            try:
                # Wait for message with timeout to allow clean shutdown
                message = await asyncio.wait_for(
                    self._message_queue.get(), timeout=1.0
                )
                await self._deliver_message(message)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message worker: {e}")
    
    async def _deliver_message(self, message: Message) -> None:
        """Deliver a message to the appropriate recipients."""
        if message.recipient_id is None:
            # Broadcast to all subscribers
            for agent_id, handler in self._subscribers.items():
                if agent_id != message.sender_id:  # Don't send to sender
                    await self._deliver_to_handler(message, handler, agent_id)
        else:
            # Send to specific recipient
            handler = self._subscribers.get(message.recipient_id)
            if handler:
                await self._deliver_to_handler(message, handler, message.recipient_id)
            else:
                self.logger.warning(f"No handler found for agent: {message.recipient_id}")
    
    async def _deliver_to_handler(self, message: Message, handler: Callable, agent_id: str) -> None:
        """Deliver message to a specific handler."""
        try:
            # Run handler in a separate task to avoid blocking
            asyncio.create_task(handler(message))
        except Exception as e:
            self.logger.error(f"Error delivering message to {agent_id}: {e}")


class MessageRouter:
    """Routes messages between agents and handles request-response patterns."""
    
    def __init__(self, message_bus: MessageBus, registry: AgentRegistry):
        self.message_bus = message_bus
        self.registry = registry
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._response_timeout = 30.0  # seconds
        self.logger = logging.getLogger("message_router")
    
    async def send_request(
        self, 
        sender_id: str, 
        recipient_id: str, 
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Send a request and wait for response."""
        
        message = Message(
            type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload
        )
        
        # Create future for response
        response_future = asyncio.Future()
        self._pending_requests[message.id] = response_future
        
        try:
            # Send the message
            success = await self.message_bus.send_message(message)
            if not success:
                return None
            
            # Wait for response
            timeout_val = timeout or self._response_timeout
            response = await asyncio.wait_for(response_future, timeout=timeout_val)
            return response
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Request {message.id} timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error in send_request: {e}")
            return None
        finally:
            # Clean up
            if message.id in self._pending_requests:
                del self._pending_requests[message.id]
    
    async def send_response(
        self, 
        original_message: Message, 
        response_type: MessageType,
        payload: Dict[str, Any]
    ) -> bool:
        """Send a response to a previous request."""
        
        response = Message(
            type=response_type,
            sender_id=original_message.recipient_id,
            recipient_id=original_message.sender_id,
            correlation_id=original_message.id,
            payload=payload
        )
        
        return await self.message_bus.send_message(response)
    
    async def handle_response(self, message: Message) -> None:
        """Handle incoming response messages."""
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self._pending_requests:
            future = self._pending_requests[correlation_id]
            if not future.done():
                future.set_result(message)
    
    async def broadcast_discovery(self, sender_id: str) -> List[Message]:
        """Broadcast agent discovery and collect responses."""
        
        discovery_message = Message(
            type=MessageType.AGENT_DISCOVERY,
            sender_id=sender_id,
            payload={"requesting_capabilities": True}
        )
        
        # Send broadcast
        await self.message_bus.broadcast_message(discovery_message)
        
        # Collect responses for a short time
        responses = []
        try:
            await asyncio.sleep(2.0)  # Wait for responses
            # Note: In a full implementation, we'd collect actual responses
            # For now, return empty list
        except Exception as e:
            self.logger.error(f"Error in broadcast_discovery: {e}")
        
        return responses
    
    async def find_agent_for_problem(self, problem_type: str) -> Optional[str]:
        """Find the best agent for a specific problem type."""
        agents = self.registry.find_agents_for_problem(problem_type)
        
        if not agents:
            return None
        
        # Return the agent with the lowest load factor
        best_agent = min(agents, key=lambda a: a.capabilities.load_factor)
        return best_agent.agent_id