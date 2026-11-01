from astrbot.api.all import *
from typing import Dict, List, Optional, Any
import time
import threading
from .history_storage import HistoryStorage
from .message_utils import MessageUtils
from astrbot.core.provider.entites import ProviderRequest
from .image_context import component_images

class LLMUtils:
    """
    大模型调用工具类
    用于构建提示词和调用记录相关功能
    """
    
    # 使用字典保存每个聊天的大模型调用状态
    # 格式: {"{platform_name}_{chat_type}_{chat_id}": {"last_call_time": timestamp, "in_progress": True/False}}
    _llm_call_status: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()  # 用于线程安全的锁
    
    @staticmethod
    def get_chat_key(platform_name: str, is_private_chat: bool, chat_id: str) -> str:
        """
        获取聊天的唯一标识
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            聊天的唯一标识
        """
        chat_type = "private" if is_private_chat else "group"
        return f"{platform_name}_{chat_type}_{chat_id}"
    
    @staticmethod
    def set_llm_in_progress(platform_name: str, is_private_chat: bool, chat_id: str, in_progress: bool = True) -> None:
        """
        设置大模型调用状态
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            in_progress: 是否正在进行大模型调用
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                LLMUtils._llm_call_status[chat_key] = {}
                
            LLMUtils._llm_call_status[chat_key]["in_progress"] = in_progress
            LLMUtils._llm_call_status[chat_key]["last_call_time"] = time.time()
    
    @staticmethod
    def is_llm_in_progress(platform_name: str, is_private_chat: bool, chat_id: str) -> bool:
        """
        检查指定聊天是否正在进行大模型调用
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            是否正在进行大模型调用
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                return False
                
            return LLMUtils._llm_call_status[chat_key].get("in_progress", False)
    
    @staticmethod
    def get_last_call_time(platform_name: str, is_private_chat: bool, chat_id: str) -> Optional[float]:
        """
        获取指定聊天最后一次大模型调用的时间戳
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            最后一次调用的时间戳，如果从未调用过则返回None
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                return None
                
            return LLMUtils._llm_call_status[chat_key].get("last_call_time")
    
    @staticmethod
    async def call_llm(event: AstrMessageEvent, config: AstrBotConfig, context: Context) -> ProviderRequest:
        """
        构建调用大模型的请求对象

        Args:
            event: 消息对象
            config: 配置对象
            context: Context 对象，用于获取LLM工具管理器

        Returns:
            ProviderRequest 对象
        """
        # Share the native conversation with @ replies. AstrBot injects its persona,
        # history and tool policy; never copy them into a second system transcript.
        manager = context.conversation_manager
        umo = event.unified_msg_origin
        cid = await manager.get_curr_conversation_id(umo)
        conversation = await manager.get_conversation(umo, cid) if cid else None
        if conversation is None:
            cid = await manager.new_conversation(umo, event.get_platform_id())
            conversation = await manager.get_conversation(umo, cid)
        if conversation is None:
            raise RuntimeError("ContextWeave could not resolve the native conversation")

        # The native session lock is acquired later. The early request hook uses
        # this snapshot to refresh only history, preserving the rest of the turn.
        event.set_extra('_contextweave_history_snapshot',
                        {'cid': conversation.cid, 'history': conversation.history})

        # Native quote text handling still runs for plugin requests, but native
        # attachment collection does not. The request hook resolves quote media.
        observing = bool(event.get_extra('_contextweave_observation'))
        current_parts = [part for part in event.message_obj.message if not isinstance(part, Reply)]
        prompt = ('观察这一小段群聊并判断是否参与' if observing else
                  await MessageUtils.outline_message_list(current_parts, umo=umo))
        if event.get_extra('_contextweave_invitation'):
            prompt = '[本条仅@你，没有附加正文]'
        images = [] if observing else await component_images(event.message_obj.message)
        event.set_extra("_contextweave_proactive", True)
        request = event.request_llm(
            prompt=prompt, conversation=conversation, image_urls=images,
        )
        # The deprecated func_tool_manager argument is ignored by AstrBot 4.28.
        # Native persona/tool filtering remains responsible for tool access.
        if config.get("use_func_tool", False) and not observing:
            request.func_tool = context.get_llm_tool_manager().get_full_tool_set()
        return request

    @staticmethod
    def clear_call_status(platform_name: str, is_private_chat: bool, chat_id: str) -> None:
        """
        清除指定聊天的大模型调用状态
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key in LLMUtils._llm_call_status:
                del LLMUtils._llm_call_status[chat_key] 
