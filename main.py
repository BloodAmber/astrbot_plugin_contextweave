from astrbot.api.all import *
from astrbot.api.event import filter
from astrbot.api.provider import ProviderRequest, LLMResponse
from .utils import *
from .utils.context_adapter import adapt_request
from .utils.observation import ObservationCoordinator, ObservationSettings
from .utils.observation_adapter import prepare_observation, process_observation_response
from .utils.animation_context import annotate_animations
from .utils.engagement import is_group_invitation
from .utils.image_focus import label_image_roles
from .utils.reply_links import install_receipt_capture, restore_receipt_capture, ReplyLinks
from .utils.text_filter import filter_thinking_stream, ThinkingPrefixFilter
from .utils.chronological_context import arrange_chronological_context
from .utils.request_lifecycle import ReplyTasks, refresh_owned_request
from astrbot.core.agent.message import TextPart
import asyncio
import copy
import time
from sys import maxsize

@register(
    "spectrecore",
    "23q3",
    "ContextWeave 群聊语境：统一会话、消息归属与主动接话",
    "3.4.3",
    "https://github.com/23q3/astrbot_plugin_SpectreCore"
)
class ContextWeave(Star):
    """
    使大模型更好的主动回复群聊中的消息，带来生动和沉浸的群聊对话体验
    """
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # Remove the retired option from existing installations.
        if 'block_sl_messages' in self.config:
            self.config.pop('block_sl_messages')
            self.config.save_config()
        self.observer = ObservationCoordinator()
        self._observation_tasks = set()
        self._reply_tasks = ReplyTasks()
        # 初始化各个工具类
        HistoryStorage.init(config)
        ImageCaptionUtils.init(context, config)

    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群消息喵"""
        try:
            # 保存用户消息到历史记录并尝试回复
            async for result in self._process_message(event):
                yield result
        except Exception as e:
            logger.error(f"处理群消息时发生错误: {e}")

    @event_message_type(EventMessageType.PRIVATE_MESSAGE)
    async def on_private_message(self, event: AstrMessageEvent):
        """处理私聊消息喵"""
        try:
            # 保存用户消息到历史记录并尝试回复
            async for result in self._process_message(event):
                yield result
        except Exception as e:
            logger.error(f"处理私聊消息时发生错误: {e}")
            
    async def _process_message(self, event: AstrMessageEvent):
        """处理消息的通用逻辑：保存历史记录并尝试回复"""
        if self.config.get('filter_thinking',True):
            event.set_extra('enable_reasoning',False)
            event.set_extra('_llm_reasoning_content','')
        # 过滤空消息(napcat会发送私聊对方正在输入的状态，导致astrbot识别为空消息)
        message_outline = event.get_message_outline()
        invitation = is_group_invitation(event)
        if (not message_outline or message_outline.strip() == "") and not invitation:
            logger.debug("收到空消息，忽略处理")
            return

        command = event.get_message_str().strip().lstrip('/').split(' ', 1)[0].lower()
        busy_on_arrival = self.reply_tasks.busy(event.unified_msg_origin)
        method = self.config.get('model_frequency', {}).get('method', '概率回复')
        ticket = None
        eligible = False
        if (not event.is_private_chat() and str(event.get_sender_id()) != str(event.get_self_id())
                and HistoryStorage.is_chat_enabled(event)):
            if method == '观察回复' or event.unified_msg_origin in self.observer.states:
                eligible = (method == '观察回复' and not event.is_at_or_wake_command
                            and command not in {'contextweave', 'cw', 'spectrecore', 'sc'}
                            and self.config.get('_temp_mute', {}).get('until', 0) <= time.time()
                            and not ReplyDecision._check_blacklist_keywords(
                                event, self.config.get('model_frequency', {}).get('blacklist_keywords', [])))
                ticket = self.observer.note(event.unified_msg_origin,
                                            str(getattr(event.message_obj, 'message_id', '') or ''),
                                            eligible=eligible)

        # Record arrival above BEFORE image persistence can await downloads.
        # 保存用户消息到历史记录
        await HistoryStorage.process_and_save_user_message(event)

        # Bare @ means an invitation into the discussion, not an empty question.
        # Supply a request for the existing native wake path, rather than yielding
        # another request here (which could trigger twice after a silent response).
        if invitation and HistoryStorage.is_chat_enabled(event):
            event.set_extra('_contextweave_invitation', True)
            event.set_extra('_contextweave_trigger', 'bare_at')
            if event.get_extra('provider_request') is None:
                request = await LLMUtils.call_llm(event, self.config, self.context)
                event.set_extra('_contextweave_proactive', False)
                event.set_extra('provider_request', request)
            return

        # Direct wakes/private messages belong to AstrBot's native path. Never
        # issue an additional probabilistic request for the same incoming event.
        if event.is_at_or_wake_command or str(event.get_sender_id()) == str(event.get_self_id()):
            return
        if command in {'contextweave', 'cw', 'spectrecore', 'sc'}:
            return

        # Still save messages while replying, but do not turn them into a backlog
        # of unsolicited answers. Explicit wakes above retain the native queue.
        if busy_on_arrival or self.reply_tasks.busy(event.unified_msg_origin):
            logger.debug('ContextWeave: skipped ordinary reply while this conversation is busy')
            return

        if method == '观察回复' and not event.is_private_chat():
            if not eligible or ticket is None:
                return
            task = asyncio.current_task()
            self._observation_tasks.add(task)
            token = None
            progress = False
            try:
                token = await self.observer.wait_for_turn(event.unified_msg_origin, ticket,
                                                        ObservationSettings.from_config(self.config))
                if (not token or self.config.get('model_frequency', {}).get('method') != '观察回复'
                        or not HistoryStorage.is_chat_enabled(event)
                        or self.config.get('_temp_mute', {}).get('until', 0) > time.time()):
                    return
                if (self.reply_tasks.busy(event.unified_msg_origin)
                        or LLMUtils.is_llm_in_progress(event.get_platform_name(), False, event.get_group_id())):
                    return
                event.set_extra('_contextweave_observation', token)
                # The protocol must be fully validated before any text is sent.
                event.set_extra('enable_streaming', False)
                event.set_extra('enable_reasoning', False)
                LLMUtils.set_llm_in_progress(event.get_platform_name(), False, event.get_group_id(), True)
                progress = True
                logger.info('ContextWeave observation: batch=%s', len(token['candidate_ids']))
                # Preserve the original recorded message, but keep native quote
                # parsing from issuing extra caption calls during observation.
                original_message = event.message_obj
                anchor = copy.copy(original_message)
                anchor.message = [Plain('观察群聊')]
                anchor.message_str = '观察群聊'
                event.message_obj = anchor
                try:
                    yield await LLMUtils.call_llm(event, self.config, self.context)
                finally:
                    event.message_obj = original_message
            finally:
                if progress:
                    LLMUtils.set_llm_in_progress(event.get_platform_name(), False, event.get_group_id(), False)
                self.observer.finish(token)
                self._observation_tasks.discard(task)
            return

        # 尝试自动回复
        if ReplyDecision.should_reply(event, self.config):
            async for result in ReplyDecision.process_and_reply(event, self.config, self.context):
                yield result

    @property
    def reply_tasks(self):
        if not hasattr(self, '_reply_tasks'):
            self._reply_tasks = ReplyTasks()
        return self._reply_tasks

    @filter.on_waiting_llm_request(priority=maxsize - 10)
    async def coordinate_reply(self, event: AstrMessageEvent):
        if not HistoryStorage.is_chat_enabled(event):
            return
        ordinary = event.get_extra('_contextweave_proactive') and not event.is_at_or_wake_command
        if ordinary and self.reply_tasks.busy(event.unified_msg_origin):
            event.stop_event()
            logger.info('ContextWeave: skipped competing proactive request')
            return
        self.reply_tasks.claim(event.unified_msg_origin)

    @filter.on_llm_request(priority=maxsize - 10)
    async def refresh_conversation(self, event: AstrMessageEvent, req: ProviderRequest):
        # Runs under AstrBot's native lock, before any context injection.
        if HistoryStorage.is_chat_enabled(event):
            await refresh_owned_request(event, req, self.context)

    @filter.on_llm_request(priority=-maxsize + 100)
    async def weave_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """Run after Flow and apply the same structure to both reply paths."""
        if self.config.get('filter_thinking',True):
            event.set_extra('enable_reasoning',False)
            event.set_extra('_llm_reasoning_content','')
        await adapt_request(event, req, self.config)
        await prepare_observation(event, req, self.config, self.observer)
        await label_image_roles(event, req, self.context.get_config(umo=event.unified_msg_origin))
        await annotate_animations(event, req, self.context.get_config(umo=event.unified_msg_origin))
        await arrange_chronological_context(event, req, self.context.get_config(umo=event.unified_msg_origin))
        if event.get_extra('_contextweave_adapted'):
            install_receipt_capture(event, req)
        stats = event.get_extra('_contextweave_stats')
        if stats:
            logger.info('ContextWeave: mode=%s messages=%s omitted=%s history_images_removed=%s',
                        stats['mode'], stats['messages'], stats['omitted'], stats['removed_history_images'])
        image_stats = event.get_extra('_contextweave_image_stats')
        if image_stats:
            logger.info('ContextWeave images: added=%s sources=%s', image_stats['added'], image_stats['source_count'])
        recent_stats = event.get_extra('_contextweave_recent_image_stats')
        if recent_stats:
            logger.info('ContextWeave recent images: selected=%s added=%s', recent_stats['selected'], recent_stats['added'])
        if event.get_extra('_contextweave_animation_count'):
            logger.info('ContextWeave animation labels: %s', event.get_extra('_contextweave_animation_count'))
        if event.get_extra('_contextweave_image_role_count') or event.get_extra('_contextweave_missing_focus_ids'):
            logger.info('ContextWeave image references: explicit=%s missing=%s labeled=%s',
                        len(event.get_extra('_contextweave_focus_images', [])),
                        len(event.get_extra('_contextweave_missing_focus_ids', [])),
                        event.get_extra('_contextweave_image_role_count', 0))
        if event.get_extra('_contextweave_chronological'):
            logger.info('ContextWeave layout: chronological images=%s',
                        event.get_extra('_contextweave_chronological_image_count', 0))
        if event.get_extra('_contextweave_reordered_turns'):
            logger.info('ContextWeave layout: reordered historical turns=%s',
                        event.get_extra('_contextweave_reordered_turns'))

    @filter.on_llm_response(priority=maxsize - 50)
    async def observation_decision(self, event: AstrMessageEvent, resp: LLMResponse):
        """Parse decision before normal filters; only the validated body survives."""
        if self.config.get('filter_thinking',True):
            event.set_extra('enable_reasoning',False)
            event.set_extra('_llm_reasoning_content','')
            resp.reasoning_content=''
            raw=resp.completion_text
            resp.completion_text=TextFilter.process_model_text(raw,self.config)
            if raw and raw.strip() and not resp.completion_text:
                logger.warning('ContextWeave: suppressed reasoning-only response without a reliable answer boundary')
        if process_observation_response(event, resp, self.config, self.observer):
            logger.info('ContextWeave observation: decision=%s',
                        'reply' if event.get_extra('_contextweave_observation_reply') else 'silent-or-stale')

    @filter.on_decorating_result(priority=maxsize - 60)
    async def guard_observation_send(self, event: AstrMessageEvent):
        token = event.get_extra('_contextweave_observation')
        if not token:
            return
        active = self.config.get('model_frequency', {}).get('method') == '观察回复'
        muted = self.config.get('_temp_mute', {}).get('until', 0) > time.time()
        if not active or muted or not self.observer.valid(token):
            resp = event.get_extra('_contextweave_observation_response')
            if resp is not None:
                resp.completion_text = '<NO_RESPONSE>'
            stored = event.get_extra('_contextweave_observation_history_message')
            if stored is not None:
                stored.content = [TextPart(text='<NO_RESPONSE>')]
            event.clear_result()
            return
        target = event.get_extra('_contextweave_observation_reply')
        result = event.get_result()
        if target and result and target != str(getattr(event.message_obj, 'message_id', '')):
            # An explicit quote disambiguates an earlier target. AstrBot will
            # not add its default latest-message quote to a non-Plain chain.
            result.chain.insert(0, Reply(id=target, sender_id=token.get('allowed_targets', {}).get(target, '')))

    @filter.on_agent_done(priority=-maxsize + 100)
    async def observation_history(self, event: AstrMessageEvent, run_context, resp: LLMResponse):
        """Native runner appends raw output before response hooks; sanitize it too."""
        messages = getattr(run_context, 'messages', [])
        if not event.get_extra('_contextweave_observation'):
            if self.config.get('filter_thinking',True) and messages and messages[-1].role=='assistant':
                plains=[part for part in messages[-1].content or [] if isinstance(part,TextPart)]
                state=ThinkingPrefixFilter()
                for index,part in enumerate(plains):
                    part.text=state.feed(part.text,final=index==len(plains)-1)
            return
        if messages and messages[-1].role == 'assistant':
            messages[-1].content = [TextPart(text=resp.completion_text)]
            event.set_extra('_contextweave_observation_history_message', messages[-1])

    @filter.on_decorating_result(priority=-maxsize + 100)
    async def native_conversation_boundary(self, event: AstrMessageEvent):
        """An explicitly successful native reset/new also resets our replay log."""
        if not event.get_extra('_clean_ltm_session', False) or not HistoryStorage.is_chat_enabled(event):
            return
        command = event.get_message_str().strip().lstrip('/').split(' ', 1)[0]
        if command in {'reset', 'new'}:
            private = event.is_private_chat()
            chat_id = event.get_sender_id() if private else event.get_group_id()
            HistoryStorage.clear_history(event.get_platform_name(), private, chat_id)
            ReplyLinks.clear(event)

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """处理bot发送的消息喵"""
        try:           
            # 保存机器人消息
            if event._result and hasattr(event._result, "chain"):
                if HistoryStorage.is_chat_enabled(event) and not event.get_extra('_contextweave_sent_marked', False):
                    self.observer.mark_sent(event.unified_msg_origin, event.get_extra('_contextweave_observation'))
                    event.set_extra('_contextweave_sent_marked', True)
                # 检查是否为重置历史记录的提示消息，如果是则不保存
                message_text = "".join([i.text for i in event._result.chain if hasattr(i, "text")])
                if "已成功重置" in message_text and "的历史记录喵~" in message_text:
                    return
                
                await HistoryStorage.save_bot_message_from_chain(event._result.chain, event)
                logger.debug(f"已保存bot回复消息到历史记录")
                
        except Exception as e:
            logger.error(f"处理bot发送的消息时发生错误: {e}")
        finally:
            restore_receipt_capture(event)

    async def terminate(self):
        self.observer.close()
        pending = [task for task in self._observation_tasks if task is not asyncio.current_task() and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._observation_tasks.clear()

    from astrbot.api.provider import LLMResponse
    @filter.on_llm_response(priority=114514)
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """处理大模型回复喵"""
        logger.debug(f"收到大模型回复喵: {resp}")
        try:
            if resp.role != "assistant":
                return
            if self.config.get('filter_thinking',True):
                event.set_extra('_llm_reasoning_content','')
                event.set_extra('enable_reasoning',False)
                resp.reasoning_content=''
            # 只进行文本过滤，不处理读空气逻辑
            resp.completion_text = TextFilter.process_model_text(resp.completion_text, self.config)
        except Exception as e:
            logger.error(f"处理大模型回复时发生错误: {e}")

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """在消息发送前处理读空气功能喵"""
        try:
            result = event.get_result()
            if result is None:
                return
            if self.config.get('filter_thinking',True):
                event.set_extra('enable_reasoning',False)
                event.set_extra('_llm_reasoning_content','')
            if (self.config.get('filter_thinking',True) and getattr(result,'async_stream',None) is not None
                    and (result.is_llm_result() or event.get_extra('_contextweave_adapted'))):
                if not event.get_extra('_contextweave_thinking_stream_filtered'):
                    result.async_stream=filter_thinking_stream(result.async_stream)
                    event.set_extra('_contextweave_thinking_stream_filtered',True)
            if not result.chain:
                return

            # 检查是否为LLM结果且包含<NO_RESPONSE>标记
            if result.is_llm_result():
                if self.config.get('filter_thinking',True):
                    state=ThinkingPrefixFilter()
                    plains=[comp for comp in result.chain if isinstance(comp,Plain)]
                    for index,comp in enumerate(plains):
                        comp.text=state.feed(comp.text,final=index==len(plains)-1)
                    if plains and not any(comp.text.strip() for comp in plains) and all(
                            isinstance(comp,(Plain,Reply,At)) for comp in result.chain):
                        event.clear_result()
                        return
                # 获取消息文本内容
                message_text = ""
                for comp in result.chain:
                    if hasattr(comp, 'text'):
                        message_text += comp.text

                # 如果包含<NO_RESPONSE>标记，清空事件结果以阻止消息发送
                if "<NO_RESPONSE>" in message_text:
                    logger.debug(f"检测到读空气标记，阻止消息发送。事件结果: {event.get_result()}")
                    event.clear_result()
                    logger.debug(f"已清空事件结果: {event.get_result()}")

        except Exception as e:
            logger.error(f"处理消息发送前事件时发生错误: {e}")

    @filter.command_group("contextweave",alias={'cw', 'spectrecore', 'sc'})
    def spectrecore(self):
        """插件的前缀喵 可以用sc代替喵"""
        pass

    @spectrecore.command("help", alias=['帮助', 'helpme'])
    async def help(self, event: AstrMessageEvent):
        """查看插件的帮助喵"""
        help_text = (
            "ContextWeave（群聊语境）帮助\n"
            "使用 /cw 或 /contextweave；兼容 /sc 和 /spectrecore\n"
            "可在插件设置中切换概率回复与观察回复\n"
            "使用reset指令重置当前聊天记录 如/sc reset\n"
            "   你也可以重置指定群聊天记录 如/sc reset 群号\n"
            "使用history指令可以查看最近聊天记录 如/sc history\n"
            "使用mute/闭嘴指令临时禁用自动回复 如/sc mute 5 或 /sc 闭嘴 10\n"
            "使用unmute/说话指令解除禁用 如/sc unmute 或 /sc 说话"
        )
        platform_name = event.get_platform_name()
        if platform_name in ("qq_official", "qq_official_webhook"):
            help_text += "\n强烈建议前往Github阅读README文档"
        else:
            help_text += "\n↓强烈建议您阅读Github中的README文档↓\nhttps://github.com/23q3/astrbot_plugin_SpectreCore"
        yield event.plain_result(help_text)
    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("history")
    async def history(self, event: AstrMessageEvent, count: int = 10):
        """查看最近的聊天记录喵，默认10条喵，示例/sc history 5"""
        try:
            # 获取平台名称
            platform_name = event.get_platform_name()
            
            # 判断是群聊还是私聊
            is_private = event.is_private_chat()
            
            # 获取聊天ID
            chat_id = event.get_group_id() if not is_private else event.get_sender_id()
            
            if not chat_id:
                yield event.plain_result("获取聊天ID失败喵，无法显示历史记录")
                return
                
            # 获取历史记录
            history = HistoryStorage.get_history(platform_name, is_private, chat_id)
            
            if not history:
                yield event.plain_result("暂无聊天记录喵")
                return
                
            # 限制记录数量
            if count > 20:
                count = 20  # 限制最大显示数量为20条
            
            # 只取最近的记录
            recent_history = history[-count:] if len(history) > count else history
            
            # 格式化历史记录
            formatted_history = await MessageUtils.format_history_for_llm(recent_history, umo=event.unified_msg_origin)
            
            # 添加标题
            chat_type = "私聊" if is_private else f"群聊({chat_id})"
            title = f"最近{len(recent_history)}条{chat_type}聊天记录喵：\n\n"
            
            # 发送结果
            full_content = title + formatted_history
            
            # 如果内容过长，转为图片发送
            if len(full_content) > 3000:
                image_url = await self.text_to_image(full_content)
                yield event.image_result(image_url)
            else:
                yield event.plain_result(full_content)
            
        except Exception as e:
            logger.error(f"获取历史记录时发生错误: {e}")
            yield event.plain_result(f"获取历史记录失败喵：{str(e)}")
    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("reset")
    async def reset(self, event: AstrMessageEvent, group_id: str | None = None):
        """重置历史记录喵，不带参数重置当前聊天记录，带群号则重置指定群聊记录 如/sc reset 123456"""
        try:
            # 获取平台名称
            platform_name = event.get_platform_name()
            
            # 判断是否提供了群号
            if group_id:
                # 重置指定群聊的历史记录
                is_private = False
                chat_id = group_id
                chat_type = f"群聊({group_id})"
            else:
                # 判断是群聊还是私聊
                is_private = event.is_private_chat()
                # 获取聊天ID
                chat_id = event.get_group_id() if not is_private else event.get_sender_id()
                chat_type = "私聊" if is_private else f"群聊({chat_id})"
                
                if not chat_id:
                    yield event.plain_result("获取聊天ID失败喵，无法重置历史记录")
                    return
            
            # 先检查是否存在历史记录
            history = HistoryStorage.get_history(platform_name, is_private, chat_id)
            if not history:
                yield event.plain_result(f"{chat_type}没有历史记录喵，无需重置")
                return
                
            # 重置历史记录
            success = HistoryStorage.clear_history(platform_name, is_private, chat_id)
            
            if success:
                yield event.plain_result(f"已成功重置{chat_type}的历史记录喵~")
            else:
                yield event.plain_result(f"重置{chat_type}的历史记录失败喵，可能发生错误")
                
        except Exception as e:
            logger.error(f"重置历史记录时发生错误: {e}")
            yield event.plain_result(f"重置历史记录失败喵：{str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("mute", alias=['闭嘴', 'shutup'])
    async def mute(self, event: AstrMessageEvent, minutes: int = 5):
        """临时禁用自动回复，默认5分钟喵，示例/sc mute 10 或 /sc 闭嘴 3"""
        try:
            # 计算禁用结束时间
            mute_until = time.time() + (minutes * 60)
            
            # 保存到配置中
            if "_temp_mute" not in self.config:
                self.config["_temp_mute"] = {}
            
            self.config["_temp_mute"]["until"] = mute_until
            self.config["_temp_mute"]["by"] = event.get_sender_id()
            self.config["_temp_mute"]["at"] = time.time()
            
            # 保存配置
            self.config.save_config()
            
            yield event.plain_result(f"好的喵，我会安静 {minutes} 分钟的~")
            
        except Exception as e:
            logger.error(f"执行闭嘴指令时发生错误: {e}")
            yield event.plain_result(f"执行闭嘴指令失败喵：{str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("unmute", alias=['说话', 'speak'])
    async def unmute(self, event: AstrMessageEvent):
        """解除禁用自动回复喵"""
        try:
            # 检查是否处于静默状态
            mute_info = self.config.get("_temp_mute", {})
            if not mute_info or mute_info.get("until", 0) <= time.time():
                yield event.plain_result("我现在本来就在正常说话喵~")
                return
            
            # 解除静默
            if "_temp_mute" in self.config:
                del self.config["_temp_mute"]
                self.config.save_config()
            
            yield event.plain_result("好耶！我又可以说话了喵~")
            
        except Exception as e:
            logger.error(f"解除闭嘴时发生错误: {e}")
            yield event.plain_result(f"解除闭嘴失败喵：{str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("callllm")
    async def callllm(self, event: AstrMessageEvent):
        """触发一次大模型回复 这是用来开发中测试的喵"""
        try:
            # 调用LLM工具类的方法构建并返回请求
            yield await LLMUtils.call_llm(event, self.config, self.context)
        except Exception as e:
            logger.error(f"调用大模型时发生错误: {e}")
            yield event.plain_result(f"触发大模型回复失败喵：{str(e)}")
