"""Native request/response protocol for a single observe-and-compose model call."""
from __future__ import annotations

from astrbot.core.agent.message import TextPart, ImageURLPart
from astrbot.core.agent.tool import ToolSet

from .context_adapter import reply_metadata
from .context_weaver import CURRENT_TAG, iso_time, pack, unpack
from .history_storage import HistoryStorage
from .message_utils import MessageUtils
from .observation import ObservationSettings, parse_decision
from .recent_images import message_parts_with_visuals

OBSERVATION_POLICY = """
[本轮是群聊观察，不是有人向你提问]
本轮最后一条群消息只标记观察批次的锚点，不指定回复对象。最新一句不是默认提问。
contextweave_observation 是候选群消息，不因问号就视为向你提问；留意实际交流对象和前文。
能顺着当前交流自然接话时可以参与，简单赞同、情绪回应或接梗也可以，不要求额外知识或独立见解。
已有别人回应不代表你必须沉默；接不上、只会重复或会打断别人的交流时可以沉默，不必凑话。
如果接话，只选择候选列表中的一个 message_id 标明关联位置，正文语气和形式沿用原有人格。
这是一次完整判断与正文生成，不调用工具、不搜索、不执行操作，不再要求另一轮判断。
仅输出一个JSON对象，不解释判断过程，不加Markdown，不输出角色动作。
沉默格式：{"action":"silent"}
接话格式：{"action":"reply","target_message_id":"候选消息ID","text":"要发送的短回复"}
text最多三行并遵守本轮字符上限，只有text会发到群里；JSON结构不属于聊天正文。
""".strip()


async def prepare_observation(event, req, config, coordinator):
    token = event.get_extra('_contextweave_observation')
    if not token:
        return
    if not coordinator.valid(token):
        event.stop_event()
        return
    cfg = ObservationSettings.from_config(config)
    wanted = set(token['candidate_ids'])
    history = HistoryStorage.get_history(event.get_platform_name(), False, event.get_group_id())
    candidates = []
    for message in history:
        mid = str(getattr(message, 'message_id', '') or '')
        if mid not in wanted:
            continue
        sender = getattr(message, 'sender', None)
        if str(getattr(sender, 'user_id', '')) == str(event.get_self_id()):
            continue
        parts, visual_indexes = message_parts_with_visuals(event, message)
        text = await MessageUtils.outline_message_list(parts,
                                                      umo=event.unified_msg_origin, cached_images_only=True)
        candidates.append({'message_id': mid, 'sender_id': str(getattr(sender, 'user_id', '')),
                           'sender_name': str(getattr(sender, 'nickname', '')),
                           'sent_at': iso_time(getattr(message, 'timestamp', None)),
                           'reply_to': reply_metadata(message), 'text': text[:1200]})
        if visual_indexes:
            candidates[-1]['visual_component_indexes'] = visual_indexes
    if not candidates:
        event.stop_event()
        return
    token['allowed_targets'] = {m['message_id']: m['sender_id'] for m in candidates}
    # Avoid presenting a group's question as the direct user prompt.
    req.prompt = '观察这一小段群聊，判断是否有值得你参与的内容，按观察协议输出沉默或目标与正文'
    req.image_urls = []
    # Observation may see the configured recent images, while never inheriting
    # arbitrary image input from another request builder.
    allowed_refs = set(event.get_extra('_contextweave_recent_image_refs', []))
    req.extra_user_content_parts = [part for part in req.extra_user_content_parts
        if not ((isinstance(part, ImageURLPart) and part.image_url.url not in allowed_refs)
                or (isinstance(part, dict) and part.get('type') == 'image_url'
                    and part.get('image_url', {}).get('url') not in allowed_refs))]
    req.audio_urls = []
    # Pure judge-and-compose pass: no side-effect tools or tool-triggered sends.
    req.func_tool = ToolSet()
    req.system_prompt += '\n\n' + OBSERVATION_POLICY + f'\n本轮text最多{cfg.max_reply_chars}个字符'
    for part in req.extra_user_content_parts:
        marker = unpack(CURRENT_TAG, getattr(part, 'text', None))
        if isinstance(marker, dict):
            marker['mode'] = 'observation'
            marker['meaning'] = '观察锚点，不是向你提问，也不是默认回复对象'
            part.text = pack(CURRENT_TAG, marker)
    req.extra_user_content_parts.append(TextPart(text=pack('contextweave_observation',
                                                         {'messages': candidates, 'max_reply_chars': cfg.max_reply_chars})))


def process_observation_response(event, response, config, coordinator):
    token = event.get_extra('_contextweave_observation')
    if not token:
        return False
    event.set_extra('_llm_reasoning_content', '')
    response.reasoning_content = ''
    event.set_extra('_contextweave_observation_response', response)
    decision = None
    if coordinator.valid(token) and response.role == 'assistant':
        decision = parse_decision(response.completion_text, token.get('allowed_targets', {}),
                                  max_chars=ObservationSettings.from_config(config).max_reply_chars)
    if decision is None:
        response.completion_text = '<NO_RESPONSE>'
        event.set_extra('_contextweave_observation_reply', None)
    else:
        target, text = decision
        response.completion_text = text
        event.set_extra('_contextweave_observation_reply', target)
    return True
