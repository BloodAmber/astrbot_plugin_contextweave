"""Render chat records and images at their original message positions."""
from __future__ import annotations

import copy
from datetime import datetime
import math
from pathlib import Path
import re

from astrbot.core.agent.message import TextPart
from astrbot.core.utils import media_utils
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .context_weaver import CURRENT_TAG, DELTA_TAG, unpack
from .engagement import interaction_metadata

HEADER = re.compile(r'^\[(?:群聊|聊天)消息 id=([^\s\]]+)')
IMAGE_POSITION = re.compile(r'^\[图片位置 component=(\d+)(?:；本轮未提供原图)?\]$')
PLACEHOLDERS = ('[历史图片：本轮未重新提供视觉内容；归属本条历史消息，不是当前新图]',
                '[图片：仅有占位，未提供视觉内容]', '[本条附带图片]', '[本条附带图片：本轮未提供原图]')


def header(record):
    refs = record.get('reply_to') or []
    quoted = ','.join(str(r.get('message_id', '')) for r in refs if r.get('message_id'))
    mid = str(record.get('message_id', 'unknown')).replace('\n', ' ')
    name = str(record.get('sender_name', record.get('sender_id', '未知'))).replace('\n', ' ')
    return (f"[聊天消息 id={mid} 时间={record.get('sent_at', 'unknown')} "
            f"发送者={name}({record.get('sender_id', '')})"
            + (f' 引用={quoted}' if quoted else '') + ']')


def part_dict(part):
    return copy.deepcopy(part) if isinstance(part, dict) else part.model_dump_for_context()


def is_internal_label(text):
    return any(unpack(tag, text) is not None for tag in
               ('contextweave_image', 'contextweave_recent_image', 'contextweave_animation'))


def order_time(record):
    try:
        if record.get('timestamp') is not None:
            stamp = float(record['timestamp'])
        else:
            stamp = datetime.fromisoformat(record.get('sent_at', '')).timestamp()
        return stamp if math.isfinite(stamp) else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def sort_chat_turns(timeline, records):
    """Order timestamped chat turns, keeping each response/tool chain intact.

    The triggering user message and following assistant/tool messages travel as
    one unit. Unknown messages and system additions are barriers, so persona
    examples or another plugin's content never acquire an invented timestamp.
    Stable ordering preserves arrival order when timestamps have only seconds.
    """
    result, pending = [], []
    moved = 0

    def flush():
        nonlocal moved
        ordered = sorted(pending, key=lambda block: block[0])
        moved += sum(before is not after for before, after in zip(pending, ordered))
        for _, nodes in ordered:
            result.extend(nodes)
        pending.clear()

    for node in timeline:
        parts = node.get('content', [])
        if isinstance(parts, str):
            parts = [{'type': 'text', 'text': parts}]
        line = next((part.get('text', '') for part in parts or []
                     if isinstance(part, dict) and HEADER.match(part.get('text', ''))), '')
        identity = HEADER.match(line) if node.get('role') == 'user' else None
        stamp = None
        if identity:
            stamp = order_time(records.get(identity.group(1), {}))
            sent_at = re.search(r' 时间=(\S+)', line)
            if stamp is None and sent_at:
                stamp = order_time({'sent_at': sent_at.group(1)})
        if stamp is not None:
            pending.append((stamp, [node]))
        elif node.get('role') in {'assistant', 'tool'} and pending:
            pending[-1][1].append(node)
        else:
            flush()
            result.append(node)
    flush()
    return result, moved


async def arrange_chronological_context(event, req, global_config):
    if not event.get_extra('_contextweave_adapted') or event.get_extra('_contextweave_chronological'):
        return
    event.set_extra('_contextweave_chronological', True)
    current = event.message_obj
    current_id = str(current.message_id)
    # Provenance is used only to put attachments back, never to infer a target.
    pool, loose, other = {}, [], []
    label, animation = None, None
    for source in req.extra_user_content_parts:
        part = part_dict(source)
        text = part.get('text', '')
        note = unpack('contextweave_image', text)
        if note is not None:
            label = note
            continue
        if unpack('contextweave_recent_image', text) is not None:
            continue
        if unpack('contextweave_animation', text) is not None:
            animation = part
            continue
        if part.get('type') == 'image_url':
            item = (part, animation)
            records = label.get('sources', []) if label and label.get('relation') == 'same_visual_multiple_sources' else ([label] if label else [])
            if records:
                for record in records:
                    key = (str(record['message_id']), int(record['component_index']))
                    pool.setdefault(key, (item, record))
            else:
                loose.append(item)
            label, animation = None, None
            continue
        other.append(part)
    # Native current images without local provenance stay with the current event.
    loose.extend(({'type':'image_url', 'image_url':{'url':ref}}, None) for ref in req.image_urls)
    req.image_urls = []
    settings = global_config.get('provider_settings', {})
    enabled = settings.get('image_compress_enabled', True) is not False
    options = settings.get('image_compress_options', {})
    options = options if isinstance(options, dict) else {}
    size = media_utils.normalize_model_image_max_size(options.get('max_size'))
    quality = options.get('quality', media_utils.IMAGE_COMPRESS_DEFAULT_QUALITY)
    if not isinstance(quality, int) or isinstance(quality, bool):
        quality = media_utils.IMAGE_COMPRESS_DEFAULT_QUALITY
    quality = max(1, min(100, quality))
    encoded = {}

    async def visual(item, prepare=True):
        part, note = item
        if not prepare:
            # Current attachments still pass through AstrBot's native image
            # preparation after this hook. Don't transcode them a second time.
            result = [{**note, '_no_save':True}] if note else []
            result.append({**part, '_no_save':True})
            return result
        ref = part['image_url']['url']
        if ref not in encoded:
            try:
                prepared = await media_utils.prepare_model_image(ref, max_size=size,
                    output_dir=Path(get_astrbot_temp_path())/'contextweave_ordered_images',
                    quality=quality, montage_max_size=size) if enabled else ref
                tracker = getattr(event, 'track_temporary_local_file', None)
                if enabled and prepared and tracker:
                    tracker(prepared)
                resolved = await media_utils.resolve_image_ref_to_base64_data(prepared) if prepared else None
                encoded[ref] = resolved.to_data_url() if resolved else None
            except Exception as exc:
                if not media_utils.is_recoverable_image_error(exc):
                    raise
                encoded[ref] = None
        if not encoded[ref]:
            return [{'type':'text', 'text':'[该消息原图本轮读取失败]', '_no_save':True}]
        result = []
        if note:
            result.append({**note, '_no_save':True})
        result.append({'type':'image_url', 'image_url':{'url':encoded[ref]}, '_no_save':True})
        return result

    used = set()

    async def append_visuals(mid, parts):
        keys = sorted((key for key in pool if key[0] == mid and key not in used), key=lambda x:x[1])
        selected_indexes = {key[1] for key in keys}
        # A selected source now has a real image here, not an unavailable marker.
        cleaned = []
        for part in parts:
            if part.get('type') == 'text':
                marker = IMAGE_POSITION.match(part.get('text', ''))
                if marker:
                    index = int(marker.group(1))
                    if index in selected_indexes:
                        continue
                    cleaned.append({**part, 'text':f'[图片位置 component={index}；本轮未提供原图]'})
                    continue
                if keys and part.get('text') in PLACEHOLDERS:
                    continue
                if part.get('text') in ('[本条附带图片]', '[本条附带图片：本轮未提供原图]'):
                    cleaned.append({**part, 'text':'[本条附带图片：本轮未提供原图]'})
                    continue
            cleaned.append(part)
        for key in keys:
            used.add(key)
            # Keep a lightweight position marker in native text history. The
            # actual bytes remain temporary; an unselected position is explicit
            # on later rounds rather than silently disappearing from history.
            cleaned.append({'type':'text','text':f'[图片位置 component={key[1]}]'})
            cleaned.extend(await visual(pool[key][0], prepare=mid != current_id))
        return cleaned

    timeline, coverage_notes = [], []
    for original in req.contexts:
        node = copy.deepcopy(original)
        if node.get('role') != 'user':
            timeline.append(node)
            continue
        raw = node.get('content', '')
        parts = raw if isinstance(raw, list) else [{'type':'text', 'text':raw}]
        # Old incremental JSON blocks become separate ordinary chat messages.
        delta = unpack(DELTA_TAG, raw) if isinstance(raw, str) else None
        if delta is not None:
            for record in delta.get('messages', []):
                mid = str(record.get('message_id', ''))
                body = [{'type':'text', 'text':header(record)}, {'type':'text', 'text':record.get('text', '')}]
                body = await append_visuals(mid, body)
                timeline.append({'role':'user', 'content':body})
            if any(delta.get('omitted', {}).values()):
                coverage_notes.append({'role':'user','content':'[部分较早群消息未包含在本轮上下文中]', '_no_save':True})
            continue
        meta = next((unpack(CURRENT_TAG, p.get('text', '')) for p in parts
                     if isinstance(p, dict) and unpack(CURRENT_TAG, p.get('text', '')) is not None), None)
        match = next((HEADER.match(p.get('text','')) for p in parts
                      if isinstance(p,dict) and HEADER.match(p.get('text',''))), None)
        mid = str(meta['message_id']) if meta else (match.group(1) if match else None)
        cleaned = [p for p in parts if not (isinstance(p, dict) and p.get('type') == 'text'
                   and (unpack(CURRENT_TAG, p.get('text', '')) is not None or is_internal_label(p.get('text', ''))))]
        if meta:
            cleaned.insert(0, {'type':'text', 'text':header(meta)})
        node['content'] = await append_visuals(mid, cleaned) if mid else cleaned
        timeline.append(node)

    # Selected/quoted images outside the retained text window become original
    # source messages too. Sort them together with text backfill below.
    records = {str(r['message_id']):r for r in event.get_extra('_contextweave_timeline_records', [])}
    def source_order(pair):
        record = records.get(pair[0][0], pair[1][1])
        stamp = order_time(record)
        return (stamp if stamp is not None else float('inf'), pair[0][1])

    for key, (item, source) in sorted(pool.items(), key=source_order):
        mid = key[0]
        if key in used or mid == current_id:
            continue
        record = records.get(mid, source)
        body = [{'type':'text','text':header(record)}, {'type':'text','text':record.get('text','')}]
        body = await append_visuals(mid, body)
        node = {'role':'user','content':body}
        timeline.append(node)
    req.contexts, moved = sort_chat_turns(timeline, records)
    req.contexts.extend(coverage_notes)
    event.set_extra('_contextweave_reordered_turns', moved)
    metadata = next((unpack(CURRENT_TAG, p.get('text','')) for p in other
                     if unpack(CURRENT_TAG, p.get('text','')) is not None), None)
    body = []
    if metadata:
        body.append({'type':'text','text':header(metadata)})
    if req.prompt:
        body.append({'type':'text','text':req.prompt})
    body = await append_visuals(current_id, body)
    for item in loose:
        body.append({'type':'text','text':'[本条附带图片]'})
        body.extend(await visual(item, prepare=False))
    # Keep the original attachments next to the current body, before reminders
    # and third-party additions; those additions are preserved without parsing.
    body.extend(p for p in other if unpack(CURRENT_TAG, p.get('text','')) is None)
    kind = interaction_metadata(event)['kind']
    meaning = {
        'direct':'明确与你交流，按正文和实际引用接话',
        'group_invitation':'只有@你，是叫你参与当前讨论，结合前文接话',
        'group_opportunity':'普通群聊接话机会，不代表发言者在问你',
        'observation':'观察批次锚点，按观察协议选择是否参与和回复对象',
    }[kind]
    body.append({'type':'text','text':f'[本轮触发方式：{meaning}]', '_no_save':True})
    req.prompt = ''
    req.extra_user_content_parts = [TextPart(text=p['text']).mark_as_temp() if p.get('_no_save') and p.get('type')=='text'
                                  else p for p in body]
    event.set_extra('_contextweave_chronological_image_count', len(used)+len(loose))
