"""Keep an image question's referent separate from later visual background."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from astrbot.api.all import Image, Reply
from astrbot.core.agent.message import ImageURLPart, TextPart
from astrbot.core.utils import media_utils
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .context_weaver import iso_time, pack
from .image_context import local_path
from .image_references import linked_reference_ids


async def attach_focus_images(event, req, history, explicit_sources):
    if event.get_extra('_contextweave_observation'):
        return [], None
    current = event.message_obj
    now = float(current.timestamp)
    by_id = {}
    for message in history:
        if float(getattr(message, 'timestamp', 0) or 0) <= now:
            by_id[str(message.message_id)] = message
        if str(message.message_id) == str(current.message_id):
            break
    by_id[str(current.message_id)] = current
    ids = list(explicit_sources) if explicit_sources else linked_reference_ids(event, req, history)
    reason = 'explicit_message_reference' if ids else None
    selected = []
    for mid in ids:
        source = by_id.get(str(mid))
        if source is None:
            continue
        sender = getattr(source, 'sender', None)
        for index, part in enumerate(source.message or []):
            if not isinstance(part, Image):
                continue
            path = local_path(part.file) or local_path(part.url)
            if not path:
                continue
            selected.append({'message_id':str(mid), 'component_index':index,
                'image_id':f'{mid}:{index}', 'sender_id':str(getattr(sender, 'user_id', '')),
                'sender_name':str(getattr(sender, 'nickname', '')),
                'sent_at':iso_time(getattr(source, 'timestamp', None)), 'path':path})
    # An embedded quote may have usable originals absent from the retained log.
    for reply in [p for p in current.message if isinstance(p, Reply)]:
        mid = str(reply.id)
        if mid not in explicit_sources or any(s['message_id'] == mid for s in selected):
            continue
        for index, part in enumerate(reply.chain or []):
            if isinstance(part, Image):
                path = local_path(part.file) or local_path(part.url)
                if path:
                    selected.append({'message_id':mid, 'component_index':index,
                        'image_id':f'{mid}:{index}', 'sender_id':str(reply.sender_id),
                        'sender_name':reply.sender_nickname or '', 'sent_at':iso_time(reply.time), 'path':path})
    for record in selected:
        record['relation'] = 'current_attachment' if record['message_id']==str(current.message_id) else 'quoted_reference'
    event.set_extra('_contextweave_historical_images', [])
    seen = {local_path(ref) or ref for ref in req.image_urls}
    for source in selected:
        native_present = source['message_id'] in explicit_sources and bool(req.image_urls)
        if not native_present and source['path'] not in seen:
            req.extra_user_content_parts.append(ImageURLPart(
                image_url=ImageURLPart.ImageURL(url=source['path'])).mark_as_temp())
            seen.add(source['path'])
    event.set_extra('_contextweave_focus_images', selected)
    missing = [str(mid) for mid in ids if not any(s['message_id']==str(mid) for s in selected)
               and not (str(mid) in explicit_sources and req.image_urls)]
    event.set_extra('_contextweave_missing_focus_ids', missing)
    return [{key:value for key,value in s.items() if key != 'path'} for s in selected], reason if selected else None


def _label_images(event, req, settings):
    sources = (event.get_extra('_contextweave_focus_images', []) +
               event.get_extra('_contextweave_historical_images', []) +
               event.get_extra('_contextweave_recent_images', []))
    if not sources:
        return 0
    roles = {(s['message_id'],s['component_index']):s['relation'] for s in
             event.get_extra('_contextweave_focus_images', [])+event.get_extra('_contextweave_historical_images', [])}
    by_path, by_digest = {}, {}
    def remember(mapping,key,record):
        records=mapping.setdefault(key,[])
        if not any(r['image_id']==record['image_id'] for r in records):
            records.append(record)
    options = settings.get('image_compress_options', {})
    options = options if isinstance(options, dict) else {}
    size = media_utils.normalize_model_image_max_size(options.get('max_size'))
    quality = options.get('quality',media_utils.IMAGE_COMPRESS_DEFAULT_QUALITY)
    quality = max(1,min(100,quality)) if isinstance(quality,int) and not isinstance(quality,bool) else media_utils.IMAGE_COMPRESS_DEFAULT_QUALITY
    enabled = settings.get('image_compress_enabled',True) is not False
    for source in sources:
        path = local_path(source['path'])
        if not path:
            continue
        record = {k:v for k,v in source.items() if k != 'path'}
        record['image_id'] = f"{source['message_id']}:{source['component_index']}"
        key = (source['message_id'],source['component_index'])
        record['relation'] = roles.get(key,'recent_background')
        remember(by_path,path,record)
        try:
            data = Path(path).read_bytes()
            remember(by_digest,hashlib.sha256(data).digest(), record)
            if enabled:
                for operation in ['convert','montage']:
                    cache_key = media_utils._image_convert_cache_key(data,
                        f'{media_utils._IMAGE_CONVERT_CACHE_VERSION}|{operation}|s={size}|q={quality}')
                    cached = Path(get_astrbot_temp_path())/media_utils.CONVERT_CACHE_DIR_NAME/(cache_key+'.img')
                    if cached.is_file():
                        remember(by_digest,hashlib.sha256(cached.read_bytes()).digest(), record)
        except OSError:
            continue

    def label(ref):
        path = local_path(ref)
        if not path:
            return None
        records = by_path.get(path)
        if not records:
            try:
                records = by_digest.get(hashlib.sha256(Path(path).read_bytes()).digest())
            except OSError:
                pass
        if not records:
            return None
        if len(records)>1:
            return TextPart(text=pack('contextweave_image',{'relation':'same_visual_multiple_sources',
                'sources':records,'meaning':'这些来源的图像内容相同，转换结果不能唯一确定发送来源，保留各条消息身份，不自动合并其讨论'})).mark_as_temp()
        record=records[0]
        meaning = {
            'current_attachment':'这张图片随当前消息提供',
            'quoted_reference':'这张图片由当前消息的明确引用链关联，来源可追溯到原消息ID',
            'historical_reference':'这张属于较早那条图像问题，仅供理解历史，不能据此认定当前还在讨论它',
            'recent_background':'这张是近期群图片，保留其来源，不预设它就是当前所指',
        }[record['relation']]
        return TextPart(text=pack('contextweave_image', {**record,'meaning':meaning})).mark_as_temp()
    parts, count = [], 0
    inputs = list(req.extra_user_content_parts)
    inputs.extend(ImageURLPart(image_url=ImageURLPart.ImageURL(url=ref)) for ref in req.image_urls)
    for part in inputs:
        ref = part.image_url.url if isinstance(part,ImageURLPart) else (
            part.get('image_url',{}).get('url') if isinstance(part,dict) and part.get('type')=='image_url' else None)
        note = label(ref) if ref else None
        if note:
            parts.append(note)
            count += 1
        parts.append(part)
    req.extra_user_content_parts = parts
    req.image_urls = []
    return count


async def label_image_roles(event, req, global_config):
    if not event.get_extra('_contextweave_adapted') or event.get_extra('_contextweave_image_roles_labeled'):
        return
    count = await asyncio.to_thread(_label_images,event,req,global_config.get('provider_settings',{}))
    event.set_extra('_contextweave_image_roles_labeled', True)
    event.set_extra('_contextweave_image_role_count', count)
