"""Describe animation inputs without changing AstrBot's image processing."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from PIL import Image as PILImage
from astrbot.api.all import Image, Reply
from astrbot.core.agent.message import ImageURLPart, TextPart
from astrbot.core.utils import media_utils
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .context_weaver import pack
from .image_context import local_path


def remember_image_origins(event, history, source_ids):
    """Save only explicit source candidates from this request's scoped log."""
    sources = []
    current = event.message_obj
    now = float(current.timestamp)
    wanted = set(source_ids)
    for message in history:
        mid = str(message.message_id)
        if mid in wanted and float(getattr(message, 'timestamp', 0) or 0) <= now:
            sources.append((mid, message.message))
        if mid == str(current.message_id):
            break
    sources.append((str(current.message_id), current.message))
    for part in current.message:
        if isinstance(part, Reply):
            sources.append((str(part.id), part.chain or []))
    refs = []
    for mid, parts in sources:
        for part in parts:
            if isinstance(part, Image):
                path = local_path(part.file) or local_path(part.url)
                if path:
                    refs.append({'message_id':mid, 'path':path})
    event.set_extra('_contextweave_animation_origins', refs)


def _inspect(path):
    try:
        with PILImage.open(path) as image:
            count = getattr(image, 'n_frames', 1)
            cover = bool(image.info.get('default_image', False))
            if count <= 1:
                return None
            return {'format':image.format, 'animation_frames':count - int(cover)}
    except (OSError, ValueError, EOFError, SyntaxError):
        return None


def _annotation(info, transformed, source_id=None):
    grid = media_utils.ANIMATED_MONTAGE_GRID
    data = {'source_type':'animation', 'source_format':info['format'],
            'original_frame_count':info['animation_frames']}
    if source_id:
        data['source_message_id'] = source_id
    if transformed:
        data.update({'input_type':'animation_frame_montage', 'grid':f'{grid}×{grid}',
                     'sampled_frames':min(info['animation_frames'], grid*grid),
                     'meaning':f'下面这张图片是同一张动图按帧顺序均匀抽帧后生成的{grid}×{grid}拼图，'
                               '从左到右、从上到下按时间顺序排列，包含首帧和末帧，不是原图自带的九宫格或多张独立图片；'
                               '未填满的格子是留白，不属于动画内容；没有提供逐帧时间、播放速度或声音，不要自行推断'})
    else:
        data.update({'input_type':'original_animation',
                     'meaning':'下面输入的原文件是动图，本轮未启用AstrBot抽帧预处理；'
                               '不要假定已经生成九宫格，不保证接口展示全部帧，不要推断未见动作或声音'})
    return TextPart(text=pack('contextweave_animation', data)).mark_as_temp()


def _annotate(event, req, provider_settings):
    enabled = provider_settings.get('image_compress_enabled', True) is not False
    options = provider_settings.get('image_compress_options', {})
    options = options if isinstance(options, dict) else {}
    size = media_utils.normalize_model_image_max_size(options.get('max_size'))
    quality = options.get('quality')
    if isinstance(quality, bool) or not isinstance(quality, int):
        quality = media_utils.IMAGE_COMPRESS_DEFAULT_QUALITY
    quality = min(100, max(1, quality))
    raw_origins = (event.get_extra('_contextweave_animation_origins', [])
                   + event.get_extra('_contextweave_recent_images', [])
                   + event.get_extra('_contextweave_historical_images', []))
    origins = [{**source, 'path':local_path(source['path'])} for source in raw_origins]
    origins = [source for source in origins if source['path']]
    inspections = {}

    def inspect(path):
        if path not in inspections:
            inspections[path] = _inspect(path)
        return inspections[path]

    # Native current/quote images may already be static prepared files by this
    # hook. Match against the native content-addressed montage cache by bytes,
    # never by ordinal position, .gif extension, or a 3x3-looking picture.
    prepared_sources = {}
    if enabled:
        for source in origins:
            path = source['path']
            info = inspect(path)
            if not info:
                continue
            try:
                key = media_utils._image_convert_cache_key(Path(path).read_bytes(),
                    f'{media_utils._IMAGE_CONVERT_CACHE_VERSION}|montage|s={size}|q={quality}')
                cache = Path(get_astrbot_temp_path())/media_utils.CONVERT_CACHE_DIR_NAME/(key+'.img')
                if cache.is_file():
                    prepared_sources[hashlib.sha256(cache.read_bytes()).digest()] = (info, source['message_id'])
            except OSError:
                continue

    def label(ref):
        path = local_path(ref)
        if not path:
            return None
        info = inspect(path)
        if info:
            mid = next((s['message_id'] for s in origins if s['path'] == path), None)
            return _annotation(info, enabled, mid)
        if enabled and prepared_sources:
            try:
                match = prepared_sources.get(hashlib.sha256(Path(path).read_bytes()).digest())
            except OSError:
                match = None
            if match:
                return _annotation(match[0], True, match[1])
        return None

    parts, count = [], 0
    for part in req.extra_user_content_parts:
        ref = part.image_url.url if isinstance(part, ImageURLPart) else (
            part.get('image_url', {}).get('url') if isinstance(part, dict) and part.get('type') == 'image_url' else None)
        note = label(ref) if ref else None
        if note:
            parts.append(note)
            count += 1
        parts.append(part)
    # ProviderRequest serializes image_urls after extra parts. Move them into
    # equivalent image parts only when needed so each note is adjacent to its image.
    notes = [label(ref) for ref in req.image_urls]
    if any(notes):
        for ref, note in zip(req.image_urls, notes):
            if note:
                parts.append(note)
                count += 1
            parts.append(ImageURLPart(image_url=ImageURLPart.ImageURL(url=ref)))
        req.image_urls = []
    req.extra_user_content_parts = parts
    return count


async def annotate_animations(event, req, global_config):
    if not event.get_extra('_contextweave_adapted') or event.get_extra('_contextweave_animation_annotated'):
        return
    settings = global_config.get('provider_settings', {})
    count = await asyncio.to_thread(_annotate, event, req, settings)
    event.set_extra('_contextweave_animation_annotated', True)
    event.set_extra('_contextweave_animation_count', count)
