"""Bounded, per-conversation observation scheduling and strict decision parsing."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import time
import uuid


@dataclass(frozen=True)
class ObservationSettings:
    quiet_seconds: float = 3
    max_wait_seconds: float = 8
    min_interval_seconds: float = 20
    cooldown_seconds: float = 45
    max_messages: int = 20
    max_reply_chars: int = 160

    @classmethod
    def from_config(cls, config):
        values = config.get('model_frequency', {}).get('observation', {})
        def number(key, default, low, high):
            try:
                value = float(values.get(key, default))
                if not (float('-inf') < value < float('inf')):
                    return default
                return min(high, max(low, value))
            except (TypeError, ValueError):
                return default
        quiet = number('quiet_seconds', 3, 0.5, 30)
        return cls(quiet, max(quiet, number('max_wait_seconds', 8, 1, 60)),
                   number('min_interval_seconds', 20, 5, 600),
                   number('cooldown_seconds', 45, 0, 1800),
                   int(number('max_messages', 20, 1, 50)),
                   int(number('max_reply_chars', 160, 20, 500)))


@dataclass
class _State:
    generation: int = 0
    batch_started: float | None = None
    last_event: float = 0
    last_check: float = float('-inf')
    last_sent: float = float('-inf')
    pending: list[str] = field(default_factory=list)
    claim_id: str | None = None


class ObservationCoordinator:
    def __init__(self, clock=time.monotonic, sleep=asyncio.sleep):
        self.clock, self.sleep = clock, sleep
        self.epoch = uuid.uuid4().hex
        self.states = {}
        self.closed = False

    def note(self, umo, message_id, *, eligible):
        if self.closed:
            return None
        state = self.states.setdefault(umo, _State())
        now = self.clock()
        state.generation += 1
        state.last_event = now
        if eligible and message_id:
            if not state.pending:
                state.batch_started = now
            if str(message_id) not in state.pending:
                state.pending.append(str(message_id))
                state.pending = state.pending[-50:]
        else:
            # A direct wake, command or exclusion supersedes pending observation.
            state.pending.clear()
            state.batch_started = None
        return state.generation

    def due_at(self, umo, cfg):
        state = self.states[umo]
        start = state.batch_started if state.batch_started is not None else state.last_event
        return max(min(state.last_event + cfg.quiet_seconds, start + cfg.max_wait_seconds),
                   state.last_check + cfg.min_interval_seconds,
                   state.last_sent + cfg.cooldown_seconds)

    async def wait_for_turn(self, umo, generation, cfg):
        while not self.closed:
            state = self.states.get(umo)
            if state is None or state.generation != generation or not state.pending:
                return None
            if state.claim_id:
                await self.sleep(0.25)
                continue
            remaining = self.due_at(umo, cfg) - self.clock()
            if remaining > 0:
                await self.sleep(remaining)
                continue
            state.last_check = self.clock()
            state.claim_id = uuid.uuid4().hex
            token = {'epoch': self.epoch, 'umo': umo, 'generation': generation,
                     'claim_id': state.claim_id, 'candidate_ids': state.pending[-cfg.max_messages:]}
            state.pending = []
            state.batch_started = None
            return token
        return None

    def valid(self, token):
        if self.closed or not token or token.get('epoch') != self.epoch:
            return False
        state = self.states.get(token.get('umo'))
        return bool(state and state.generation == token.get('generation') and
                    state.claim_id == token.get('claim_id'))

    def finish(self, token):
        if not token:
            return
        state = self.states.get(token.get('umo'))
        if state and state.claim_id == token.get('claim_id'):
            state.claim_id = None

    def mark_sent(self, umo, token=None):
        state = self.states.setdefault(umo, _State())
        state.last_sent = self.clock()
        if not token:
            state.generation += 1
            state.pending.clear()
            state.batch_started = None

    def close(self):
        self.closed = True
        self.states.clear()


def parse_decision(raw, candidates, *, max_chars=160):
    """Any malformed, ambiguous or non-candidate output fails closed."""
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if raw == '<NO_RESPONSE>':
        return None
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or value.get('action') != 'reply':
        return None
    target, text = value.get('target_message_id'), value.get('text')
    if not isinstance(target, str) or target not in candidates or not isinstance(text, str):
        return None
    text = text.strip()
    if not text or len(text) > max_chars or len(text.splitlines()) > 3:
        return None
    if any(marker in text for marker in ('<NO_RESPONSE>', '<contextweave_', '"target_message_id"')):
        return None
    return target, text
