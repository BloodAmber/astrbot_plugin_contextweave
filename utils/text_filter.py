"""Remove reasoning wrappers at the start of assistant text, not quoted examples."""
import re

_TAG = re.compile(r'<\s*(/?)\s*(think|thought)\s*>',re.I)
_TOKENS = ('<think>','</think>','<thought>','</thought>')


def _partial_tag(text):
    compact=re.sub(r'\s+','',text).lower()
    return bool(compact) and any(token.startswith(compact) for token in _TOKENS)


class ThinkingPrefixFilter:
    """Buffer only the opening prefix; hidden reasoning content is discarded."""
    def __init__(self):
        self.buffer=''
        self.stack=[]
        self.removed=False
        self.body=False
        self.brain_header=False
        self.blocked=False

    def feed(self,text,final=False):
        if self.blocked:
            return ''
        if self.body:
            return text
        self.buffer+=text
        while True:
            if self.stack:
                match=_TAG.search(self.buffer)
                if not match:
                    tail=self.buffer[self.buffer.rfind('<'):] if '<' in self.buffer else ''
                    self.buffer=tail if not final and _partial_tag(tail) else ''
                    return ''
                self.buffer=self.buffer[match.end():]
                name=match.group(2).lower()
                if not match.group(1):
                    self.stack.append(name)
                elif self.stack[-1]==name:
                    self.stack.pop()
                continue
            stripped=self.buffer.lstrip('\ufeff \t\r\n')
            if not stripped:
                if final:
                    return '' if self.removed else self.buffer
                return ''
            if not self.removed and stripped.startswith('🧠'):
                remainder=stripped[1:].lstrip('\ufe0f \t')
                if not remainder and not final:
                    return ''
                if remainder.startswith(('\n','\r')):
                    self.brain_header=True
                    self.removed=True
                    self.buffer=remainder.lstrip('\r\n')
                    continue
            match=_TAG.match(stripped)
            if match:
                self.brain_header=False
                self.removed=True
                self.buffer=stripped[match.end():]
                if not match.group(1):
                    self.stack.append(match.group(2).lower())
                continue
            if _partial_tag(stripped):
                return ''
            if self.brain_header:
                # A standalone brain heading with unstructured reasoning has
                # no trustworthy answer boundary. Don't guess a final sentence.
                self.buffer=''
                self.blocked=True
                return ''
            self.body=True
            result=self.buffer.lstrip('\ufeff \t\r\n') if self.removed else self.buffer
            self.buffer=''
            return result


class TextFilter:
    @staticmethod
    def filter_thinking_process(text):
        return ThinkingPrefixFilter().feed(text,final=True) if text else text

    @staticmethod
    def process_model_text(reply_text,config):
        if not reply_text:
            return reply_text
        filtered=TextFilter.filter_thinking_process(reply_text)
        if '<NO_RESPONSE>' in filtered:
            return '<NO_RESPONSE>'
        return filtered if config.get('filter_thinking',True) else reply_text

    @staticmethod
    def clean_assistant_history(contexts,config):
        """Only called on request copies, never rewrites user text or the DB."""
        if not config.get('filter_thinking',True):
            return
        for message in contexts:
            if message.get('role')!='assistant':
                continue
            content=message.get('content')
            if isinstance(content,str):
                message['content']=TextFilter.process_model_text(content,config)
            elif isinstance(content,list):
                state=ThinkingPrefixFilter()
                parts=[p for p in content if isinstance(p,dict) and p.get('type')=='text']
                for index,part in enumerate(parts):
                    part['text']=state.feed(part.get('text',''),final=index==len(parts)-1)
        # Plain leaked reasoning may have no answer at all. Don't pass an empty
        # assistant bubble to providers that reject empty content.
        contexts[:]=[m for m in contexts if not (
            m.get('role')=='assistant' and not m.get('tool_calls') and (
                isinstance(m.get('content'),str) and not m['content'].strip()
                or isinstance(m.get('content'),list) and m['content']
                   and all(isinstance(p,dict) and p.get('type')=='text' and not p.get('text','').strip()
                           for p in m['content']))) ]


async def filter_thinking_stream(stream):
    from astrbot.api.all import Plain, At, Reply
    state=ThinkingPrefixFilter()
    pending=[]
    last=None
    async for chain in stream:
        if getattr(chain,'type',None)=='reasoning':
            continue
        if not hasattr(chain,'chain') or not hasattr(chain,'derive'):
            yield chain
            continue
        last=chain
        visible=False
        for component in chain.chain:
            if isinstance(component,Plain):
                text=state.feed(component.text)
                if text:
                    pending.append(Plain(text))
                    visible=True
            else:
                pending.append(component)
                visible=visible or not isinstance(component,(At,Reply))
        if visible:
            yield chain.derive(pending)
            pending=[]
    tail=state.feed('',final=True)
    if tail and last:
        yield last.derive(pending+[Plain(tail)])
