#!/usr/bin/env python3

import os
import json
import argparse
import openai
import tiktoken
import datetime
from typing import List

from rich.console import Console
from rich.markdown import Markdown, MarkdownIt
from rich.live import Live

try:
    import rlcompleter
    import readline
except ImportError:
    pass

class Config:
    sep = Markdown("---")
    baseDir = os.path.dirname(os.path.realpath(__file__))
    default = os.path.join(baseDir, "config.json")

    def __init__(self) -> None:
        self.cfg = {}
        self.history = []

    def load(self, file):
        with open(file, "r") as f:
            self.cfg = json.load(f)
        self.history.extend(self.cfg.get("history", []))

    @property
    def key(self):
        return self.cfg.get("key", os.environ.get("OPENAI_API_KEY", ""))

    @property
    def model(self):
        return self.cfg.get("model", "gpt-3.5-turbo")

    @property
    def prompt(self):
        return self.cfg.get("prompt", [])

    @property
    def stream(self):
        return self.cfg.get("stream", False)

    @property
    def response(self):
        return self.cfg.get("response", False)

    @property
    def proxy(self):
        return self.cfg.get("proxy", "")

    @property
    def threshold(self):
        return self.cfg.get("threshold", 2000)

    @property
    def prompt_summarize(self):
        return self.cfg.get("prompt_summarize", [])

c = Console()
kConfig = Config()

def query_openai(data: dict,summarize=False):
    messages = []
    if summarize:
        messages.extend(kConfig.prompt_summarize)
        #messages.extend(kConfig.prompt)
        temperature=0.5
    else:
        messages.extend(kConfig.prompt)
        temperature=0.7
    messages.extend(data)
    try:
        response = openai.ChatCompletion.create(
            model=kConfig.model,
            messages=messages,
            temperature=temperature
        )
        content = response["choices"][0]["message"]["content"]
        c.print(Markdown(content), Config.sep)
        return content
    except openai.error.OpenAIError as e:
        c.print(e)
    except Exception as e:
        c.print(e)
    return ""

def query_openai_stream(data: dict,summarize=False):
    messages = []
    if summarize:
        temperature=0.5
        messages.extend(kConfig.prompt_summarize)
        #messages.extend(kConfig.prompt)
    else:
        temperature=0.7
        messages.extend(kConfig.prompt)
    messages.extend(data)
    md = Markdown("")
    parser = MarkdownIt().enable("strikethrough")
    answer = ""
    try:
        response = openai.ChatCompletion.create(
            model=kConfig.model,
            messages=messages,
            temperature=temperature,
            stream=True)
        with Live(md, auto_refresh=False) as lv:
            for part in response:
                finish_reason = part["choices"][0]["finish_reason"]
                if "content" in part["choices"][0]["delta"]:
                    content = part["choices"][0]["delta"]["content"]
                    answer += content
                    md.markup = answer
                    md.parsed = parser.parse(md.markup)
                    lv.refresh()
                elif finish_reason:
                    pass
    except KeyboardInterrupt:
        c.print("Canceled")
    except openai.error.OpenAIError as e:
        c.print(e)
        answer = ""
    except Exception as e:
        c.print(e)
    c.print(Config.sep)
    return answer

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301" or model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

class ChatConsole:

    def __init__(self, hist=[]) -> None:
        parser = argparse.ArgumentParser("Input", add_help=False)
        parser.add_argument('-help', action='help', default=argparse.SUPPRESS, help="show this help message")
        parser.add_argument("-reset", action='store_true',
                            help="reset session, i.e. clear chat history")
        parser.add_argument("-exit", action='store_true',
                            help="exit console")
        parser.add_argument("-multiline", action='store_true',
                            help="input multiple lines, end with ctrl-d(Linux/macOS) or ctrl-z(Windows). cancel with ctrl-c")
        parser.add_argument("-summarize", action="store_true", help="summarize the conversation")
        parser.add_argument("-history", action="store_true", help="output conversation history")
        parser.add_argument("-rollback", action="store_true", help="rollback one conversation pair")
        self.parser = parser
        self.hist = hist
        try:
            self.init_readline([opt for action in parser._actions for opt in action.option_strings])
        except Exception as e:
            c.print("Failed to setup readline, autocomplete may not work:", e)

    def init_readline(self, options: List[str]):
        def completer(text, state):
            matches = [o for o in options if o.startswith(text)]
            if state < len(matches):
                return matches[state]
            else:
                return None
        readline.set_completer(completer)
        readline.set_completer_delims(readline.get_completer_delims().replace('-', ''))
        readline.parse_and_bind('tab:complete')

    def parse_input(self) -> str:
        # content = c.input("[bold yellow]Input:[/] ").strip()
        with c.capture() as capture:
            c.print("[bold yellow]Input:[/] ", end="")
        content = input(capture.get())
        if not content.startswith("-"):
            return content
        # handle console options locally
        try:
            args = self.parser.parse_args(content.split())
        except SystemExit:
            return ""
        except argparse.ArgumentError as e:
            print(e)
            return ""
        if args.reset:
            kConfig.history.clear()
            c.print("Session cleared.")
        elif args.multiline:
            return self.read_multiline()
        elif args.summarize:
            self.summarize_history()
        elif args.history:
            self.output_history()
        elif args.rollback:
            self.rollback()
        elif args.exit:
            raise EOFError
        else:
            print("???", args)
        return ""

    def read_multiline(self) -> str:
        contents = []
        while True:
            try:
                line = input("> ")
            except EOFError:
                c.print("--- EOF ---")
                break
            except KeyboardInterrupt:
                return ""
            contents.append(line)
        return "\n".join(contents)

    def summarize_history(self):
        summarize_content = "Please summarize the above conversation in English"
        #summarize_content = "Help me summarize the above conversation in English, to reduce the number of words while ensuring the quality of the conversation. You have to summarize them using figurative language well and without any inappropriate and offensive words. Summary should includes all the point the conversations have, so you should use adequate token (around 500) to cover them. Please do not include this sentence in your summary."
        summarize_messages = self.hist[0:-4] + [{"role": "user", "content": summarize_content}]
        try:
            if kConfig.stream:
                summary = query_openai_stream(summarize_messages, summarize=True)
                #summary = query_openai_stream(summarize_messages)
            else:
                summary = query_openai(summarize_messages, summarize=True)
                #summary = query_openai(summarize_messages)
        except Exception as e:
            c.print(e)
            return
        if summary and summary.find("I'm sorry") != 0 and summary.find("I apologize") != 0 \
                and summary.find("As an AI language model") == -1 and len(summary) > 20:
            self.hist[2:-6] = [{"role": "assistant", "content": summary}]
        else:
            c.print("no summary")
            return
    
    def output_history(self):
        for msg in self.hist:
            if msg['role'] == 'user':
                c.print(f"## User:\n\n{msg['content']}\n\n")
            else:
                c.print(f"## Bot:\n\n{msg['content']}\n\n")
        c.print(f"## end ##")

    def rollback(self):
        self.hist.pop()
        self.hist.pop()
        c.print(f"## rollback one conversations ##")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", dest="config", help="path to config.json", default=Config.default)
    args = parser.parse_args()

    c.print(f"Loading config from {args.config}")
    kConfig.load(args.config)
    if kConfig.key:
        openai.api_key = kConfig.key
    if kConfig.proxy:
        c.print(f"Using proxy: {kConfig.proxy}")
        openai.proxy = kConfig.proxy
    c.print(f"Response in prompt: {kConfig.response}")
    c.print(f"Stream mode: {kConfig.stream}")

    hist = kConfig.history # just alias
    raw_hist = []
    chat = ChatConsole(hist=hist)
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"chatlog-{now}.md"

    while True:
        try:
            content = chat.parse_input().strip()
            if not content:
                continue
            hist.append({"role": "user", "content": content})
            if kConfig.stream:
                answer = query_openai_stream(hist)
            else:
                answer = query_openai(hist)
        except KeyboardInterrupt:
            c.print("Bye!")
            break
        except EOFError as e:
            c.print("Bye!")
            break
        if not answer:
            hist.pop()
        elif kConfig.response:
            hist.append({"role": "assistant", "content": answer})
            raw_hist.append({"role": "user", "content": content})
            raw_hist.append({"role": "assistant", "content": answer})
            with open(log_file, "a", encoding='utf-8' ) as f:
                f.write(f"## User:\n\n{content}\n\n")
                f.write(f"## Bot:\n\n{answer}\n\n")
        num_tokens = num_tokens_from_messages(hist, kConfig.model)
        c.print(f"num of tokens: {num_tokens}")
        if  num_tokens >= kConfig.threshold:
            chat.summarize_history()

if __name__ == '__main__':
    main()
