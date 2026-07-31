# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Interactive prompt helpers for manual hardware-in-the-loop tests."""

import asyncio
import contextlib
import contextvars
import functools
import os
import sys
from typing import IO, Any, Dict, Generic, Iterator, Optional, Set, TextIO, TypeVar, Union, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.input import Input
from prompt_toolkit.output import Output
from prompt_toolkit.patch_stdout import StdoutProxy
from prompt_toolkit.styles import BaseStyle, Style
from prompt_toolkit.validation import Validator

T = TypeVar("T", bound=IO)


class IOProxy(Generic[T]):
    """An IO stream proxy that allows redirection."""

    class Redirection:
        """A reversible IO proxy redirection."""

        def __init__(self, proxy: "IOProxy[T]", stream: T) -> None:
            """Initialize a redirection for a proxy to a target stream.

            Args:
                proxy: IO proxy whose stream should be redirected.
                stream: target stream to use while the redirection is enabled.
            """
            self.proxy = proxy
            self.base_stream = proxy.stream
            self.target_stream = stream

        def enable(self) -> None:
            """Route the proxy to the target stream."""
            self.proxy.stream = self.target_stream

        def disable(self) -> None:
            """Restore the proxy to its original stream."""
            self.proxy.stream = self.base_stream

    def __init__(self, stream: T) -> None:
        """Initializes proxy, binding it to the given `stream`."""
        self.stream = stream

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the currently proxied stream.

        Args:
            name: attribute name to resolve on the wrapped stream.

        Returns:
            the attribute value from the current stream.
        """
        return getattr(self.stream, name)

    @contextlib.contextmanager
    def redirect_to(self, stream: T, deferred: bool = False) -> Iterator[Redirection]:
        """Redirects all IO within scope to the given `stream`."""
        redirect = IOProxy.Redirection(self, stream)
        if not deferred:
            redirect.enable()
        try:
            yield redirect
        finally:
            redirect.disable()


class PatchedStdoutProxy(StdoutProxy):
    """A patched `prompt_toolkit.patch_stdout.StdoutProxy` to enforce context consistency across threads."""

    class ContextBoundEventLoop:
        """An `asyncio.AbstractEventLoop` decorator that binds a given context to every call."""

        def __init__(self, unbound_loop: asyncio.AbstractEventLoop, context: contextvars.Context) -> None:
            """Initialize a loop wrapper that preserves context for scheduled callbacks.

            Args:
                unbound_loop: event loop to decorate.
                context: context to bind to scheduled callback methods.
            """
            self.unbound_loop = unbound_loop
            self.context = context

            self.call_at = functools.partial(self.unbound_loop.call_at, context=self.context)
            self.call_later = functools.partial(self.unbound_loop.call_later, context=self.context)
            self.call_soon = functools.partial(self.unbound_loop.call_soon, context=self.context)
            self.call_soon_threadsafe = functools.partial(self.unbound_loop.call_soon_threadsafe, context=self.context)

        def __getattr__(self, name: str) -> Any:
            """Delegate unknown attributes to the wrapped event loop.

            Args:
                name: attribute name to resolve on the wrapped loop.

            Returns:
                the attribute value from the wrapped loop.
            """
            return getattr(self.unbound_loop, name)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a stdout proxy with the current context captured.

        Args:
            args: positional arguments forwarded to `StdoutProxy`.
            kwargs: keyword arguments forwarded to `StdoutProxy`.
        """
        super().__init__(*args, **kwargs)
        self.context = contextvars.copy_context()

    def _get_app_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Return the active prompt application loop bound to the captured context.

        Returns:
            the context-bound prompt application loop, or None if no loop is available.
        """
        app = self.app_session.app
        if app is None:
            return None

        loop = app.loop
        if loop is None:
            return None

        bound_loop = PatchedStdoutProxy.ContextBoundEventLoop(loop, self.context)
        return cast(asyncio.AbstractEventLoop, bound_loop)


class TestingPromptSession:
    """A `prompt_toolkit.PromptSession` wrapper, tailored for testing purposes."""

    __test__ = False  # to avoid pytest from collecting it

    TRUTHY_STRINGS = ("y", "yes")
    FALSY_STRINGS = ("n", "no")
    BOOLEAN_STRINGS = TRUTHY_STRINGS + FALSY_STRINGS

    DEFAULT_STYLE = Style.from_dict(
        {
            "ask": "bold",
            "attention": "fg:cyan",
            "placeholder": "fg:lightgrey",
        }
    )

    def __init__(
        self,
        input: Input,
        output: Output,
        *,
        erase_when_done: bool = True,
        style: Optional[BaseStyle] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes prompt session with suitable defaults for testing purposes."""
        self.__input = input
        self.__output = output
        if style is None:
            style = TestingPromptSession.DEFAULT_STYLE
        self.settings = dict(erase_when_done=erase_when_done, style=style, **kwargs)
        self.output: IOProxy[TextIO] = IOProxy(self.__output.stdout if self.__output.stdout is not None else sys.stdout)

    def choose(self, question_or_request: str, choices: Union[Set[str], Dict[str, Any]], **kwargs: Any) -> str:
        """
        Prompts the user to make choice.

        Args:
            question_or_request: a question or ask to put the choice in context.
            choices: the set of alternative to choose from.
            kwargs: optional keyword arguments to be forwarded to `TestingPromptSession.ask`.

        Returns:
            the chosen alternative.
        """
        if not isinstance(choices, dict):
            choices = dict(zip(choices, choices, strict=False))
        validator = Validator.from_callable(
            lambda doc: doc in choices,
            move_cursor_to_end=True,
            error_message="Robot name is not valid",
        )
        return choices[
            self.ask(
                question_or_request,
                completer=WordCompleter(list(choices.keys())),
                placeholder=[("class:placeholder", "TAB to autocomplete")],
                validator=validator,
                validate_while_typing=False,
                **kwargs,
            )
        ]

    def boolean_ask(self, question: str, **kwargs: Any) -> bool:
        """
        Prompts the user with a yes-no `question`.

        Args:
            question: a yes-no question to ask the user.
            kwargs: optional keyword arguments to be forwarded to `TestingPromptSession.ask`.

        Returns:
            True for affirmative answers, False for negative answers.
        """
        validator = Validator.from_callable(
            lambda doc: doc.lower() in TestingPromptSession.BOOLEAN_STRINGS,
            error_message="Answer y(es) or n(o) only",
            move_cursor_to_end=True,
        )
        return (
            self.ask(
                question,
                completer=WordCompleter(list(TestingPromptSession.BOOLEAN_STRINGS)),
                placeholder=[("class:placeholder", "y(es) or n(o)")],
                validator=validator,
                validate_while_typing=False,
                **kwargs,
            ).lower()
            in TestingPromptSession.TRUTHY_STRINGS
        )

    def integer_ask(self, question_or_request: str, **kwargs: Any) -> int:
        """
        Prompts the user with a question or an specific ask whose answer is an integer number.

        Args:
            question_or_request: a question or ask for an integer number.
            kwargs: optional keyword arguments to be forwarded to `TestingPromptSession.ask`.

        Returns:
            User given integer number.
        """

        def is_integer(value: Any) -> bool:
            """Return whether a value can be parsed as an integer.

            Args:
                value: candidate value to parse.

            Returns:
                True if the value can be parsed as an integer.
            """
            try:
                int(value)
                return True
            except ValueError:
                return False

        validator = Validator.from_callable(
            is_integer,
            error_message="Answer must be an integer",
            move_cursor_to_end=True,
        )
        return int(
            self.ask(
                question_or_request,
                validator=validator,
                validate_while_typing=False,
                **kwargs,
            )
        )

    def ask(self, question_or_request: str, *, who: Optional[str] = None, **kwargs: Any) -> str:
        """
        Prompts the user with an open-ended question or an specific ask.

        Args:
            question_or_request: an open-ended question or ask to make.
            who: optional label for the user to identify the source of the ask,
            defaults to full name of the current pytest test.
            kwargs: optional keyword arguments to be forwarded to `TestingPromptSession.prompt`.

        Returns:
            User reply as text.
        """
        if who is None:
            who = os.environ.get("PYTEST_CURRENT_TEST", "pytest")
        settings = dict(**self.settings, **kwargs)
        with create_app_session(self.__input, self.__output):
            session: PromptSession[str] = PromptSession(**settings)
            with PatchedStdoutProxy(raw=True) as proxy:
                with self.output.redirect_to(cast(TextIO, proxy), deferred=True) as redirect:
                    return session.prompt(
                        [
                            ("", "\n\n"),
                            ("class:attention", f"!{who} asks:"),
                            ("", " "),
                            ("class:ask", question_or_request),
                            ("", " "),
                        ],
                        pre_run=redirect.enable,
                    )
