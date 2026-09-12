"""The spatial actions the agent may take, as Gemini tool declarations + an executor."""
from __future__ import annotations

from dataclasses import dataclass, field

from google.genai import types

from . import spatial
from .perception import Snapshot
from .shell import Shell, ShellError

WHERE = ["in_front", "left", "right", "on_table", "on_wall", "where_looking"]
PANEL_GAP_M = 0.45


def _where_schema(desc: str) -> types.Schema:
    return types.Schema(type=types.Type.STRING, enum=WHERE, description=desc)


DECLARATIONS = [
    types.FunctionDeclaration(
        name="place_note",
        description="Create a note card floating in the room with the given text. Use for reminders, labels, lists, answers the wearer should keep seeing.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "text": types.Schema(type=types.Type.STRING, description="Short note text, under 60 characters."),
                "where": _where_schema("Where to put it relative to the wearer or a surface."),
            },
            required=["text", "where"],
        ),
    ),
    types.FunctionDeclaration(
        name="open_app",
        description="Open a Mac app as a floating panel in the room.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "app": types.Schema(type=types.Type.STRING, description="App name as the user says it, e.g. Safari, Notes, Terminal, Spotify."),
                "where": _where_schema("Where to put the panel."),
            },
            required=["app", "where"],
        ),
    ),
    types.FunctionDeclaration(
        name="move_panel",
        description="Move an existing panel (by handle from the scene) somewhere else.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "handle": types.Schema(type=types.Type.INTEGER),
                "where": _where_schema("Destination."),
            },
            required=["handle", "where"],
        ),
    ),
    types.FunctionDeclaration(
        name="close_panel",
        description="Close a panel by handle.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"handle": types.Schema(type=types.Type.INTEGER)},
            required=["handle"],
        ),
    ),
    types.FunctionDeclaration(
        name="gather_panels",
        description="Bring every panel back in front of the wearer. Use when they say they lost their windows.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="say",
        description="Speak a short reply to the wearer. Always call this once, last, with one sentence.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"text": types.Schema(type=types.Type.STRING)},
            required=["text"],
        ),
    ),
]

APP_ALIASES = {
    "safari": "com.apple.Safari",
    "notes": "com.apple.Notes",
    "terminal": "com.apple.Terminal",
    "finder": "com.apple.finder",
    "messages": "com.apple.MobileSMS",
    "music": "com.apple.Music",
    "spotify": "com.spotify.client",
    "calendar": "com.apple.iCal",
    "chrome": "com.google.Chrome",
    "mail": "com.apple.mail",
    "reminders": "com.apple.reminders",
    "vscode": "com.microsoft.VSCode",
    "code": "com.microsoft.VSCode",
}


@dataclass
class Outcome:
    log: list[str] = field(default_factory=list)
    spoken: str | None = None


class Executor:
    def __init__(self, shell: Shell, snap: Snapshot) -> None:
        self.shell = shell
        self.snap = snap
        self.placed = 0

    def run(self, calls: list[types.FunctionCall]) -> Outcome:
        out = Outcome()
        for call in calls:
            handler = getattr(self, f"_do_{call.name}", None)
            if handler is None:
                out.log.append(f"unknown action {call.name}")
                continue
            try:
                msg = handler(out, **(call.args or {}))
            except ShellError as e:
                msg = f"{call.name} failed: {e}"
            out.log.append(msg)
        return out

    def _target(self, where: str) -> spatial.Vec3:
        head = self.snap.head
        side = self.placed * PANEL_GAP_M
        self.placed += 1
        match where:
            case "left":
                return spatial.in_front(head, 0.8, right=-0.5 - side)
            case "right":
                return spatial.in_front(head, 0.8, right=0.5 + side)
            case "on_table":
                return spatial.in_front(head, 0.7, up=-0.25, right=side)
            case "on_wall" | "where_looking":
                return spatial.in_front(head, 1.6, right=side)
            case _:
                return spatial.in_front(head, 0.9, right=side)

    def _place(self, handle: int, where: str) -> None:
        self.shell.move(handle, self._target(where))
        if where == "on_table":
            self.shell.anchor(handle, "closest-horizontal")
        elif where in ("on_wall", "where_looking"):
            self.shell.anchor(handle, "closest-wall")

    def _do_place_note(self, out: Outcome, text: str, where: str) -> str:
        handle = self.shell.launch_card(text)
        self._place(handle, where)
        return f"note #{handle} {text!r} -> {where}"

    def _do_open_app(self, out: Outcome, app: str, where: str) -> str:
        target = APP_ALIASES.get(app.lower().strip(), app)
        handle = self.shell.launch_app(target)
        self._place(handle, where)
        return f"app {app} #{handle} -> {where}"

    def _do_move_panel(self, out: Outcome, handle: int, where: str) -> str:
        self._place(int(handle), where)
        return f"panel #{handle} -> {where}"

    def _do_close_panel(self, out: Outcome, handle: int) -> str:
        self.shell.close_window(int(handle))
        return f"closed #{handle}"

    def _do_gather_panels(self, out: Outcome) -> str:
        self.shell.gather()
        return "gathered"

    def _do_say(self, out: Outcome, text: str) -> str:
        out.spoken = text
        return f"say {text!r}"
