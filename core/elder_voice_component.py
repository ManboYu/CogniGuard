from __future__ import annotations

from typing import Any

import streamlit.components.v1 as components

from core.config import PROJECT_ROOT


ELDER_RECORDER_DIR = PROJECT_ROOT / "components" / "elder_voice_recorder"


_elder_voice_recorder = components.declare_component(
    "elder_voice_recorder",
    path=str(ELDER_RECORDER_DIR),
)


def elder_voice_recorder(**kwargs: Any) -> Any:
    return _elder_voice_recorder(**kwargs)
