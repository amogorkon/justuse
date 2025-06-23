import sys

from pytest import skip


def test_441_discord(reuse):
    try:
        imported = "discord" in sys.modules
        import discord

        if not imported:
            del sys.modules["discord"]
    except ImportError:
        skip("discord is not installed")
        return
    mod = reuse("discord")
    assert mod
