"""BeichtBot V2 - Discord bot for anonymous confessions.

This module contains the complete bot logic in a single file as requested.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands

def load_env_file(path: Path = Path(".env")) -> None:
    """Load environment variables from a .env file if present."""

    if not path.exists():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
    except OSError as exc:
        logging.getLogger("beichtbot").warning("Konnte .env-Datei nicht lesen: %s", exc)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("beichtbot")

load_env_file()

CONFIG_FILE = Path("beichtbot_config.json")
DEFAULT_SECRET = os.environ.get("BEICHTBOT_SECRET", "beichtbot-secret")
AI_MODEL = os.environ.get("BEICHTBOT_OPENAI_MODEL", "gpt-3.5-turbo")

LINK_PATTERN = re.compile(r"https?://", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
CRISIS_KEYWORDS = {
    "suicide",
    "selbstmord",
    "ich kann nicht mehr",
    "kill myself",
    "notfall",
    "hilfe sofort",
}


@dataclass
class MessageMeta:
    allow_replies: bool = False
    thread_id: Optional[int] = None
    author_hash: Optional[str] = None
    created_at: float = field(default_factory=lambda: dt.datetime.utcnow().timestamp())


@dataclass
class GuildConfig:
    guild_id: int
    target_channel_id: Optional[int] = None
    mod_channel_id: Optional[int] = None
    allowed_channel_ids: List[int] = field(default_factory=list)
    cooldown_seconds: int = 300
    auto_delete_minutes: Optional[int] = None
    ai_moderation: bool = False
    default_thread_lock: bool = False
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)
    banner_message_id: Optional[int] = None
    stats: Dict[str, int] = field(default_factory=lambda: {
        "confessions": 0,
        "reports": 0,
        "ai_flags": 0,
    })
    message_meta: Dict[str, MessageMeta] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["message_meta"] = {mid: asdict(meta) for mid, meta in self.message_meta.items()}
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "GuildConfig":
        data = data.copy()
        message_meta = {
            mid: MessageMeta(**meta) for mid, meta in data.get("message_meta", {}).items()
        }
        data["message_meta"] = message_meta
        return cls(**data)


class GuildConfigStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._configs: Dict[int, GuildConfig] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            LOGGER.info("No config file found, starting with empty configuration.")
            return
        with self.path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for gid, cfg in raw.items():
            self._configs[int(gid)] = GuildConfig.from_json(cfg)
        LOGGER.info("Loaded configuration for %s guilds", len(self._configs))

    def save(self) -> None:
        raw = {str(gid): cfg.to_json() for gid, cfg in self._configs.items()}
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2, ensure_ascii=False)
        LOGGER.debug("Configuration saved.")

    def get(self, guild_id: int) -> GuildConfig:
        if guild_id not in self._configs:
            self._configs[guild_id] = GuildConfig(guild_id=guild_id)
        return self._configs[guild_id]

    def update(self, guild_id: int, **kwargs: Any) -> GuildConfig:
        cfg = self.get(guild_id)
        for key, value in kwargs.items():
            setattr(cfg, key, value)
        self.save()
        return cfg


class ConfessionModal(discord.ui.Modal, title="Beichte einreichen"):
    confession: discord.ui.TextInput[discord.ui.Modal] = discord.ui.TextInput(
        label="Deine Beichte", style=discord.TextStyle.paragraph, max_length=1900
    )
    triggers: discord.ui.TextInput[discord.ui.Modal] = discord.ui.TextInput(
        label="Triggerwörter (optional)", required=False, max_length=200
    )
    target_channel: discord.ui.TextInput[discord.ui.Modal] = discord.ui.TextInput(
        label="Ziel-Channel-ID (optional)", required=False, max_length=25
    )

    def __init__(
        self,
        bot: "BeichtBot",
        interaction: discord.Interaction,
        allow_replies: bool,
        thread_lock: Optional[bool],
        post_with_name: bool,
    ) -> None:
        super().__init__()
        self.bot = bot
        self.interaction = interaction
        self.allow_replies = allow_replies
        self.thread_lock = thread_lock
        self.post_with_name = post_with_name

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self.bot.handle_confession_submission(
            command_interaction=self.interaction,
            modal_interaction=interaction,
            text=str(self.confession.value),
            triggers=str(self.triggers.value or ""),
            target_channel=str(self.target_channel.value or ""),
            allow_replies=self.allow_replies,
            thread_lock=self.thread_lock,
            post_with_name=self.post_with_name,
        )

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        LOGGER.exception("Error in confession modal: %s", error)
        await interaction.response.send_message(
            "Es ist ein Fehler aufgetreten. Bitte versuche es erneut.", ephemeral=True
        )


class ReplyModal(discord.ui.Modal, title="Anonyme Antwort"):
    reply_text: discord.ui.TextInput[discord.ui.Modal] = discord.ui.TextInput(
        label="Antwort", style=discord.TextStyle.paragraph, max_length=1800
    )

    def __init__(self, bot: "BeichtBot", message_id: int, unlock_thread: bool) -> None:
        super().__init__()
        self.bot = bot
        self.message_id = message_id
        self.unlock_thread = unlock_thread

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self.bot.handle_reply_submission(
            interaction=interaction,
            message_id=self.message_id,
            text=str(self.reply_text.value),
            unlock_thread=self.unlock_thread,
        )

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        LOGGER.exception("Error in reply modal: %s", error)
        await interaction.response.send_message(
            "Antwort konnte nicht gesendet werden.", ephemeral=True
        )


class BeichtBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.messages = True
        intents.message_content = False
        intents.members = False
        super().__init__(command_prefix="!", intents=intents)
        self.config_store = GuildConfigStore(CONFIG_FILE)
        self.cooldowns: Dict[int, Dict[int, dt.datetime]] = {}

    async def setup_hook(self) -> None:
        await self.tree.sync()
        LOGGER.info("Slash commands synchronised.")

    def get_guild_config(self, guild: discord.abc.Snowflake) -> GuildConfig:
        return self.config_store.get(guild.id)

    def is_admin(self, interaction: discord.Interaction) -> bool:
        return interaction.user.guild_permissions.manage_guild  # type: ignore[return-value]

    def is_moderator(self, interaction: discord.Interaction) -> bool:
        perms = interaction.user.guild_permissions  # type: ignore[attr-defined]
        return perms.manage_messages or perms.manage_guild

    def sanitize_content(self, content: str) -> str:
        return discord.utils.escape_mentions(content)

    def build_trigger_prefix(self, triggers: List[str]) -> str:
        if not triggers:
            return ""
        return "TW: " + ", ".join(triggers)

    def wrap_spoiler(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text.startswith("||") and text.endswith("||"):
            return text
        return f"||{text}||"

    def hash_user(self, user_id: int, message_id: int) -> str:
        payload = f"{user_id}:{message_id}".encode("utf-8")
        return hmac.new(DEFAULT_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    def check_cooldown(self, guild_id: int, user_id: int, seconds: int) -> Optional[int]:
        now = dt.datetime.utcnow()
        per_guild = self.cooldowns.setdefault(guild_id, {})
        last = per_guild.get(user_id)
        if last and seconds > 0:
            delta = now - last
            if delta.total_seconds() < seconds:
                return int(seconds - delta.total_seconds())
        per_guild[user_id] = now
        return None

    async def handle_confession_submission(
        self,
        command_interaction: discord.Interaction,
        modal_interaction: discord.Interaction,
        text: str,
        triggers: str,
        target_channel: str,
        allow_replies: bool,
        thread_lock: Optional[bool],
        post_with_name: bool,
    ) -> None:
        if not command_interaction.guild:
            await modal_interaction.response.send_message(
                "Dieser Befehl kann nur in einem Server verwendet werden.", ephemeral=True
            )
            return

        cfg = self.get_guild_config(command_interaction.guild)
        if cfg.target_channel_id is None:
            await modal_interaction.response.send_message(
                "Der BeichtBot ist noch nicht konfiguriert. Bitte wende dich an das Team.",
                ephemeral=True,
            )
            return

        cooldown = self.check_cooldown(
            command_interaction.guild.id,
            command_interaction.user.id,
            cfg.cooldown_seconds,
        )
        if cooldown is not None:
            await modal_interaction.response.send_message(
                f"Bitte warte {cooldown} Sekunden, bevor du erneut postest.",
                ephemeral=True,
            )
            return

        sanitized_text = self.sanitize_content(text)
        triggers_list = [t.strip() for t in triggers.split(",") if t.strip()]
        if cfg.blacklist:
            lowered = sanitized_text.lower()
            if any(word.lower() in lowered for word in cfg.blacklist):
                await modal_interaction.response.send_message(
                    "Dein Text enthält gesperrte Begriffe.", ephemeral=True
                )
                return
        if cfg.whitelist and not any(
            word.lower() in sanitized_text.lower() for word in cfg.whitelist
        ):
            await modal_interaction.response.send_message(
                "Dein Text erfüllt nicht die Freigabe-Kriterien.", ephemeral=True
            )
            return

        chosen_channel_id = cfg.target_channel_id
        if target_channel:
            try:
                provided_id = int(target_channel)
            except ValueError:
                provided_id = None
            if provided_id and (not cfg.allowed_channel_ids or provided_id in cfg.allowed_channel_ids):
                chosen_channel_id = provided_id

        channel = command_interaction.guild.get_channel(chosen_channel_id) if chosen_channel_id else None
        if not isinstance(channel, (discord.TextChannel, discord.ForumChannel)):
            await modal_interaction.response.send_message(
                "Der Ziel-Channel konnte nicht gefunden werden.", ephemeral=True
            )
            return

        prefix = self.build_trigger_prefix(triggers_list)
        spoilered = self.wrap_spoiler(sanitized_text)
        if post_with_name:
            author_line = f"Eingereicht von: {command_interaction.user.display_name}"
        else:
            author_line = "Eingereicht von: anonym"

        content_parts = [author_line]
        if prefix:
            content_parts.append(prefix)
        if spoilered:
            content_parts.append(spoilered)
        content = "\n".join(content_parts)

        if isinstance(channel, discord.ForumChannel):
            thread = await channel.create_thread(name=prefix or "Neue Beichte", content=content)
            message = thread.message
        else:
            message = await channel.send(content)
            thread_name = (prefix or "Beichte")[0:90]
            thread = await message.create_thread(name=thread_name)

        if thread_lock if thread_lock is not None else cfg.default_thread_lock:
            await thread.edit(locked=True)

        if not allow_replies:
            await thread.edit(locked=True)

        cfg.stats["confessions"] = cfg.stats.get("confessions", 0) + 1

        meta = MessageMeta(
            allow_replies=allow_replies,
            thread_id=thread.id,
            author_hash=self.hash_user(command_interaction.user.id, message.id),
        )
        cfg.message_meta[str(message.id)] = meta
        self.config_store.save()

        await modal_interaction.response.send_message(
            "Deine Beichte wurde anonym veröffentlicht. Danke für dein Vertrauen!",
            ephemeral=True,
        )

        if cfg.auto_delete_minutes:
            delay = cfg.auto_delete_minutes * 60
            asyncio.create_task(self._schedule_auto_delete(message, thread, delay))

        hints = self.detect_risk_signals(sanitized_text)
        if hints and cfg.mod_channel_id:
            await self.notify_moderators(cfg, message, hints)

        if cfg.ai_moderation:
            flagged = await self.run_ai_moderation(sanitized_text)
            if flagged:
                cfg.stats["ai_flags"] = cfg.stats.get("ai_flags", 0) + 1
                self.config_store.save()
                if cfg.mod_channel_id:
                    await self.notify_moderators(
                        cfg,
                        message,
                        hints + ["KI: mögliche Krise erkannt"],
                    )

    async def _schedule_auto_delete(
        self, message: discord.Message, thread: discord.Thread, delay: int
    ) -> None:
        await asyncio.sleep(delay)
        try:
            await message.delete()
        except discord.NotFound:
            pass
        try:
            await thread.delete()
        except discord.HTTPException:
            pass

    async def notify_moderators(
        self, cfg: GuildConfig, message: discord.Message, reasons: List[str]
    ) -> None:
        guild = message.guild
        if not guild or not cfg.mod_channel_id:
            return
        channel = guild.get_channel(cfg.mod_channel_id)
        if not isinstance(channel, discord.TextChannel):
            return
        reason_text = ", ".join(reasons)
        await channel.send(
            f"🛎️ Hinweis zum Post {message.jump_url}: {reason_text}"
        )

    def detect_risk_signals(self, text: str) -> List[str]:
        reasons: List[str] = []
        if LINK_PATTERN.search(text):
            reasons.append("Link erkannt")
        if EMAIL_PATTERN.search(text):
            reasons.append("E-Mail erkannt")
        lower = text.lower()
        crisis_hits = [kw for kw in CRISIS_KEYWORDS if kw in lower]
        if crisis_hits:
            reasons.append("Krisenhinweis: " + ", ".join(crisis_hits))
        return reasons

    async def run_ai_moderation(self, text: str) -> bool:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            LOGGER.warning("AI moderation enabled but OPENAI_API_KEY not set.")
            return False
        try:
            from openai import AsyncOpenAI
        except ImportError:
            LOGGER.exception("openai package not installed.")
            return False
        client = AsyncOpenAI(api_key=api_key)
        try:
            response = await client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": "Analysiere den Text auf Krisensituationen."},
                    {"role": "user", "content": text},
                ],
                max_tokens=10,
            )
            choice = response.choices[0]
            content = (choice.message.content or "").lower()
            return any(keyword in content for keyword in ["ja", "crisis", "risk"])
        except Exception as exc:
            LOGGER.exception("AI moderation failed: %s", exc)
            return False

    async def handle_reply_submission(
        self,
        interaction: discord.Interaction,
        message_id: int,
        text: str,
        unlock_thread: bool,
    ) -> None:
        if not interaction.guild:
            await interaction.response.send_message(
                "Dieser Befehl ist nur auf Servern verfügbar.", ephemeral=True
            )
            return
        cfg = self.get_guild_config(interaction.guild)
        meta = cfg.message_meta.get(str(message_id))
        if not meta:
            await interaction.response.send_message(
                "Die Beichte wurde nicht gefunden.", ephemeral=True
            )
            return
        if not meta.allow_replies:
            await interaction.response.send_message(
                "Für diese Beichte sind Antworten deaktiviert.", ephemeral=True
            )
            return
        if not meta.thread_id:
            await interaction.response.send_message(
                "Diese Beichte besitzt keinen Thread.", ephemeral=True
            )
            return
        thread = interaction.guild.get_thread(meta.thread_id)
        if not thread:
            await interaction.response.send_message(
                "Thread wurde nicht gefunden.", ephemeral=True
            )
            return
        if unlock_thread and thread.locked:
            await thread.edit(locked=False)
        sanitized = self.sanitize_content(text)
        await thread.send(self.wrap_spoiler(sanitized))
        await interaction.response.send_message(
            "Antwort gesendet.", ephemeral=True
        )

    async def on_ready(self) -> None:
        LOGGER.info("BeichtBot ist eingeloggt als %s", self.user)


bot = BeichtBot()


@bot.tree.command(name="hilfe", description="Informationen über den BeichtBot")
async def help_command(interaction: discord.Interaction) -> None:
    embed = discord.Embed(
        title="BeichtBot – Hilfe",
        description="Anonyme Unterstützung für sensible Themen.",
        colour=discord.Colour.blurple(),
    )
    embed.add_field(
        name="/beichten",
        value="Öffnet ein anonymes Formular. TW können angegeben werden.",
        inline=False,
    )
    embed.add_field(
        name="/beichtantwort",
        value="Antwortet anonym auf eine Beichte (wenn freigeschaltet).",
        inline=False,
    )
    embed.add_field(
        name="/melden",
        value="Meldet diskret eine Beichte an das Moderationsteam.",
        inline=False,
    )
    embed.set_footer(text="Hinweis: Der Bot speichert nur pseudonyme Hashes zur Missbrauchsprüfung.")
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="beichten", description="Anonyme Beichte abgeben")
@app_commands.describe(
    antworten_erlauben="Erlaubt anonyme Antworten im Thread",
    thread_sperren="Thread sofort sperren",
    name_anzeigen="Mit Servernamen posten",
)
async def beichten_command(
    interaction: discord.Interaction,
    antworten_erlauben: Optional[bool] = False,
    thread_sperren: Optional[bool] = None,
    name_anzeigen: Optional[bool] = False,
) -> None:
    modal = ConfessionModal(
        bot,
        interaction,
        allow_replies=bool(antworten_erlauben),
        thread_lock=thread_sperren,
        post_with_name=bool(name_anzeigen),
    )
    await interaction.response.send_modal(modal)


@bot.tree.command(name="beichtantwort", description="Anonyme Antwort senden")
@app_commands.describe(
    nachricht_id="ID der ursprünglichen BeichtBot-Nachricht",
    entsperren="Thread zum Antworten entsperren",
)
async def reply_command(
    interaction: discord.Interaction,
    nachricht_id: str,
    entsperren: Optional[bool] = False,
) -> None:
    try:
        message_id = int(nachricht_id)
    except ValueError:
        await interaction.response.send_message("Ungültige Nachrichten-ID.", ephemeral=True)
        return
    modal = ReplyModal(bot, message_id=message_id, unlock_thread=bool(entsperren))
    await interaction.response.send_modal(modal)


@bot.tree.command(name="melden", description="BeichtBot-Post melden")
@app_commands.describe(
    nachricht_id="ID der Nachricht",
    grund="Optionaler Grund",
)
async def report_command(
    interaction: discord.Interaction,
    nachricht_id: str,
    grund: Optional[str] = None,
) -> None:
    if not interaction.guild:
        await interaction.response.send_message("Nur auf Servern verfügbar.", ephemeral=True)
        return
    cfg = bot.get_guild_config(interaction.guild)
    if cfg.mod_channel_id is None:
        await interaction.response.send_message(
            "Es ist kein Moderationskanal konfiguriert.", ephemeral=True
        )
        return
    try:
        message_id = int(nachricht_id)
    except ValueError:
        await interaction.response.send_message("Ungültige Nachrichten-ID.", ephemeral=True)
        return
    cfg.stats["reports"] = cfg.stats.get("reports", 0) + 1
    bot.config_store.save()
    channel = interaction.guild.get_channel(cfg.mod_channel_id)
    if isinstance(channel, discord.TextChannel):
        await channel.send(
            f"⚠️ Meldung von {interaction.user.display_name}: Nachricht {message_id} – {grund or 'kein Grund angegeben'}"
        )
    await interaction.response.send_message("Danke für deine Meldung.", ephemeral=True)


# Admin and moderator commands

def admin_only(func):
    async def wrapper(interaction: discord.Interaction, *args, **kwargs):
        if not interaction.guild or not bot.is_admin(interaction):
            await interaction.response.send_message(
                "Du benötigst Adminrechte für diesen Befehl.", ephemeral=True
            )
            return
        return await func(interaction, *args, **kwargs)

    return wrapper


def moderator_only(func):
    async def wrapper(interaction: discord.Interaction, *args, **kwargs):
        if not interaction.guild or not bot.is_moderator(interaction):
            await interaction.response.send_message(
                "Du benötigst Moderationsrechte.", ephemeral=True
            )
            return
        return await func(interaction, *args, **kwargs)

    return wrapper


@bot.tree.command(name="beichtbot-setup", description="BeichtBot konfigurieren")
@app_commands.describe(
    ziel_channel="Standard-Ziel-Channel",
    mod_channel="Moderationshinweis-Channel",
    cooldown="Cooldown in Sekunden",
    auto_delete="Automatisches Löschen in Minuten",
    ai_moderation="KI-Moderation aktivieren",
    thread_lock="Threads standardmäßig sperren",
)
async def setup_command(
    interaction: discord.Interaction,
    ziel_channel: str,
    mod_channel: Optional[str] = None,
    cooldown: Optional[int] = 300,
    auto_delete: Optional[int] = None,
    ai_moderation: Optional[bool] = False,
    thread_lock: Optional[bool] = False,
) -> None:
    await admin_only(_setup_impl)(
        interaction,
        ziel_channel=ziel_channel,
        mod_channel=mod_channel,
        cooldown=cooldown,
        auto_delete=auto_delete,
        ai_moderation=ai_moderation,
        thread_lock=thread_lock,
    )


async def _setup_impl(
    interaction: discord.Interaction,
    ziel_channel: str,
    mod_channel: Optional[str],
    cooldown: Optional[int],
    auto_delete: Optional[int],
    ai_moderation: Optional[bool],
    thread_lock: Optional[bool],
) -> None:
    try:
        ziel_channel_id = int(ziel_channel)
    except ValueError:
        await interaction.response.send_message("Ungültige Ziel-Channel-ID.", ephemeral=True)
        return
    mod_channel_id: Optional[int] = None
    if mod_channel:
        try:
            mod_channel_id = int(mod_channel)
        except ValueError:
            await interaction.response.send_message("Ungültige Mod-Channel-ID.", ephemeral=True)
            return
    cfg = bot.config_store.update(
        interaction.guild.id,
        target_channel_id=ziel_channel_id,
        mod_channel_id=mod_channel_id,
        cooldown_seconds=cooldown or 0,
        auto_delete_minutes=auto_delete,
        ai_moderation=bool(ai_moderation),
        default_thread_lock=bool(thread_lock),
    )
    await interaction.response.send_message(
        f"BeichtBot konfiguriert. Ziel: {cfg.target_channel_id}.", ephemeral=True
    )


@bot.tree.command(name="beichtbot-kanäle", description="Erlaubte Channels festlegen")
@app_commands.describe(ids="Kommagetrennte Channel-IDs")
async def channels_command(interaction: discord.Interaction, ids: Optional[str] = None) -> None:
    await admin_only(_channels_impl)(interaction, ids=ids)


async def _channels_impl(interaction: discord.Interaction, ids: Optional[str]) -> None:
    cfg = bot.get_guild_config(interaction.guild)
    if ids:
        try:
            channel_ids = [int(cid.strip()) for cid in ids.split(",") if cid.strip()]
        except ValueError:
            await interaction.response.send_message("Ungültige ID-Liste.", ephemeral=True)
            return
    else:
        channel_ids = []
    cfg.allowed_channel_ids = channel_ids
    bot.config_store.save()
    if channel_ids:
        await interaction.response.send_message(
            "Folgende Channels sind erlaubt: " + ", ".join(map(str, channel_ids)),
            ephemeral=True,
        )
    else:
        await interaction.response.send_message(
            "Alle Channels sind wieder erlaubt.", ephemeral=True
        )


@bot.tree.command(name="beichtbot-wörter", description="Listen verwalten")
@app_commands.describe(
    blacklist="Kommagetrennte Liste blockierter Wörter",
    whitelist="Kommagetrennte Liste erforderlicher Wörter",
)
async def words_command(
    interaction: discord.Interaction,
    blacklist: Optional[str] = None,
    whitelist: Optional[str] = None,
) -> None:
    await moderator_only(_words_impl)(interaction, blacklist=blacklist, whitelist=whitelist)


async def _words_impl(
    interaction: discord.Interaction,
    blacklist: Optional[str],
    whitelist: Optional[str],
) -> None:
    cfg = bot.get_guild_config(interaction.guild)
    cfg.blacklist = [w.strip() for w in blacklist.split(",")] if blacklist else []
    cfg.whitelist = [w.strip() for w in whitelist.split(",")] if whitelist else []
    bot.config_store.save()
    await interaction.response.send_message("Listen aktualisiert.", ephemeral=True)


@bot.tree.command(name="beichtbot-hash", description="Hash zu einer Nachricht anzeigen")
@app_commands.describe(nachricht_id="ID der BeichtBot-Nachricht")
async def hash_command(interaction: discord.Interaction, nachricht_id: str) -> None:
    await moderator_only(_hash_impl)(interaction, nachricht_id=nachricht_id)


async def _hash_impl(interaction: discord.Interaction, nachricht_id: str) -> None:
    cfg = bot.get_guild_config(interaction.guild)
    meta = cfg.message_meta.get(nachricht_id)
    if not meta:
        await interaction.response.send_message("Keine Daten vorhanden.", ephemeral=True)
        return
    await interaction.response.send_message(
        f"Hash-ID: {meta.author_hash}", ephemeral=True
    )


@bot.tree.command(name="beichtbot-stats", description="Statistiken anzeigen")
async def stats_command(interaction: discord.Interaction) -> None:
    await moderator_only(_stats_impl)(interaction)


async def _stats_impl(interaction: discord.Interaction) -> None:
    cfg = bot.get_guild_config(interaction.guild)
    embed = discord.Embed(title="BeichtBot-Statistiken", colour=discord.Colour.green())
    for key, value in cfg.stats.items():
        embed.add_field(name=key.title(), value=str(value))
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="beichtbot-reset", description="Konfiguration zurücksetzen")
async def reset_command(interaction: discord.Interaction) -> None:
    await admin_only(_reset_impl)(interaction)


async def _reset_impl(interaction: discord.Interaction) -> None:
    guild_id = interaction.guild.id
    bot.config_store._configs[guild_id] = GuildConfig(guild_id=guild_id)
    bot.config_store.save()
    await interaction.response.send_message("BeichtBot wurde zurückgesetzt.", ephemeral=True)


@bot.tree.command(name="beichtbot-banner", description="Banner festlegen")
@app_commands.describe(text="Hinweistext für den Banner")
async def banner_command(interaction: discord.Interaction, text: str) -> None:
    await moderator_only(_banner_impl)(interaction, text=text)


async def _banner_impl(interaction: discord.Interaction, text: str) -> None:
    cfg = bot.get_guild_config(interaction.guild)
    channel_id = cfg.target_channel_id
    if not channel_id:
        await interaction.response.send_message("Kein Ziel-Channel konfiguriert.", ephemeral=True)
        return
    channel = interaction.guild.get_channel(channel_id)
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message("Ziel-Channel ist kein Textkanal.", ephemeral=True)
        return
    if cfg.banner_message_id:
        try:
            message = await channel.fetch_message(cfg.banner_message_id)
            await message.edit(content=text)
        except discord.NotFound:
            cfg.banner_message_id = None
    if not cfg.banner_message_id:
        message = await channel.send(text)
        await message.pin()
        cfg.banner_message_id = message.id
    bot.config_store.save()
    await interaction.response.send_message("Banner aktualisiert.", ephemeral=True)


@bot.tree.command(name="beichtbot-nachricht", description="Thread-Link abrufen")
@app_commands.describe(nachricht_id="ID der BeichtBot-Nachricht")
async def message_command(interaction: discord.Interaction, nachricht_id: str) -> None:
    await moderator_only(_message_impl)(interaction, nachricht_id=nachricht_id)


async def _message_impl(interaction: discord.Interaction, nachricht_id: str) -> None:
    cfg = bot.get_guild_config(interaction.guild)
    meta = cfg.message_meta.get(nachricht_id)
    if not meta or not meta.thread_id:
        await interaction.response.send_message("Keine Thread-Informationen.", ephemeral=True)
        return
    thread = interaction.guild.get_thread(meta.thread_id)
    if not thread:
        await interaction.response.send_message("Thread nicht gefunden.", ephemeral=True)
        return
    await interaction.response.send_message(f"Thread-Link: {thread.jump_url}", ephemeral=True)


@bot.tree.command(name="beichtbot-cooldown", description="Cooldown steuern")
@app_commands.describe(user="User-ID für Reset", reset="Cooldown komplett zurücksetzen")
async def cooldown_command(
    interaction: discord.Interaction,
    user: Optional[str] = None,
    reset: Optional[bool] = False,
) -> None:
    await moderator_only(_cooldown_impl)(interaction, user=user, reset=reset)


async def _cooldown_impl(
    interaction: discord.Interaction,
    user: Optional[str],
    reset: Optional[bool],
) -> None:
    guild_cooldowns = bot.cooldowns.setdefault(interaction.guild.id, {})
    if reset:
        guild_cooldowns.clear()
        await interaction.response.send_message("Alle Cooldowns zurückgesetzt.", ephemeral=True)
        return
    if user:
        try:
            user_id = int(user)
        except ValueError:
            await interaction.response.send_message("Ungültige User-ID.", ephemeral=True)
            return
        guild_cooldowns.pop(user_id, None)
        await interaction.response.send_message(
            f"Cooldown für {user_id} entfernt.", ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "Bitte User-ID oder reset angeben.", ephemeral=True
        )


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Setze das DISCORD_TOKEN in den Umgebungsvariablen.")
    bot.run(token)
