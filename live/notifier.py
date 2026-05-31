import logging
import requests
import config

logger = logging.getLogger(__name__)


def send(message: str) -> None:
    """POST message to Discord Webhook. No-op if DISCORD_WEBHOOK_URL not configured."""
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        logger.warning("DISCORD_WEBHOOK_URL not set — skipping notification")
        return
    try:
        resp = requests.post(url, json={"content": message}, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
