"""Regenerate the three QR codes on the final slide.

Edit the three constants below, then run `python3 deck/make_qr.py` (or just `python3 deck/build_deck.py`,
which calls this). Any URL containing "PLACEHOLDER" is drawn on the slide with a red
"PLACEHOLDER, replace" label, so a placeholder can never ship unnoticed.
Output: deck/qr/qr_website.png, deck/qr/qr_instagram.png, deck/qr/qr_linkedin.png
"""
from pathlib import Path

import segno

WEBSITE_URL = "https://quant-soc.com"
INSTAGRAM_URL = "https://www.instagram.com/quantsoc_edinburgh/"
LINKEDIN_URL = "https://www.linkedin.com/company/quantsociety"

QR_DIR = Path(__file__).resolve().parent / "qr"
CODES = [
    ("website", "Website", WEBSITE_URL),
    ("instagram", "Instagram", INSTAGRAM_URL),
    ("linkedin", "LinkedIn", LINKEDIN_URL),
]


def is_placeholder(url: str) -> bool:
    return "PLACEHOLDER" in url.upper()


def make_all() -> list[dict]:
    QR_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for key, label, url in CODES:
        path = QR_DIR / f"qr_{key}.png"
        segno.make(url, error="m").save(path, scale=12, border=2, dark="#111827", light="#ffffff")
        out.append({"key": key, "label": label, "url": url, "path": path, "placeholder": is_placeholder(url)})
    return out


if __name__ == "__main__":
    for c in make_all():
        flag = "  [PLACEHOLDER, replace]" if c["placeholder"] else ""
        print(f"{c['path'].name}: {c['url']}{flag}")
