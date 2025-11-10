"""
Generic real-estate scraping template wired for the Nigerian portals listed in
the README. Each portal gets its own configuration block describing pagination
and CSS selectors. Verify the selectors against the live markup before running.

Always confirm the site allows automated access (robots.txt + Terms of Service)
and throttle requests so you respect their infrastructure.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import re

import requests
from bs4 import BeautifulSoup, Tag


@dataclass
class SiteConfig:
    name: str
    base_url: str
    card_selector: str
    pagination: Callable[[int], str]
    selectors: dict[str, str] | None = None
    parser: Callable[[Tag | BeautifulSoup], dict[str, str]] | None = None
    delay_seconds: float = 2.0


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; friendly-bot/0.1)"}
OUT_PATH = Path("data/properties.csv")
TARGET_ROWS = 500

MAINLAND_KEYWORDS = [
    "ikeja",
    "yaba",
    "surulere",
    "maryland",
    "ogba",
    "ojodu",
    "berger",
    "mushin",
    "ilupeju",
    "oshodi",
    "gbagada",
    "shomolu",
    "bariga",
    "ketu",
    "magodo",
    "ajibode",
    "isolo",
    "abule egba",
    "ipaja",
    "egbeda",
    "agege",
    "ikotun",
    "ojota",
    "apapa",
]

# Fieldnames to keep CSV columns consistent across sources.
FIELDNAMES = [
    "location",
    "bedrooms",
    "bathrooms",
    "toilets",
    "area_sqm",
    "type",
    "price_naira",
]


def clean_numeric(text: str) -> str:
    """Keep digits/decimal point; drop currency symbols and group separators."""
    cleaned = text.replace("₦", "").replace(",", "").strip()
    cleaned = cleaned.split(" ")[0] if cleaned else ""
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return match.group(0) if match else ""


def grab_text(node: Tag | BeautifulSoup, selector: str) -> str:
    """Return stripped text for a CSS selector, or empty string if missing."""
    element = node.select_one(selector)
    return element.get_text(strip=True) if element else ""


def parse_listing(card: Tag | BeautifulSoup, selectors: dict[str, str]) -> dict[str, str]:
    """Extract required columns for a single listing card."""
    price_raw = grab_text(card, selectors["price"])

    return {
        "location": grab_text(card, selectors["location"]),
        "bedrooms": grab_text(card, selectors["bedrooms"]).split(" ")[0],
        "bathrooms": grab_text(card, selectors["bathrooms"]).split(" ")[0],
        "toilets": grab_text(card, selectors["toilets"]).split(" ")[0],
        "area_sqm": grab_text(card, selectors["area"]).split(" ")[0],
        "type": grab_text(card, selectors["type"]),
        "price_naira": clean_numeric(price_raw),
    }


def page_urls(config: SiteConfig) -> Iterable[str]:
    """Yield page URLs based on the configured pagination function."""
    page = 1
    while True:
        yield config.pagination(page)
        page += 1


def fetch_page(url: str) -> BeautifulSoup:
    """Download a page and return a BeautifulSoup parser."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def is_mainland(location: str) -> bool:
    """Return True if the location text matches a known mainland keyword."""
    text = (location or "").lower()
    return any(keyword in text for keyword in MAINLAND_KEYWORDS)


def harvest_site(config: SiteConfig, limit: int, mainland_only: bool = False) -> list[dict[str, str]]:
    """Scrape listings for a single site until `limit` rows or pages exhausted."""
    rows: list[dict[str, str]] = []

    for url in page_urls(config):
        if len(rows) >= limit:
            break

        print(f"[{config.name}] Fetching {url}")
        soup = fetch_page(url)
        cards = soup.select(config.card_selector)
        if not cards:
            print(f"[{config.name}] No listing cards found; stopping pagination.")
            break

        for card in cards:
            if config.parser:
                row = config.parser(card)
            elif config.selectors:
                row = parse_listing(card, config.selectors)
            else:
                raise ValueError(f"No parser or selectors defined for {config.name}")
            if mainland_only and not is_mainland(row.get("location", "")):
                continue
            rows.append(row)
            if len(rows) >= limit:
                break

        time.sleep(config.delay_seconds)

    print(f"[{config.name}] Collected {len(rows)} rows.")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Create parent directories and write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def get_site_configs() -> dict[str, SiteConfig]:
    """
    Return the portal configurations.

    The selectors below are placeholders. Inspect the live DOM for each site
    and replace them with stable attributes (e.g., data-testid values). Do not
    rely on auto-generated class names—they change frequently.
    """

    def propertypro_pagination(page: int) -> str:
        return f"https://www.propertypro.ng/property-for-rent?page={page}"

    def privateproperty_pagination(page: int) -> str:
        return f"https://www.privateproperty.com.ng/property-for-rent?page={page}"

    def npc_pagination(page: int) -> str:
        return f"https://nigeriapropertycentre.com/for-rent?page={page}"

    def extract_with_keywords(text: str, keywords: tuple[str, ...]) -> str:
        for keyword in keywords:
            match = re.search(r"(\d+(?:\.\d+)?)\s*" + keyword, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def propertypro_parser(card: Tag | BeautifulSoup) -> dict[str, str]:
        stats_text = grab_text(card, ".pl-price h6")
        info_text = grab_text(card, ".pl-title h6")
        type_links = card.select(".pl-title h6 a")
        property_type = ""
        if len(type_links) >= 2:
            property_type = type_links[1].get_text(strip=True)
        elif info_text:
            property_type = info_text

        area = extract_with_keywords(info_text, ("sqm", "sq m", "square metres"))

        # Some layouts show counts in `.property-benefit li`.
        benefit_items = [li.get_text(strip=True) for li in card.select(".property-benefit li")]
        bedrooms = extract_with_keywords(stats_text, ("bed", "beds"))
        bathrooms = extract_with_keywords(stats_text, ("bath", "baths"))
        toilets = extract_with_keywords(stats_text, ("toilet", "toilets"))

        if benefit_items:
            if len(benefit_items) >= 1 and not bedrooms:
                bedrooms = re.sub(r"\D", "", benefit_items[0])
            if len(benefit_items) >= 2 and not bathrooms:
                bathrooms = re.sub(r"\D", "", benefit_items[1])
            if len(benefit_items) >= 3 and not toilets:
                toilets = re.sub(r"\D", "", benefit_items[2])

        price_text = grab_text(card, ".pl-price h3")
        if not price_text:
            price_text = grab_text(card, ".similar-listings-price h4")

        location = grab_text(card, ".pl-title > p")
        if not location:
            location = grab_text(card, ".similar-listing-location")

        return {
            "location": location,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "toilets": toilets,
            "area_sqm": area,
            "type": property_type,
            "price_naira": clean_numeric(price_text),
        }

    def normalize_count(text: str) -> str:
        return re.sub(r"[^\d.]", "", text)

    def privateproperty_parser(card: Tag | BeautifulSoup) -> dict[str, str]:
        price_text = grab_text(card, ".similar-listings-price h4 span[content]") or grab_text(
            card, ".similar-listings-price h4"
        )
        benefit_items = [normalize_count(li.get_text(strip=True)) for li in card.select(".property-benefit li")]

        type_text = grab_text(card, ".similar-listings-info h3")
        type_clean = type_text.replace("FOR RENT", "").replace("For Rent", "").strip()

        return {
            "location": grab_text(card, ".listings-location"),
            "bedrooms": benefit_items[0] if len(benefit_items) >= 1 else "",
            "bathrooms": benefit_items[1] if len(benefit_items) >= 2 else "",
            "toilets": benefit_items[2] if len(benefit_items) >= 3 else "",
            "area_sqm": "",
            "type": type_clean,
            "price_naira": clean_numeric(price_text),
        }

    def extract_feature_value(card: Tag | BeautifulSoup, feature_name: str) -> str:
        selector = (
            f"li[itemprop='additionalProperty']:has(span[itemprop='name']:contains('{feature_name}')) "
            "span[itemprop='value']"
        )
        element = card.select_one(selector)
        if element:
            return normalize_count(element.get_text(strip=True))
        return ""

    def npc_parser(card: Tag | BeautifulSoup) -> dict[str, str]:
        price_span = card.select_one(".wp-block-content span[itemprop='price']")
        price_text = price_span.get_text(strip=True) if price_span else grab_text(
            card, ".wp-block-content .price"
        )

        area_value = ""
        area_li = card.select_one(
            "li[itemprop='additionalProperty']:has(span[itemprop='unitText']) span[itemprop='value']"
        )
        if area_li:
            area_value = normalize_count(area_li.get_text(strip=True))

        return {
            "location": grab_text(card, ".wp-block-content p").strip(" \n"),
            "bedrooms": extract_feature_value(card, "Bedroom"),
            "bathrooms": extract_feature_value(card, "Bathroom"),
            "toilets": extract_feature_value(card, "Toilet"),
            "area_sqm": area_value,
            "type": grab_text(card, ".wp-block-content h4"),
            "price_naira": clean_numeric(price_text),
        }

    return {
        "propertypro": SiteConfig(
            name="PropertyPro",
            base_url="https://www.propertypro.ng/property-for-rent",
            card_selector=".property-listing-content",
            pagination=propertypro_pagination,
            parser=propertypro_parser,
        ),
        "privateproperty": SiteConfig(
            name="PrivateProperty",
            base_url="https://www.privateproperty.com.ng/property-for-rent",
            card_selector=".similar-listings-item",
            pagination=privateproperty_pagination,
            parser=privateproperty_parser,
        ),
        "npc": SiteConfig(
            name="NigeriaPropertyCentre",
            base_url="https://nigeriapropertycentre.com/for-rent",
            card_selector=".wp-block.property.list",
            pagination=npc_pagination,
            parser=npc_parser,
        ),
        # "jiji": SiteConfig(
        #     name="Jiji",
        #     base_url="https://jiji.ng/api_web/v1/listing",
        #     card_selector="",
        #     pagination=lambda page: f"https://jiji.ng/api_web/v1/listing?slug=houses-apartments-for-rent&region_slug=lagos&page={page}&init_page={'true' if page == 1 else 'false'}&webp=false",
        #     parser=None,
        # ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Nigerian property portals.")
    parser.add_argument(
        "site",
        choices=list(get_site_configs().keys()),
        help="Which site configuration to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=TARGET_ROWS,
        help="Maximum number of rows to collect (default: 500).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help="CSV path to write (default: data/properties.csv).",
    )
    parser.add_argument(
        "--mainland-only",
        action="store_true",
        help="Keep only listings whose location matches mainland Lagos neighborhoods.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = get_site_configs()
    config = configs[args.site]

    rows = harvest_site(config, args.limit, mainland_only=args.mainland_only)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
