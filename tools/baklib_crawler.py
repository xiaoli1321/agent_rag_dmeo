from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from ..ingest.pipeline import DEFAULT_SOURCES_PATH


DEFAULT_START_URL = "https://sibionics.demo.baklib.vip/"
SKIP_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".rar",
    ".svg",
    ".webp",
    ".zip",
}


@dataclass(slots=True)
class PageData:
    url: str
    title: str
    text: str
    links: list[str]


class ContentParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._in_body = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "body":
            self._in_body = True
        if tag == "a" and (href := attrs_dict.get("href")):
            self.links.append(urllib.parse.urljoin(self.base_url, href))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag == "body":
            self._in_body = False
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = normalize_text(data)
        if not text or self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(text)
            return
        current = self._tag_stack[-1] if self._tag_stack else ""
        if self._in_body and current not in {"button", "input", "select", "textarea"}:
            self.text_parts.append(text)

    def page_data(self, url: str) -> PageData:
        title = clean_title(" ".join(self.title_parts))
        text = normalize_text(" ".join(self.text_parts))
        if not text:
            text = normalize_text(" ".join(part for part in self.text_parts if part))
        text = clean_baklib_text(text)
        return PageData(url=url, title=title or url, text=text, links=self.links)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_title(value: str) -> str:
    title = normalize_text(value)
    return re.sub(r"\s*\|\s*国内客服知识库\s*$", "", title).strip()


def clean_baklib_text(value: str) -> str:
    text = normalize_text(value)
    replacements = [
        "将页面以 Markdown 格式复制给 LLMs 在 ChatGPT 中打开 询问有关此页面的问题 在 Claude 中打开 询问有关此页面的问题",
        "复制 MCP 安装命令 复制 npx 命令以安装 MCP 服务器",
        "连接到Cursor 在 Cursor 中安装 MCP Server",
        "连接到VS Code 在 VS Code 中安装 MCP Server",
        "国内客服知识库",
        "搜索... ⌘ K",
        "复制页面",
    ]
    for item in replacements:
        text = text.replace(item, " ")
    text = re.sub(r"\b李强\b", " ", text)
    text = re.sub(
        r"提交反馈(?: 上一页 .*?)?(?: 下一页 .*?)? 页内目录 © Baklib\.cn 版权所有 蜀ICP备15035023号$",
        " ",
        text,
    )
    text = re.sub(r"© Baklib\.cn 版权所有 蜀ICP备15035023号$", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch(url: str, *, cookie: str, timeout: float) -> str:
    headers = {
        "User-Agent": "customer-agent-demo-authorized-crawler/0.1",
        "Accept": "text/html,application/xhtml+xml",
        "Cookie": cookie,
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):
            raise ValueError(f"skip non-html content: {content_type}")
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def same_site_url(url: str, *, root: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    root_parsed = urllib.parse.urlparse(root)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc != root_parsed.netloc:
        return None
    clean = parsed._replace(fragment="", query="")
    if Path(clean.path).suffix.lower() in SKIP_EXTENSIONS:
        return None
    return urllib.parse.urlunparse(clean)


def crawl(
    start_url: str,
    *,
    cookie: str,
    max_pages: int,
    delay_seconds: float,
    timeout: float,
) -> list[PageData]:
    queue = [start_url]
    seen: set[str] = set()
    pages: list[PageData] = []

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        normalized = same_site_url(url, root=start_url)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        try:
            html = fetch(normalized, cookie=cookie, timeout=timeout)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
        ) as exc:
            print(f"[skip] {normalized}: {exc}", file=sys.stderr)
            continue

        parser = ContentParser(normalized)
        parser.feed(html)
        page = parser.page_data(normalized)
        if page.text:
            pages.append(page)
            print(f"[page] {len(pages):03d} {page.title} {normalized}", file=sys.stderr)

        for link in page.links:
            next_url = same_site_url(link, root=start_url)
            if next_url and next_url not in seen and next_url not in queue:
                queue.append(next_url)
        time.sleep(delay_seconds)

    return pages


def to_source_records(
    pages: list[PageData], *, product_tags: list[str], language: str
) -> list[dict[str, object]]:
    return [
        {
            "source_title": page.title,
            "source_url": page.url,
            "source_type": "baklib-authorized-page",
            "language": language,
            "product_tags": product_tags,
            "text": page.text,
        }
        for page in pages
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl authorized Baklib pages into customer_agent_demo source JSON."
    )
    parser.add_argument(
        "--start-url", default=DEFAULT_START_URL, help=f"Default: {DEFAULT_START_URL}"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_SOURCES_PATH, help="Output JSON path."
    )
    parser.add_argument(
        "--cookie-env",
        default="BAKLIB_COOKIE",
        help="Environment variable that contains the login Cookie header.",
    )
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--delay-seconds", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument(
        "--product-tags",
        nargs="+",
        default=["未分类"],
        help="Applicable models or business objects, for example: --product-tags GS3 ECO",
    )
    parser.add_argument("--language", default="zh")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output instead of replacing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cookie = os.environ.get(args.cookie_env, "").strip()
    if not cookie:
        raise SystemExit(
            f"Missing authorized Cookie. Set {args.cookie_env} before running."
        )
    pages = crawl(
        args.start_url,
        cookie=cookie,
        max_pages=args.max_pages,
        delay_seconds=args.delay_seconds,
        timeout=args.timeout,
    )
    if not pages:
        raise SystemExit(
            "No pages were crawled. Check whether the Cookie is valid and the start URL is accessible."
        )
    records = to_source_records(
        pages, product_tags=args.product_tags, language=args.language
    )
    if args.append and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        records = [*existing, *records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
