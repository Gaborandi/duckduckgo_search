# duckduckgo_search.py (replace entire file)

from __future__ import annotations
import logging
import os
import warnings
from datetime import datetime, timezone
from functools import cached_property
from itertools import cycle
from random import choice, shuffle
from time import sleep, time
from types import TracebackType
from typing import cast
import primp
from lxml.etree import _Element
from lxml.html import HTMLParser as LHTMLParser
from lxml.html import document_fromstring
from .exceptions import ConversationLimitException, DuckDuckGoSearchException, RatelimitException, TimeoutException
from .utils import (
    _expand_proxy_tb_alias,
    _extract_vqd,
    _normalize,
    _normalize_url,
    json_loads,
)

logger = logging.getLogger("duckduckgo_search.DDGS")

class DDGS:
    _impersonates = (
        "chrome_100","chrome_101","chrome_104","chrome_105","chrome_106","chrome_107","chrome_108","chrome_109",
        "chrome_114","chrome_116","chrome_117","chrome_118","chrome_119","chrome_120","chrome_123","chrome_124",
        "chrome_126","chrome_127","chrome_128","chrome_129","chrome_130","chrome_131","safari_ios_16.5",
        "safari_ios_17.2","safari_ios_17.4.1","safari_ios_18.1.1","safari_15.3","safari_15.5","safari_15.6.1",
        "safari_16","safari_16.5","safari_17.0","safari_17.2.1","safari_17.4.1","safari_17.5","safari_18",
        "safari_18.2","safari_ipad_18","edge_101","edge_122","edge_127","edge_131","firefox_109","firefox_133",
    )

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
        proxies: dict[str, str] | str | None = None,
        timeout: int | None = 10,
        verify: bool = True,
    ) -> None:
        ddgs_proxy: str | None = os.environ.get("DDGS_PROXY")
        self.proxy: str | None = ddgs_proxy if ddgs_proxy else _expand_proxy_tb_alias(proxy)
        if not proxy and proxies:
            warnings.warn("'proxies' is deprecated, use 'proxy' instead.", stacklevel=1)
            if isinstance(proxies, dict):
                self.proxy = proxies.get("http") or proxies.get("https")
            else:
                self.proxy = proxies
        self.headers = headers if headers else {}
        self.headers["Referer"] = "https://duckduckgo.com/"
        self.client = primp.Client(
            headers=self.headers,
            proxy=self.proxy,
            timeout=timeout,
            cookie_store=True,
            referer=True,
            impersonate=choice(self._impersonates),
            follow_redirects=False,
            verify=verify,
        )
        self._chat_messages: list[dict[str, str]] = []
        self._chat_tokens_count = 0
        self._chat_vqd: str = ""
        self.sleep_timestamp = 0.0
        self._chat_docs_text = ""

    def __enter__(self) -> DDGS:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        pass

    @cached_property
    def parser(self) -> LHTMLParser:
        return LHTMLParser(remove_blank_text=True, remove_comments=True, remove_pis=True, collect_ids=False)

    def _sleep(self, sleeptime: float = 0.75) -> None:
        delay = 0.0 if not self.sleep_timestamp else 0.0 if time() - self.sleep_timestamp >= 20 else sleeptime
        self.sleep_timestamp = time()
        sleep(delay)

    def _get_url(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        content: bytes | None = None,
        data: dict[str, str] | bytes | None = None,
        cookies: dict[str, str] | None = None,
    ) -> bytes:
        self._sleep()
        try:
            resp = self.client.request(method, url, params=params, content=content, data=data, cookies=cookies)
        except Exception as ex:
            if "time" in str(ex).lower():
                raise TimeoutException(f"{url} {type(ex).__name__}: {ex}") from ex
            raise DuckDuckGoSearchException(f"{url} {type(ex).__name__}: {ex}") from ex
        if resp.status_code == 200:
            return cast(bytes, resp.content)
        elif resp.status_code in (202, 301, 403):
            raise RatelimitException(f"{resp.url} {resp.status_code} Ratelimit")
        raise DuckDuckGoSearchException(f"{resp.url} return None")

    def _get_vqd(self, keywords: str) -> str:
        resp_content = self._get_url("GET", "https://duckduckgo.com", params={"q": keywords})
        return _extract_vqd(resp_content, keywords)

    def chat(self, keywords: str, model: str = "gpt-4o-mini", timeout: int = 30) -> str:
        models_deprecated = {
            "gpt-3.5": "gpt-4o-mini",
            "llama-3-70b": "llama-3.1-70b",
        }
        if model in models_deprecated:
            model = models_deprecated[model]
        models = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "gpt-4o-mini": "gpt-4o-mini",
            "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }
        if not self._chat_vqd:
            r = self.client.get("https://duckduckgo.com/duckchat/v1/status", headers={"x-vqd-accept": "1"})
            self._chat_vqd = r.headers.get("x-vqd-4", "")
        content = keywords
        if self._chat_docs_text:
            content += "\n\nAdditional Document Context:\n" + self._chat_docs_text
        self._chat_messages.append({"role": "user", "content": content})
        if len(keywords) >= 4:
            self._chat_tokens_count += len(keywords) // 4
        else:
            self._chat_tokens_count += 1
        json_data = {
            "model": models[model],
            "messages": self._chat_messages,
        }
        resp = self.client.post(
            "https://duckduckgo.com/duckchat/v1/chat",
            headers={"x-vqd-4": self._chat_vqd},
            json=json_data,
            timeout=timeout,
        )
        self._chat_vqd = resp.headers.get("x-vqd-4", "")
        data_text = resp.text.rstrip("[DONE]LIMT_CVRSA\n")
        data_split = data_text.split("data:")
        combined = []
        for line in data_split:
            x = line.strip()
            if x:
                combined.append(x)
        data_str = "[" + ",".join(combined) + "]"
        data = json_loads(data_str)
        results = []
        for x in data:
            if x.get("action") == "error":
                err_message = x.get("type", "")
                if x.get("status") == 429:
                    if err_message == "ERR_CONVERSATION_LIMIT":
                        raise ConversationLimitException(err_message)
                    else:
                        raise RatelimitException(err_message)
                raise DuckDuckGoSearchException(err_message)
            elif x.get("message"):
                results.append(x["message"])
        result = "".join(results)
        self._chat_messages.append({"role": "assistant", "content": result})
        self._chat_tokens_count += len(results)
        return result

    def text(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        backend: str = "auto",
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        if backend == "api":
            warnings.warn("'api' backend is deprecated, using backend='auto'", stacklevel=2)
            backend = "auto"
        backends = ["html", "lite", "ecosia"] if backend == "auto" else [backend]
        shuffle(backends)
        results = []
        err = None
        for b in backends:
            try:
                if b == "html":
                    results = self._text_html(keywords, region, timelimit, max_results)
                elif b == "lite":
                    results = self._text_lite(keywords, region, timelimit, max_results)
                elif b == "ecosia":
                    results = self._text_ecosia(keywords, region, safesearch, max_results)
                return results
            except Exception as ex:
                err = ex
        raise DuckDuckGoSearchException(err)

    def _text_html(
        self,
        keywords: str,
        region: str = "wt-wt",
        timelimit: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        payload = {
            "q": keywords,
            "s": "0",
            "o": "json",
            "api": "d.js",
            "vqd": "",
            "kl": region,
            "bing_market": region,
        }
        if timelimit:
            payload["df"] = timelimit
        cache = set()
        results: list[dict[str, str]] = []
        for _ in range(5):
            resp_content = self._get_url("POST", "https://html.duckduckgo.com/html", data=payload)
            if b"No  results." in resp_content:
                return results
            tree = document_fromstring(resp_content, self.parser)
            elements = tree.xpath("//div[h2]")
            if not isinstance(elements, list):
                return results
            for e in elements:
                if isinstance(e, _Element):
                    hrefxpath = e.xpath("./a/@href")
                    href = None
                    if hrefxpath and isinstance(hrefxpath, list):
                        href = str(hrefxpath[0])
                    if href and href not in cache and not href.startswith(
                        ("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")
                    ):
                        cache.add(href)
                        titlexpath = e.xpath("./h2/a/text()")
                        title = ""
                        if titlexpath and isinstance(titlexpath, list):
                            title = str(titlexpath[0])
                        bodyxpath = e.xpath("./a//text()")
                        body = ""
                        if bodyxpath and isinstance(bodyxpath, list):
                            body = "".join(str(x) for x in bodyxpath)
                        results.append(
                            {
                                "title": _normalize(title),
                                "href": _normalize_url(href),
                                "body": _normalize(body),
                            }
                        )
                        if max_results and len(results) >= max_results:
                            return results
            npx = tree.xpath('.//div[@class="nav-link"]')
            if not npx or not max_results:
                return results
            next_page = npx[-1] if isinstance(npx, list) else None
            if isinstance(next_page, _Element):
                names = next_page.xpath('.//input[@type="hidden"]/@name')
                values = next_page.xpath('.//input[@type="hidden"]/@value')
                if isinstance(names, list) and isinstance(values, list):
                    payload = {str(n): str(v) for n, v in zip(names, values)}
        return results

    def _text_lite(
        self,
        keywords: str,
        region: str = "wt-wt",
        timelimit: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        payload = {
            "q": keywords,
            "s": "0",
            "o": "json",
            "api": "d.js",
            "vqd": "",
            "kl": region,
            "bing_market": region,
        }
        if timelimit:
            payload["df"] = timelimit
        cache = set()
        results: list[dict[str, str]] = []
        for _ in range(5):
            resp_content = self._get_url("POST", "https://lite.duckduckgo.com/lite/", data=payload)
            if b"No more results." in resp_content:
                return results
            tree = document_fromstring(resp_content, self.parser)
            elements = tree.xpath("//table[last()]//tr")
            if not isinstance(elements, list):
                return results
            data_cycle = zip(cycle(range(1, 5)), elements)
            for i, e in data_cycle:
                if isinstance(e, _Element):
                    if i == 1:
                        hrefxpath = e.xpath(".//a//@href")
                        href = None
                        if hrefxpath and isinstance(hrefxpath, list):
                            href = str(hrefxpath[0])
                        if (
                            href is None
                            or href in cache
                            or href.startswith(
                                ("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")
                            )
                        ):
                            for _ in range(3):
                                next(data_cycle, None)
                        else:
                            cache.add(href)
                            titlexpath = e.xpath(".//a//text()")
                            title = ""
                            if titlexpath and isinstance(titlexpath, list):
                                title = str(titlexpath[0])
                    elif i == 2:
                        bodyxpath = e.xpath(".//td[@class='result-snippet']//text()")
                        body = ""
                        if bodyxpath and isinstance(bodyxpath, list):
                            body = "".join(str(x) for x in bodyxpath).strip()
                        if href:
                            results.append(
                                {
                                    "title": _normalize(title),
                                    "href": _normalize_url(href),
                                    "body": _normalize(body),
                                }
                            )
                            if max_results and len(results) >= max_results:
                                return results
            next_page_s = tree.xpath("//form[./input[contains(@value, 'ext')]]/input[@name='s']/@value")
            if not next_page_s or not max_results:
                return results
            if isinstance(next_page_s, list):
                payload["s"] = str(next_page_s[0])
        return results

    def _text_ecosia(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        payload = {"q": keywords}
        cookies = {
            "a": "0",
            "as": "0",
            "cs": "1",
            "dt": "pc",
            "f": "y" if safesearch == "on" else "n" if safesearch == "off" else "i",
            "fr": "0",
            "fs": "1",
            "l": "en",
            "lt": f"{int(time() * 1000)}",
            "mc": f"{region[3:]}-{region[:2]}",
            "nf": "0",
            "nt": "0",
            "pz": "0",
            "t": "6",
            "tt": "",
            "tu": "auto",
            "wu": "auto",
            "ma": "1",
        }
        cache = set()
        results: list[dict[str, str]] = []
        for _ in range(5):
            resp_content = self._get_url("GET", "https://www.ecosia.org/search", params=payload, cookies=cookies)
            if b"Unfortunately we didn\xe2\x80\x99t find any results for" in resp_content:
                return results
            tree = document_fromstring(resp_content, self.parser)
            elements = tree.xpath("//div[@class='result__body']")
            if not isinstance(elements, list):
                return results
            for e in elements:
                if isinstance(e, _Element):
                    hrefxpath = e.xpath(".//div[@class='result__title']/a/@href")
                    href = None
                    if hrefxpath and isinstance(hrefxpath, list):
                        href = str(hrefxpath[0])
                    if href and href not in cache:
                        cache.add(href)
                        titlexpath = e.xpath(".//div[@class='result__title']/a/h2/text()")
                        title = ""
                        if titlexpath and isinstance(titlexpath, list):
                            title = str(titlexpath[0])
                        bodyxpath = e.xpath(".//div[@class='result__description']//text()")
                        body = ""
                        if bodyxpath and isinstance(bodyxpath, list):
                            body = "".join(str(x) for x in bodyxpath)
                        results.append(
                            {
                                "title": _normalize(title.strip()),
                                "href": _normalize_url(href),
                                "body": _normalize(body.strip()),
                            }
                        )
                        if max_results and len(results) >= max_results:
                            return results
            npx = tree.xpath("//div[contains(@class, 'pagination')]//a[contains(@data-test-id, 'next')]/@href")
            if not npx or not max_results:
                return results
            if isinstance(npx, list):
                payload["p"] = str(npx[-1]).split("p=")[1].split("&")[0]
        return results

    def images(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        size: str | None = None,
        color: str | None = None,
        type_image: str | None = None,
        layout: str | None = None,
        license_image: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        vqd = self._get_vqd(keywords)
        safesearch_base = {"on": "1", "moderate": "1", "off": "-1"}
        tl = f"time:{timelimit}" if timelimit else ""
        sz = f"size:{size}" if size else ""
        co = f"color:{color}" if color else ""
        ty = f"type:{type_image}" if type_image else ""
        la = f"layout:{layout}" if layout else ""
        lic = f"license:{license_image}" if license_image else ""
        payload = {
            "l": region,
            "o": "json",
            "q": keywords,
            "vqd": vqd,
            "f": f"{tl},{sz},{co},{ty},{la},{lic}",
            "p": safesearch_base[safesearch.lower()],
        }
        cache = set()
        results: list[dict[str, str]] = []
        for _ in range(5):
            resp_content = self._get_url("GET", "https://duckduckgo.com/i.js", params=payload)
            resp_json = json_loads(resp_content)
            page_data = resp_json.get("results", [])
            for row in page_data:
                image_url = row.get("image")
                if image_url and image_url not in cache:
                    cache.add(image_url)
                    result = {
                        "title": row["title"],
                        "image": _normalize_url(image_url),
                        "thumbnail": _normalize_url(row["thumbnail"]),
                        "url": _normalize_url(row["url"]),
                        "height": row["height"],
                        "width": row["width"],
                        "source": row["source"],
                    }
                    results.append(result)
                    if max_results and len(results) >= max_results:
                        return results
            next_val = resp_json.get("next")
            if not next_val or not max_results:
                return results
            payload["s"] = next_val.split("s=")[-1].split("&")[0]
        return results

    def videos(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        resolution: str | None = None,
        duration: str | None = None,
        license_videos: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        vqd = self._get_vqd(keywords)
        safesearch_base = {"on": "1", "moderate": "-1", "off": "-2"}
        tl = f"publishedAfter:{timelimit}" if timelimit else ""
        rs = f"videoDefinition:{resolution}" if resolution else ""
        du = f"videoDuration:{duration}" if duration else ""
        lv = f"videoLicense:{license_videos}" if license_videos else ""
        payload = {
            "l": region,
            "o": "json",
            "q": keywords,
            "vqd": vqd,
            "f": f"{tl},{rs},{du},{lv}",
            "p": safesearch_base[safesearch.lower()],
        }
        cache = set()
        results: list[dict[str, str]] = []
        for _ in range(8):
            resp_content = self._get_url("GET", "https://duckduckgo.com/v.js", params=payload)
            resp_json = json_loads(resp_content)
            page_data = resp_json.get("results", [])
            for row in page_data:
                if row["content"] not in cache:
                    cache.add(row["content"])
                    results.append(row)
                    if max_results and len(results) >= max_results:
                        return results
            next_val = resp_json.get("next")
            if not next_val or not max_results:
                return results
            payload["s"] = next_val.split("s=")[-1].split("&")[0]
        return results

    def news(
        self,
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        vqd = self._get_vqd(keywords)
        safesearch_base = {"on": "1", "moderate": "-1", "off": "-2"}
        payload = {
            "l": region,
            "o": "json",
            "noamp": "1",
            "q": keywords,
            "vqd": vqd,
            "p": safesearch_base[safesearch.lower()],
        }
        if timelimit:
            payload["df"] = timelimit
        cache = set()
        results: list[dict[str, str]] = []
        for _ in range(5):
            resp_content = self._get_url("GET", "https://duckduckgo.com/news.js", params=payload)
            resp_json = json_loads(resp_content)
            page_data = resp_json.get("results", [])
            for row in page_data:
                if row["url"] not in cache:
                    cache.add(row["url"])
                    image_url = row.get("image", None)
                    result = {
                        "date": datetime.fromtimestamp(row["date"], timezone.utc).isoformat(),
                        "title": row["title"],
                        "body": _normalize(row["excerpt"]),
                        "url": _normalize_url(row["url"]),
                        "image": _normalize_url(image_url),
                        "source": row["source"],
                    }
                    results.append(result)
                    if max_results and len(results) >= max_results:
                        return results
            next_val = resp_json.get("next")
            if not next_val or not max_results:
                return results
            payload["s"] = next_val.split("s=")[-1].split("&")[0]
        return results

    def add_docs_text(self, text: str) -> None:
        self._chat_docs_text += "\n" + text
