# cli.py

import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import click
import primp

from .duckduckgo_search import DDGS
from .utils import _expand_proxy_tb_alias, json_dumps, json_loads, parse_document
from .version import __version__

logger = logging.getLogger(__name__)
COLORS = {
    0: "black",
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "magenta",
    6: "cyan",
    7: "bright_black",
    8: "bright_red",
    9: "bright_green",
    10: "bright_yellow",
    11: "bright_blue",
    12: "bright_magenta",
    13: "bright_cyan",
    14: "white",
    15: "bright_white",
}

def _save_data(keywords, data, function_name, filename):
    temp = filename
    if temp and temp.endswith((".csv", ".json")):
        pass
    else:
        temp = None
    if temp:
        fbase, ext = temp.rsplit(".", 1)
    else:
        fbase = None
        ext = None
    if fbase is None:
        fbase = f"{function_name}_{keywords}_{datetime.now():%Y%m%d_%H%M%S}"
    if ext == "csv":
        _save_csv(f"{fbase}.{ext}", data)
    elif ext == "json":
        _save_json(f"{fbase}.{ext}", data)

def _save_json(jsonfile, data):
    with open(jsonfile, "w", encoding="utf-8") as file:
        file.write(json_dumps(data))

def _save_csv(csvfile, data):
    with open(csvfile, "w", newline="", encoding="utf-8") as file:
        if data:
            headers = data[0].keys()
            writer = csv.DictWriter(file, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(data)

def _print_data(data):
    if data:
        for i, e in enumerate(data, start=1):
            click.secho(f"{i}.\t    {'=' * 78}", bg="black", fg="white")
            c = 0
            for k, v in e.items():
                if v:
                    c += 1
                    text = click.wrap_text(
                        f"{v}",
                        width=300 if k in ("content","href","image","source","thumbnail","url") else 78,
                        initial_indent="",
                        subsequent_indent=" " * 12,
                        preserve_paragraphs=True
                    )
                    click.secho(f"{k:<12}{text}", bg="black", fg=COLORS[c], overline=True)
            input()

def _sanitize_keywords(keywords):
    r = (
        keywords.replace("filetype", "")
        .replace(":", "")
        .replace('"', "'")
        .replace("site", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
    )
    return r

def _download_file(url, dir_path, filename, proxy, verify):
    try:
        resp = primp.Client(proxy=proxy, impersonate="chrome_131", timeout=10, verify=verify).get(url)
        if resp.status_code == 200:
            with open(os.path.join(dir_path, filename[:200]), "wb") as file:
                file.write(resp.content)
    except Exception:
        pass

def _download_results(keywords, results, function_name, proxy=None, threads=None, verify=True, pathname=None):
    path = pathname if pathname else f"{function_name}_{keywords}_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(path, exist_ok=True)
    t = 10 if threads is None else threads
    with ThreadPoolExecutor(max_workers=t) as executor:
        futures = []
        for i, res in enumerate(results, start=1):
            if function_name == "images":
                url = res["image"]
            else:
                url = res["href"]
            filename = unquote(url.split("/")[-1].split("?")[0])
            f = executor.submit(_download_file, url, path, f"{i}_{filename}", proxy, verify)
            futures.append(f)
        with click.progressbar(length=len(futures), label="Downloading", show_percent=True, show_pos=True, width=50) as bar:
            for future in as_completed(futures):
                future.result()
                bar.update(1)

@click.group(chain=True)
def cli():
    pass

def safe_entry_point():
    try:
        cli()
    except Exception as ex:
        click.echo(f"{type(ex).__name__}: {ex}")

@cli.command()
def version():
    print(__version__)
    return __version__

@cli.command()
@click.option("-l", "--load", is_flag=True, default=False)
@click.option("-p", "--proxy")
@click.option("-ml", "--multiline", is_flag=True, default=False)
@click.option("-t", "--timeout", default=30)
@click.option("-v", "--verify", default=True)
@click.option(
    "-m",
    "--model",
    prompt="[1]: gpt-4o-mini\n[2]: claude-3-haiku\n[3]: llama-3.1-70b\n[4]: mixtral-8x7b\n",
    type=click.Choice(["1","2","3","4"]),
    default="1",
)
@click.option("--docs", multiple=True)
def chat(load, proxy, multiline, timeout, verify, model, docs):
    """
    Start an interactive DuckDuckGo chat session.

    Slash Commands (inside chat):
      /docs <filename>   -> parse PDF, DOC, CSV, etc.
      /search <query>    -> do a text/web search
      /images <query>    -> do an images search
      /videos <query>    -> do a videos search
      /news <query>      -> do a news search

    Then simply type your normal chat messages to the AI.
    """
    client = DDGS(proxy=_expand_proxy_tb_alias(proxy), verify=verify)
    model = ["gpt-4o-mini","claude-3-haiku","llama-3.1-70b","mixtral-8x7b"][int(model) - 1]
    cache_file = "ddgs_chat_conversation.json"
    # load existing conversation if requested
    if load and Path(cache_file).exists():
        with open(cache_file) as f:
            c = json_loads(f.read())
            client._chat_vqd = c.get("vqd", "")
            client._chat_messages = c.get("messages", [])
            client._chat_tokens_count = c.get("tokens", 0)
            client._chat_docs_text = ""
    # parse documents if provided at command line
    if docs:
        for d in docs:
            text_content = parse_document(d)
            client.add_docs_text(text_content)
    # interactive loop
    while True:
        print(f"{'-'*78}\nYou[model={model} tokens={client._chat_tokens_count}]: ", end="")
        if multiline:
            if sys.platform == "win32":
                print("[multiline, use ctrl+Z to send]")
            else:
                print("[multiline, use ctrl+D to send]")
            user_input = sys.stdin.read()
            print("...")
        else:
            user_input = input().strip()
        if not user_input:
            continue
        # -----------
        # /docs
        if user_input.startswith("/docs "):
            docpath = user_input[6:].strip()
            text_content = parse_document(docpath)
            client.add_docs_text(text_content)
            msg = f"Document {docpath} parsed and added to context."
            click.secho(msg, fg="yellow")
            client._chat_messages.append({"role":"assistant","content":msg})
        # -----------
        # /search
        elif user_input.startswith("/search "):
            query = user_input[8:].strip()
            data = client.text(keywords=query)
            if data:
                results_str = "\n\n".join([f"- {x.get('title','NoTitle')}: {x.get('href','NoHref')}" for x in data])
                client._chat_messages.append({"role":"assistant","content":results_str})
                click.secho(f"Search Results:\n{results_str}", fg="yellow")
            else:
                client._chat_messages.append({"role":"assistant","content":"No results."})
                click.secho("No results.", fg="yellow")
        # -----------
        # /images
        elif user_input.startswith("/images "):
            query = user_input[8:].strip()
            data = client.images(keywords=query)
            if data:
                results_str = "\n\n".join([f"- {x.get('title','NoTitle')}: {x.get('image','NoImage')}" for x in data])
                client._chat_messages.append({"role":"assistant","content":results_str})
                click.secho(f"Images:\n{results_str}", fg="cyan")
            else:
                client._chat_messages.append({"role":"assistant","content":"No image results."})
                click.secho("No image results.", fg="cyan")
        # -----------
        # /videos
        elif user_input.startswith("/videos "):
            query = user_input[8:].strip()
            data = client.videos(keywords=query)
            if data:
                # For videos, we might not always have "title"; adapt if needed
                results_str = "\n\n".join([
                    f"- {x.get('title','NoTitle')} => {x.get('url','NoUrl')}"
                    for x in data
                ])
                client._chat_messages.append({"role":"assistant","content":results_str})
                click.secho(f"Videos:\n{results_str}", fg="magenta")
            else:
                client._chat_messages.append({"role":"assistant","content":"No video results."})
                click.secho("No video results.", fg="magenta")
        # -----------
        # /news
        elif user_input.startswith("/news "):
            query = user_input[6:].strip()
            data = client.news(keywords=query)
            if data:
                results_str = "\n\n".join([
                    f"- {x.get('title','NoTitle')} => {x.get('url','NoUrl')}"
                    for x in data
                ])
                client._chat_messages.append({"role":"assistant","content":results_str})
                click.secho(f"News:\n{results_str}", fg="blue")
            else:
                client._chat_messages.append({"role":"assistant","content":"No news results."})
                click.secho("No news results.", fg="blue")
        # -----------
        # Normal chat
        else:
            resp_answer = client.chat(keywords=user_input, model=model, timeout=timeout)
            click.secho(f"AI: {resp_answer}", fg="green")
            s = {
                "vqd": client._chat_vqd,
                "tokens": client._chat_tokens_count,
                "messages": client._chat_messages
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(json_dumps(s))

@cli.command()
@click.option("-k","--keywords",required=True)
@click.option("-r","--region",default="wt-wt")
@click.option("-s","--safesearch",default="moderate",type=click.Choice(["on","moderate","off"]))
@click.option("-t","--timelimit",type=click.Choice(["d","w","m","y"]))
@click.option("-m","--max_results",type=int)
@click.option("-o","--output")
@click.option("-d","--download",is_flag=True,default=False)
@click.option("-dd","--download-directory")
@click.option("-b","--backend",default="auto",type=click.Choice(["auto","html","lite","ecosia"]))
@click.option("-th","--threads",default=10)
@click.option("-p","--proxy")
@click.option("-v","--verify",default=True)
def text(keywords, region, safesearch, timelimit, backend, output, download, download_directory, threads, max_results, proxy, verify):
    """
    Traditional one-shot text search (separate from the chat).
    """
    data = DDGS(proxy=_expand_proxy_tb_alias(proxy), verify=verify).text(
        keywords=keywords,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        backend=backend,
        max_results=max_results,
    )
    skey = _sanitize_keywords(keywords)
    if output:
        _save_data(skey, data, "text", filename=output)
    if download:
        _download_results(skey, data, "text", proxy=proxy, threads=threads, verify=verify, pathname=download_directory)
    if not output and not download:
        _print_data(data)

@cli.command()
@click.option("-k","--keywords",required=True)
@click.option("-r","--region",default="wt-wt")
@click.option("-s","--safesearch",default="moderate",type=click.Choice(["on","moderate","off"]))
@click.option("-t","--timelimit",type=click.Choice(["Day","Week","Month","Year"]))
@click.option("-size","--size",type=click.Choice(["Small","Medium","Large","Wallpaper"]))
@click.option("-c","--color",type=click.Choice(["color","Monochrome","Red","Orange","Yellow","Green","Blue","Purple","Pink","Brown","Black","Gray","Teal","White"]))
@click.option("-type","--type_image",type=click.Choice(["photo","clipart","gif","transparent","line"]))
@click.option("-l","--layout",type=click.Choice(["Square","Tall","Wide"]))
@click.option("-lic","--license_image",type=click.Choice(["any","Public","Share","ShareCommercially","Modify","ModifyCommercially"]))
@click.option("-m","--max_results",type=int)
@click.option("-o","--output")
@click.option("-d","--download", is_flag=True, default=False)
@click.option("-dd","--download-directory")
@click.option("-th","--threads",default=10)
@click.option("-p","--proxy")
@click.option("-v","--verify",default=True)
def images(
    keywords, region, safesearch, timelimit, size, color, type_image,
    layout, license_image, download, download_directory, threads,
    max_results, output, proxy, verify
):
    """
    Traditional one-shot image search (separate from the chat).
    """
    data = DDGS(proxy=_expand_proxy_tb_alias(proxy), verify=verify).images(
        keywords=keywords,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        size=size,
        color=color,
        type_image=type_image,
        layout=layout,
        license_image=license_image,
        max_results=max_results,
    )
    skey = _sanitize_keywords(keywords)
    if output:
        _save_data(skey, data, "images", filename=output)
    if download:
        _download_results(skey, data, "images", proxy=proxy, threads=threads, verify=verify, pathname=download_directory)
    if not output and not download:
        _print_data(data)

@cli.command()
@click.option("-k","--keywords",required=True)
@click.option("-r","--region",default="wt-wt")
@click.option("-s","--safesearch",default="moderate",type=click.Choice(["on","moderate","off"]))
@click.option("-t","--timelimit",type=click.Choice(["d","w","m"]))
@click.option("-res","--resolution",type=click.Choice(["high","standart"]))
@click.option("-d","--duration",type=click.Choice(["short","medium","long"]))
@click.option("-lic","--license_videos",type=click.Choice(["creativeCommon","youtube"]))
@click.option("-m","--max_results",type=int)
@click.option("-o","--output")
@click.option("-p","--proxy")
@click.option("-v","--verify",default=True)
def videos(keywords, region, safesearch, timelimit, resolution, duration, license_videos, max_results, output, proxy, verify):
    """
    Traditional one-shot video search (separate from the chat).
    """
    data = DDGS(proxy=_expand_proxy_tb_alias(proxy), verify=verify).videos(
        keywords=keywords,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        resolution=resolution,
        duration=duration,
        license_videos=license_videos,
        max_results=max_results,
    )
    skey = _sanitize_keywords(keywords)
    if output:
        _save_data(skey, data, "videos", filename=output)
    else:
        _print_data(data)

@cli.command()
@click.option("-k","--keywords",required=True)
@click.option("-r","--region",default="wt-wt")
@click.option("-s","--safesearch",default="moderate",type=click.Choice(["on","moderate","off"]))
@click.option("-t","--timelimit",type=click.Choice(["d","w","m","y"]))
@click.option("-m","--max_results",type=int)
@click.option("-o","--output")
@click.option("-p","--proxy")
@click.option("-v","--verify",default=True)
def news(keywords, region, safesearch, timelimit, max_results, output, proxy, verify):
    """
    Traditional one-shot news search (separate from the chat).
    """
    data = DDGS(proxy=_expand_proxy_tb_alias(proxy), verify=verify).news(
        keywords=keywords, region=region, safesearch=safesearch, timelimit=timelimit, max_results=max_results
    )
    skey = _sanitize_keywords(keywords)
    if output:
        _save_data(skey, data, "news", filename=output)
    else:
        _print_data(data)

###########################################
@cli.command()
@click.option("-q", "--query", required=True, help="Search query")
@click.option("-p", "--paths", required=True, multiple=True, help="Paths to search")
@click.option("-i", "--index-path", required=True, help="Path to store search index")
@click.option("-t", "--search-type", default="combined", 
              type=click.Choice(["keyword", "semantic", "combined"]),
              help="Type of search to perform")
@click.option("-w", "--include-web", is_flag=True, default=True,
              help="Include web results in search")
@click.option("-l", "--local-results", default=10, help="Maximum local results")
@click.option("-wr", "--web-results", default=10, help="Maximum web results")
@click.option("-s", "--min-score", default=0.1, help="Minimum relevance score")
@click.option("-a", "--analyze", is_flag=True, help="Perform AI analysis of results")
@click.option("--watch", is_flag=True, help="Watch for new results")
@click.option("-o", "--output", help="Save results to file (json/csv)")
@click.option("-p", "--proxy", help="Proxy for web requests")
@click.option("-v", "--verify", default=True, help="Verify SSL certificates")
def unified_search(
    query, paths, index_path, search_type, include_web,
    local_results, web_results, min_score, analyze,
    watch, output, proxy, verify
):
    """Perform unified search across local files and web."""
    try:
        # Initialize unified search
        search = UnifiedSearch(
            monitored_paths=paths,
            index_path=index_path,
            proxy=_expand_proxy_tb_alias(proxy)
        )

        # Index existing files with progress bar
        with click.progressbar(
            length=100, label="Indexing files"
        ) as bar:
            def progress_callback(file_path, current, total):
                bar.update(current / total * 100)
            search.index_existing_files(callback=progress_callback)

        # Perform search
        results = list(search.search(
            query=query,
            search_type=search_type,
            include_web=include_web,
            max_local_results=local_results,
            max_web_results=web_results,
            min_score=min_score
        ))

        if analyze:
            analysis = search.analyze_results(query, results)
            click.echo("\nAI Analysis:")
            click.echo(json_dumps(analysis))

        if output:
            _save_data(query, results, "unified_search", output)
        
        if watch:
            click.echo("\nWatching for new results (Ctrl+C to stop)...")
            def result_callback(result):
                click.echo(f"\nNew result found: {result.title}")
            search.watch_query(query, result_callback)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            _print_results(results)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)

def _print_results(results):
    """Print search results."""
    for i, result in enumerate(results, 1):
        click.echo(f"\n{i}. {'[LOCAL]' if result.source == 'local' else '[WEB]'} "
                  f"{result.title}")
        if result.url:
            click.echo(f"URL: {result.url}")
        if result.file_path:
            click.echo(f"Path: {result.file_path}")
        click.echo(f"Score: {result.score:.2f}")
        if result.highlights:
            click.echo("Highlights:")
            click.echo(result.highlights)
        else:
            click.echo("Content preview:")
            click.echo(result.content[:200] + "...")
        click.echo("-" * 80)

@cli.command()
@click.option("-p", "--paths", required=True, multiple=True,
              help="Paths to monitor")
@click.option("-i", "--index-path", required=True,
              help="Path to store search index")
@click.option("-e", "--extensions", multiple=True,
              help="Allowed file extensions")
@click.option("-s", "--max-size", default=100,
              help="Maximum file size in MB")
def monitor(paths, index_path, extensions, max_size):
    """Monitor directories for changes and maintain search index."""
    try:
        allowed_extensions = set(extensions) if extensions else None
        max_file_size = max_size * 1024 * 1024  # Convert to bytes

        search = UnifiedSearch(
            monitored_paths=paths,
            index_path=index_path,
            allowed_extensions=allowed_extensions,
            max_file_size=max_file_size
        )

        click.echo("Indexing existing files...")
        with click.progressbar(
            length=100, label="Progress"
        ) as bar:
            def progress_callback(file_path, current, total):
                bar.update(current / total * 100)
            search.index_existing_files(callback=progress_callback)

        click.echo("\nMonitoring for changes (Ctrl+C to stop)...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            search.cleanup()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.option("-p", "--path", required=True,
              help="Path to file")
@click.option("-i", "--index-path", required=True,
              help="Path to search index")
def preview(path, index_path):
    """Preview indexed document content."""
    try:
        search = UnifiedSearch(
            monitored_paths=[],  # No monitoring needed
            index_path=index_path
        )

        preview = search.get_document_preview(path)
        if preview:
            click.echo(f"\nDocument: {preview['title']}")
            click.echo(f"\nSummary:\n{preview['summary']}")
            click.echo(f"\nKeywords: {', '.join(preview['keywords'])}")
            click.echo(f"\nContent Preview:\n{preview['content']}")
            click.echo("\nMetadata:")
            click.echo(json_dumps(preview['metadata']))
        else:
            click.echo("Document not found in index.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
###########################################

if __name__ == "__main__":
    cli(prog_name="duckduckgo_search")
