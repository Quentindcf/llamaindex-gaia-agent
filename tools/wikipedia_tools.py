"""Wikipedia tool spec."""

from typing import Any, Dict

from llama_index.core.tools.tool_spec.base import BaseToolSpec
import requests
from markdownify import markdownify as md
from llama_index.core import Document
from bs4 import BeautifulSoup
import regex

def clean_html_keep_text_and_tables(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    if soup.body is None:
        return "[Error: no <body> tag found in Wikipedia HTML]"

    # Remove non-informative tags
    for tag in soup.find_all(["img", "figure", "audio", "video", "style", "script", "nav", "footer", "aside", "sup", "ol", "ul"]):
        try:
            tag.decompose()
        except Exception:
            continue

    # Flatten links to text
    for a in soup.find_all("a"):
        try:
            a.replace_with(a.get_text())
        except Exception:
            continue

    # Remove layout-only or infobox tables
    for table in soup.find_all("table"):
        try:
            classes = table.get("class", [])
            if not isinstance(classes, list):
                classes = [classes]  # just in case class is a string
            if "infobox" in classes or "vertical-navbox" in classes:
                table.decompose()
        except Exception:
            continue

    # Remove content after certain headings
    cut_headers = ["References", "External links", "See also", "Further reading", "Navigation menu", "Notes"]
    for header in soup.find_all(["h2", "h3", "h4"]):
        try:
            if header.get_text().strip().strip("[]") in cut_headers:
                for el in header.find_all_next():
                    el.decompose()
                header.decompose()
        except Exception:
            continue

    return str(soup)

def convert_html_to_text_and_tables(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    if soup.body is None:
        return "[Error: no <body> tag found in Wikipedia HTML]"
    
    output_lines = []

    for elem in soup.body.descendants:
        if elem.name == "table":
            markdown_table = md(str(elem), heading_style="ATX")
            output_lines.append(markdown_table.strip())
        elif elem.name in ["p", "h1", "h2", "h3", "h4"]:
            text = elem.get_text(strip=True)
            if text:
                output_lines.append(text)

    return "\n\n".join(output_lines)

def get_markdown_doc_content(page) -> str:
    try:
        response = requests.get(page.url, headers={"User-Agent": "WikipediaMarkdownBot/1.0"})
        if response.status_code != 200:
            return f"Couldn't fetch Wikipedia page {page.url}"

        html = clean_html_keep_text_and_tables(response.text)
        clean_text = convert_html_to_text_and_tables(html)
        return clean_text
    except Exception as e:
        return f"[Wikipedia content error: {e}]"


class WikipediaToolSpec(BaseToolSpec):
    """
    Specifies two tools for querying information from Wikipedia.
    """

    spec_functions = ["load_data", "search_data"]

    def load_data(
        self, page: str, lang: str = "en", **load_kwargs: Dict[str, Any]
    ) -> str:
        """
        Retrieve a Wikipedia page. Useful for learning about a particular concept that isn't private information.

        Args:
            page (str): Title of the page to read.
            lang (str): Language of Wikipedia to read. (default: English)
        """
        import wikipedia

        wikipedia.set_lang(lang)
        try:
            wikipedia_page = wikipedia.page(page, **load_kwargs, auto_suggest=False)
        except wikipedia.PageError:
            wikipedia_page = self.search_data(page, lang)
            if wikipedia_page == "No search results.":
                return "No search results."
            return "Unable to load page. This was the closest:"+get_markdown_doc_content(wikipedia_page)
        return get_markdown_doc_content(wikipedia_page)

    def search_data(self, query: str, lang: str = "en") -> str:
        """
        Search Wikipedia for a page related to the given query.
        Use this tool when `load_data` returns no results.

        Args:
            query (str): the string to search for
        """
        import wikipedia

        pages = wikipedia.search(query)
        if len(pages) == 0:
            return "No search results."
        return self.load_data(pages[0], lang)
    

