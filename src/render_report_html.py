from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _render_inline(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def _render_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell.replace(":", "").replace("-", "")) == set() for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""

    header = rows[0]
    body = rows[1:]
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{_render_inline(cell)}</th>" for cell in header)
    parts.append("</tr></thead>")
    if body:
        parts.append("<tbody>")
        for row in body:
            parts.append("<tr>")
            parts.extend(f"<td>{_render_inline(cell)}</td>" for cell in row)
            parts.append("</tr>")
        parts.append("</tbody>")
    parts.append("</table>")
    return "\n".join(parts)


def render_markdown_subset(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    table_lines: list[str] = []
    in_code = False
    code_lang = ""
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append(f"<p>{_render_inline(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            output.append("<ul>")
            output.extend(f"<li>{_render_inline(item)}</li>" for item in list_items)
            output.append("</ul>")
            list_items = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            rendered = _render_table(table_lines)
            if rendered:
                output.append(rendered)
            table_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                output.append(
                    f'<pre><code class="language-{html.escape(code_lang)}">{html.escape(chr(10).join(code_lines))}</code></pre>'
                )
                in_code = False
                code_lang = ""
                code_lines = []
            else:
                flush_paragraph()
                flush_list()
                flush_table()
                in_code = True
                code_lang = stripped.removeprefix("```").strip()
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            flush_list()
            table_lines.append(stripped)
            continue
        flush_table()

        if stripped.startswith("#"):
            flush_paragraph()
            flush_list()
            level = min(len(stripped) - len(stripped.lstrip("#")), 4)
            text = stripped[level:].strip()
            output.append(f"<h{level}>{_render_inline(text)}</h{level}>")
        elif stripped.startswith("- "):
            flush_paragraph()
            list_items.append(stripped[2:].strip())
        else:
            flush_list()
            paragraph.append(stripped)

    flush_paragraph()
    flush_list()
    flush_table()
    return "\n".join(output)


def render_html(markdown_text: str, title: str) -> str:
    body = render_markdown_subset(markdown_text)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    @page {{ size: A4; margin: 18mm; }}
    body {{
      color: #17202a;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 10.5pt;
      line-height: 1.45;
      margin: 0 auto;
      max-width: 980px;
    }}
    h1 {{ color: #10243f; font-size: 25pt; margin: 0 0 12px; }}
    h2 {{ border-bottom: 1px solid #d7dee8; color: #173a63; font-size: 16pt; margin-top: 28px; padding-bottom: 5px; }}
    h3 {{ color: #245178; font-size: 12.5pt; margin-top: 18px; }}
    table {{ border-collapse: collapse; font-size: 8.5pt; margin: 12px 0; width: 100%; }}
    th, td {{ border: 1px solid #ccd5df; padding: 5px 6px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; color: #10243f; }}
    code {{ background: #f3f6f9; border-radius: 3px; font-family: Consolas, monospace; padding: 1px 3px; }}
    pre {{ background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 4px; overflow-x: auto; padding: 10px; }}
    pre code {{ background: transparent; padding: 0; }}
    ul {{ margin-top: 6px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the final Markdown report to a PDF-ready HTML file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root() / "reports" / "final_project_report_draft.md",
        help="Input Markdown report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "reports" / "final_project_report.html",
        help="Output HTML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown_text = args.input.read_text(encoding="utf-8")
    html_text = render_html(markdown_text, title="Predictive Maintenance and RUL Forecasting Platform")
    args.output.write_text(html_text, encoding="utf-8")
    print(f"HTML report written to: {args.output}")


if __name__ == "__main__":
    main()
