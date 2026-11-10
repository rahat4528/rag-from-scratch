"""
Helper module to build standard-compliant Jupyter notebooks for the RAG course.
Ensures every notebook strictly follows sections a through i of the template.
"""

from pathlib import Path
from typing import Any, List, Optional
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def create_course_notebook(
    title: str,
    promise: str,
    objectives: list[str],
    prerequisites: list[str],
    setup_code: str,
    concept_md: str,
    steps: list[dict[str, str]],
    try_it_yourself_md: str,
    solutions_code: str,
    common_mistakes: list[str],
    recap: list[str],
    next_teaser: str,
    video_chapters: list[dict[str, str]],
    output_path: str | Path,
) -> None:
    """Build and save a course notebook following the global template rules."""
    nb = new_notebook()
    cells = []

    # a & b: Title, Promise, Learning Objectives & Prerequisites
    obj_bullets = "\n".join(f"- {obj}" for obj in objectives)
    prereq_bullets = "\n".join(f"- {p}" for p in prerequisites) if prerequisites else "- None! Motivated beginner with basic Python."
    header_md = f"""# {title}

> **Promise:** {promise}

### Learning Objectives
{obj_bullets}

### Prerequisites
{prereq_bullets}
"""
    cells.append(new_markdown_cell(header_md))

    # c: Setup cell
    setup_banner_md = "## 1. Setup & Environment\nLoad environment variables and initialize our provider-agnostic model and embedding helper."
    cells.append(new_markdown_cell(setup_banner_md))
    cells.append(new_code_cell(setup_code.strip()))

    # d: Concept section
    cells.append(new_markdown_cell(f"## 2. Concept & Intuition\n\n{concept_md.strip()}"))

    # e: Step-by-step build
    cells.append(new_markdown_cell("## 3. Step-by-Step Hands-On Build"))
    for idx, step in enumerate(steps, start=1):
        if step.get("before"):
            cells.append(new_markdown_cell(f"### Step {idx}: {step.get('title', '')}\n{step['before'].strip()}"))
        if step.get("code"):
            cells.append(new_code_cell(step["code"].strip()))
        if step.get("after"):
            cells.append(new_markdown_cell(f"> **What just happened?**\n> {step['after'].strip()}"))

    # f: Try it yourself
    cells.append(new_markdown_cell(f"## 4. Try It Yourself!\n\n{try_it_yourself_md.strip()}"))
    cells.append(new_code_cell(solutions_code.strip()))

    # g: Common Mistakes & Recap
    mistakes_bullets = "\n".join(f"- **{m.split(':')[0]}:**{':'.join(m.split(':')[1:])}" if ":" in m else f"- {m}" for m in common_mistakes)
    recap_bullets = "\n".join(f"- {r}" for r in recap)
    review_md = f"""## 5. Common Mistakes & Pitfalls
{mistakes_bullets}

---

## 6. Notebook Recap
{recap_bullets}
"""
    cells.append(new_markdown_cell(review_md))

    # h: Next Notebook Teaser
    cells.append(new_markdown_cell(f"## 7. What's Next?\n\n{next_teaser.strip()}"))

    # i: Video Chapter Outline
    chapters_content = "\n".join(
        f"- **{ch['timestamp']} - {ch['title']}:** {ch['talking_points']}"
        for ch in video_chapters
    )
    video_md = f"""## 8. Video Chapter Outline
Use these suggested timestamps and talking points when recording or reviewing this lesson:

{chapters_content}
"""
    cells.append(new_markdown_cell(video_md))

    nb.cells = cells
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.13"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Created notebook: {out_file.name}")
