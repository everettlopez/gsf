---
name: backend
description: The page's content data and the build that fills the template. Loaded for backend tasks.
---

# Backend: the content and the build

You own `content/site.json` and `build.py`.

## The content

- `content/site.json` holds every piece of text on the page, keyed by placeholder name:
  `company_name`, `tagline`, `about`, `contact_email`, and `team` (a list of
  `{"name", "role"}`).
- Real, short copy for a small company; no lorem ipsum.

## The build

- `python build.py` reads `site/template.html` and `content/site.json`, escapes every text
  value (`html.escape`), renders `team` as `<li><strong>name</strong>, role</li>` items,
  fills each `{{key}}`, copies `site/styles.css`, and writes `dist/index.html` and
  `dist/styles.css`.
- It fails loudly (non-zero exit, a clear message) on a placeholder with no content, a
  content key the template doesn't use, or a `{{` left in the output.
- Keep `build.py` importable: the work in functions (`load_content`, `render`, `build`), the
  command line in `if __name__ == "__main__":`.

## Testing it

`tests/test_build.py` tests `render` with a small inline template and content: escaping,
the team list, and each failure case. A test that runs the full build uses a temporary
directory, so it never depends on the frontend's template being finished.
