---
name: frontend
description: Building the page's layout and styles. Loaded for frontend tasks.
---

# Frontend: the layout and the styles

You own `site/template.html` and `site/styles.css`.

## The page

- One `index` page with, in order: a header with `{{company_name}}` and `{{tagline}}`, an
  about section (`{{about}}`), the team (`<ul>{{team}}</ul>`), and a contact section with a
  `mailto:` link built from `{{contact_email}}`.
- Semantic HTML5: `header`, `main`, `section` with an `id` per section, `footer`. One `h1`.
- `<!doctype html>`, `<html lang="en">`, `<meta charset="utf-8">`, a viewport meta, a
  `<title>`, and `<link rel="stylesheet" href="styles.css">`.
- Accessible: headings in order, visible focus styles, text contrast of at least 4.5:1,
  nothing that only works with a mouse.

## Styles

- Plain CSS, no frameworks. Mobile first; one breakpoint is plenty.
- Custom properties for colors and spacing at the top of `styles.css`.

## Testing it

`tests/test_page.py` parses the template with `html.parser` and checks the structure: the
sections exist with their ids, there is one `h1`, the stylesheet is linked, and the
placeholders are exactly the contract's keys. Test against the template itself, with a
small sample content dict, not against the real `content/site.json`.
