from . import config


def render_sinhala_pdf(paragraphs: list[str]) -> bytes:
    try:
        from weasyprint import HTML
    except ImportError as e:
        raise RuntimeError(
            "weasyprint is not installed. Run: pip install weasyprint "
            "(also requires system libraries -- see README)."
        ) from e

    if not config.SINHALA_FONT_PATH.exists():
        raise RuntimeError(
            f"Sinhala font not found at {config.SINHALA_FONT_PATH}. "
            "Download 'Noto Sans Sinhala' from fonts.google.com and place "
            "NotoSansSinhala-Regular.ttf in the fonts/ directory."
        )

    body_html = "".join(f"<p>{para}</p>" for para in paragraphs if para.strip())

    html_content = f"""
    <!DOCTYPE html>
    <html lang="si">
    <head>
    <meta charset="UTF-8">
    <style>
    @font-face {{
        font-family: 'Sinhala';
        src: url('{config.SINHALA_FONT_PATH.as_uri()}');
    }}
    body {{
        font-family: 'Sinhala';
        font-size: 16px;
        line-height: 1.8;
        margin: 40px;
    }}
    h1 {{ font-size: 24px; }}
    p {{ margin-bottom: 12px; text-align: justify; }}
    </style>
    </head>
    <body>
    <h1>සිංහල පරිවර්තනය</h1>
    {body_html}
    </body>
    </html>
    """

    return HTML(string=html_content).write_pdf()
