"""Screenshots for the README and landing page, taken with the locally installed Microsoft Edge.

Start the UI first (uv run python -m gspaenli.ui), then:
    uv run --with playwright==1.63.0 python scripts/screenshots.py [url]

Asks real questions, so the GPU wakes up (a few cents, about 2 minutes when cold).
"""

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
ANSWER_TIMEOUT_MS = 300_000
SKIP_TOUR = "try { localStorage.setItem('gspaenli-tour-done', '1') } catch (e) {}"


def wait_for_app(page: Page) -> None:
    page.goto(URL)
    page.wait_for_selector("#gs-personas label", timeout=60_000)
    page.wait_for_timeout(800)


def ask(page: Page, persona: str, question: str) -> None:
    page.locator("#gs-personas label", has_text=persona).click()
    page.wait_for_timeout(500)
    page.locator("#gs-input textarea").fill(question)
    page.locator("#gs-ask").click()
    page.wait_for_function(
        "() => [...document.querySelectorAll('#gs-chat .message-row')]"
        ".some(r => r.innerText.includes('How I answered'))",
        timeout=ANSWER_TIMEOUT_MS,
    )
    page.wait_for_timeout(500)


def open_trace(page: Page) -> None:
    page.locator("#gs-chat").get_by_text("How I answered").last.click()
    page.wait_for_timeout(500)
    page.evaluate(
        "() => { const c = document.querySelector('#gs-chat .bubble-wrap') || "
        "document.querySelector('#gs-chat [role=log]'); if (c) c.scrollTop = c.scrollHeight; }"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")

        # 1. Tour, first visit, desktop
        page = browser.new_page(viewport={"width": 1280, "height": 820}, color_scheme="light")
        wait_for_app(page)
        page.wait_for_selector(".driver-popover", timeout=10_000)
        page.locator(".driver-popover-next-btn").click()
        page.wait_for_timeout(600)
        page.screenshot(path=OUT / "tour.png")
        page.close()

        # 2. Desktop chat with the How I answered panel open
        ctx = browser.new_context(viewport={"width": 1280, "height": 1000}, color_scheme="light")
        ctx.add_init_script(SKIP_TOUR)
        page = ctx.new_page()
        wait_for_app(page)
        ask(page, "Newcomer guide", "What is the minimum health insurance premium in Lausanne for an adult?")
        open_trace(page)
        page.locator("#gs-chat").scroll_into_view_if_needed()
        page.screenshot(path=OUT / "chat.png", full_page=True)
        ctx.close()

        # 3. Mobile, French question, dark mode
        ctx = browser.new_context(
            viewport={"width": 390, "height": 844}, device_scale_factor=2, color_scheme="dark", is_mobile=True
        )
        ctx.add_init_script(SKIP_TOUR)
        page = ctx.new_page()
        wait_for_app(page)
        ask(page, "Bünzli neighbour", "Quand sont les vacances scolaires d'automne dans le canton de Vaud ?")
        page.locator("#gs-chat").scroll_into_view_if_needed()
        page.screenshot(path=OUT / "mobile-fr.png")
        ctx.close()

        browser.close()
    for f in sorted(OUT.glob("*.png")):
        print(f"{f.name}: {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
