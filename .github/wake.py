"""Opens each Streamlit app in a real browser and wakes it up if it has gone to sleep.
Run by .github/workflows/keep-awake.yml every few hours."""
import sys
from playwright.sync_api import sync_playwright

APPS = [
    "https://court-finish.streamlit.app/",
    "https://caption-checker.streamlit.app/",
    "https://tournament-report.streamlit.app/",
]

failed = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    for url in APPS:
        page = browser.new_page()
        try:
            page.goto(url, timeout=90_000)
            page.wait_for_timeout(8_000)
            wake = page.get_by_role("button", name="Yes, get this app back up!")
            if wake.count():
                wake.click()
                print(f"{url}: was asleep, waking up")
                page.wait_for_timeout(90_000)  # give it time to boot
            else:
                print(f"{url}: awake")
            # the app itself runs inside an iframe; wait until something has rendered there
            frame = page.frame_locator("iframe[title='streamlitApp']")
            frame.locator("h1").first.wait_for(timeout=120_000)
            print(f"{url}: OK, title: {frame.locator('h1').first.inner_text()[:60]}")
        except Exception as e:
            failed.append(url)
            print(f"{url}: FAILED: {e}".splitlines()[0])
        finally:
            page.close()
    browser.close()

sys.exit(1 if failed else 0)
