#!/usr/bin/env python3
"""Exercise Dia2's actual chat picker, progressive WebAudio, and interruption."""
import argparse
import json
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8688")
args = parser.parse_args()
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script(r"""
      window.__diaTest = { starts: [], requests: [], cancelled: 0, finished: 0 };
      const start = AudioBufferSourceNode.prototype.start;
      AudioBufferSourceNode.prototype.start = function(...args) {
        __diaTest.starts.push(performance.now()); return start.apply(this, args);
      };
      const fetchOriginal = window.fetch;
      window.fetch = async function(url, options) {
        if (String(url) !== '/api/tts') return fetchOriginal.apply(this, arguments);
        __diaTest.requests.push(JSON.parse(options.body));
        let timer;
        const body = new ReadableStream({
          start(controller) {
            // Odd boundaries verify that a split PCM sample is reassembled.
            controller.enqueue(new Uint8Array(15361));
            timer = setTimeout(() => {
              controller.enqueue(new Uint8Array(15359));
              controller.close(); ++__diaTest.finished;
            }, 1800);
            options.signal.addEventListener('abort', () => {
              clearTimeout(timer); ++__diaTest.cancelled;
              try { controller.error(new DOMException('Aborted', 'AbortError')); } catch {}
            });
          },
          cancel() { clearTimeout(timer); },
        });
        return new Response(body, { headers: {'Content-Type': 'audio/pcm'} });
      };
    """)
    page.goto(args.url, wait_until="domcontentloaded")
    page.locator("#settingsBtn").click()
    page.wait_for_selector('#setTtsVoice option[value="dia2:lou"]', state="attached")
    assert page.locator('#setTtsVoice option[value="dia2:kitt"]').text_content() == "Dia 2 KITT (CPU)"
    assert page.locator('#setTtsVoice option[value="dia2:lou"]').text_content() == "Dia 2 Lou (CPU)"
    assert page.locator("#setDia2Expressive").is_checked()
    page.locator("#setTtsVoice").select_option("dia2:lou")
    page.locator("#ttsTestBtn").click()
    page.wait_for_function("__diaTest.starts.length > 0")
    before_completion = page.evaluate("__diaTest.finished")
    assert before_completion == 0, "Browser buffered the whole utterance"
    page.wait_for_function("__diaTest.finished === 1")
    page.wait_for_timeout(500)
    page.locator("#ttsTestBtn").click()
    page.wait_for_function("__diaTest.requests.length === 2")
    page.locator("#ttsTestBtn").click()
    page.wait_for_function("__diaTest.requests.length === 3 && __diaTest.cancelled >= 2")
    page.wait_for_function("__diaTest.finished === 2")
    result = page.evaluate("__diaTest")
    assert all(r["stream"] and r["voice"] == "dia2:lou" for r in result["requests"])
    assert not errors, errors
    print(json.dumps({"picker": "PASS", "progressive_playback": "PASS",
                      "odd_pcm_boundary": "PASS", "interrupt_restart": "PASS",
                      "page_errors": errors, "requests": len(result["requests"])}))
    browser.close()
