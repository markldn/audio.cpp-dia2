#!/usr/bin/env python3
"""Exercise Dia2's actual chat picker, progressive WebAudio, and interruption."""
import argparse
import json
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8688")
parser.add_argument("--voice", default="dia2:lou")
args = parser.parse_args()
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
    page = browser.new_page()
    def instrument_app(route):
        response = route.fetch()
        source = response.text()
        marker = 'document.addEventListener("DOMContentLoaded", init);'
        assert marker in source
        hook = """
          window.__diaHooks = {state, voiceConversation, chatTtsVoice, generate,
            readyLou() {
              newConv = opts => {
                state.conv = {id:'browser-test-private', private:true, created:Date.now(),
                  updated:Date.now(), params:{}, messages:[], ...opts};
                return state.conv;
              };
              if (!voiceConversation.h.ready({mode:'lou', model:'test-no-inference'}))
                throw new Error('Lou setup failed');
              state.settings.ttsAutoplay = false;
            }
          };
        """
        route.fulfill(response=response, body=source.replace(marker, hook + marker))
    page.route("**/app.js*", instrument_app)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script(r"""
      window.__diaTest = { starts: [], requests: [], jobs: [], settings: null, cancelled: 0, finished: 0 };
      const start = AudioBufferSourceNode.prototype.start;
      AudioBufferSourceNode.prototype.start = function(...args) {
        __diaTest.starts.push(performance.now()); return start.apply(this, args);
      };
      const fetchOriginal = window.fetch;
      window.fetch = async function(url, options) {
        if (String(url).endsWith('/api/settings') && options?.method === 'PUT') {
          __diaTest.settings = JSON.parse(options.body);
          return new Response(JSON.stringify({ok: true}), {headers: {'Content-Type': 'application/json'}});
        }
        if (String(url).endsWith('/api/jobs') && options?.method === 'POST') {
          __diaTest.jobs.push(JSON.parse(options.body));
          return new Response(JSON.stringify({error: 'Browser test stops before LLM inference'}), {status: 503});
        }
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
    if args.voice.startswith("dia2:gpu1:"):
        for size in ("1b", "2b"):
            for quant in ("f16", "q8_0", "q4_0"):
                for speaker in ("default", "kitt", "lou"):
                    assert page.locator(f'#setTtsVoice option[value="dia2:gpu1:{size}:{quant}:{speaker}"]').count() == 1
        assert page.locator('#setTtsVoice option[value="dia2:gpu1:2b:q8_0:lou"]').text_content() == "Dia 2 2B Q8_0 Lou (GPU 1 / 16 GB)"
    page.locator("#setTtsVoice").select_option(args.voice)
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
    assert all(r["stream"] and r["voice"] == args.voice for r in result["requests"])
    page.locator("#setVoicePromptMode").select_option("lou")
    page.locator("#setVoiceModeTtsVoice").select_option(args.voice)
    page.locator("#setVoiceModeEmotions").uncheck()
    page.locator("#setVoicePromptMode").select_option("kitt")
    page.locator("#setVoicePromptMode").select_option("lou")
    assert page.locator("#setVoiceModeTtsVoice").input_value() == args.voice
    assert not page.locator("#setVoiceModeEmotions").is_checked()
    page.locator("#setTtsVoice").select_option("af_bella")
    page.locator("#saveSettingsBtn").click()
    page.wait_for_function("__diaTest.settings !== null")
    assert page.evaluate("__diaTest.settings.voiceModeVoices.lou") == args.voice
    page.evaluate("__diaHooks.readyLou()")
    assert page.evaluate("__diaHooks.chatTtsVoice()") == args.voice
    assert page.evaluate("__diaHooks.state.settings.ttsVoice") == "af_bella"
    assert page.evaluate("__diaHooks.state.conv.voiceEmotions") is False
    page.evaluate("__diaHooks.generate(__diaHooks.state.conv)")
    assert not page.evaluate("__diaTest.jobs.at(-1).messages.some(m => String(m.content).includes('Dia 2 vocal cue'))")
    page.evaluate("""async () => {
      __diaHooks.state.conv.voiceEmotions = true;
      __diaHooks.state.settings.ttsDia2Expressive = false;
      await __diaHooks.generate(__diaHooks.state.conv);
    }""")
    if args.voice.startswith("dia2:"):
        assert page.evaluate("__diaTest.jobs.at(-1).messages.some(m => String(m.content).includes('Dia 2 vocal cue'))")
    assert not errors, errors
    print(json.dumps({"picker": "PASS", "progressive_playback": "PASS",
                      "odd_pcm_boundary": "PASS", "interrupt_restart": "PASS",
                      "voice_mode_preset": "PASS", "per_chat_voice": "PASS", "emotion_toggle": "PASS",
                      "page_errors": errors, "requests": len(result["requests"])}))
    browser.close()
