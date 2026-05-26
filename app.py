"""
VoiceClone — Production backend with IndexTTS2 + DeepSeek + FunASR.
UI matches prototype: HTML structure + JS in head.
"""
import json
import os
import tempfile
import traceback
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from prompts import get_style_prompt, get_style_list

warnings.filterwarnings("ignore")

_tts = None
_llm = None
_asr = None


def _scan_voices() -> list:
    import yaml
    voices = []
    vdir = Path("voices")
    if not vdir.exists():
        return voices
    for d in sorted(vdir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "reference.wav").exists():
            continue
        sy = d / "speaker.yaml"
        name = d.name
        if sy.exists():
            with open(sy, "r", encoding="utf-8") as f:
                info = yaml.safe_load(f) or {}
            name = info.get("name", d.name)
        voices.append({"id": d.name, "name": name})
    return voices


def _voice_id(name: str) -> str:
    for v in _scan_voices():
        if v["name"] == name:
            return v["id"]
    return ""


def _get_tts():
    global _tts
    if _tts is not None:
        return _tts
    from engines.tts_engine import TTSEngine
    _tts = TTSEngine()
    return _tts


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    from engines.llm_engine import LLMEngine
    _llm = LLMEngine()
    return _llm


def _get_asr():
    global _asr
    if _asr is not None:
        return _asr
    from engines.asr_engine import ASREngine
    _asr = ASREngine()
    return _asr


def _save_audio(audio_tensor, sample_rate: int) -> str:
    audio_np = audio_tensor.cpu().numpy()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_np, sample_rate)
    tmp.close()
    return tmp.name


def process(data_json: str):
    if not data_json or not data_json.strip():
        return None, ""

    try:
        data = json.loads(data_json)
    except json.JSONDecodeError:
        return None, ""

    mode = data.get("mode", "")
    voice = data.get("voice", "")
    style = data.get("style", "")
    text = data.get("text", "")
    audio_path_input = data.get("audio_path", "")

    try:
        vid = _voice_id(voice)
        if not vid:
            return None, f"未找到声音: {voice}，请先添加声音样本到 voices/ 目录"

        if mode == "语音聊天":
            if not audio_path_input:
                return None, "未收到录音。"
            asr_engine = _get_asr()
            transcript = asr_engine.recognize(audio_path_input)
            if not transcript:
                return None, "未检测到语音内容。"
            llm = _get_llm()
            prompt = get_style_prompt(style)
            reply = llm.chat(transcript, prompt)
            tts = _get_tts()
            audio = tts.synthesize(reply, vid)
            audio_file = _save_audio(audio, tts.sample_rate)
            return audio_file, reply

        elif mode == "发送短信":
            if not text.strip():
                return None, "请输入消息。"
            llm = _get_llm()
            prompt = get_style_prompt(style)
            reply = llm.chat(text, prompt)
            tts = _get_tts()
            audio = tts.synthesize(reply, vid)
            audio_file = _save_audio(audio, tts.sample_rate)
            return audio_file, reply

        elif mode == "朗读内容":
            if not text.strip():
                return None, "请输入朗读文字。"
            tts = _get_tts()
            audio = tts.synthesize(text, vid)
            audio_file = _save_audio(audio, tts.sample_rate)
            return audio_file, text

        else:
            return None, f"未知模式: {mode}"

    except Exception as e:
        traceback.print_exc()
        return None, f"错误: {e}"


# ── HTML (structure only) ──
HTML_UI = """
<div id="app-root" style="width:100%;max-width:1100px;margin:0 auto;padding:50px 60px 40px;display:flex;flex-direction:column;align-items:center;box-sizing:border-box;">
  <div id="waveform-wrap" style="width:100%;height:300px;position:relative;margin-bottom:2px;">
    <canvas id="waveform-canvas" style="width:100%;height:300px;display:block;"></canvas>
  </div>
  <div style="display:flex;justify-content:center;margin-bottom:210px;">
    <select id="voice-dd" style="padding:2px 26px 2px 8px;border:1px solid #eaeae4;border-radius:4px;background:#f7f7f4;color:#a0a0a0;font-size:11px;cursor:pointer;outline:none;appearance:none;text-align:center;letter-spacing:0.5px;min-width:80px;background-image:url(&quot;data:image/svg+xml,%3Csvg width='7' height='4' viewBox='0 0 7 4' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L3.5 3L6 1' stroke='%23aaa' stroke-width='1' stroke-linecap='round'/%3E%3C/svg%3E&quot;);background-repeat:no-repeat;background-position:right 6px center;">%VOICE_OPTIONS%</select>
  </div>
  <div style="display:flex;flex-direction:row;align-items:center;justify-content:center;gap:20px;margin-bottom:40px;flex-wrap:nowrap;">
    <div id="mode-tabs" style="display:flex;flex-direction:row;align-items:stretch;gap:4px;background:#eee;border-radius:10px;padding:3px;flex-shrink:0;">
      <button id="tab-voice" class="mode-tab active" data-mode="voice" style="min-width:110px;height:38px;border-radius:8px;font-size:14px;font-weight:500;font-family:inherit;color:#1a1a1a;cursor:pointer;border:none;outline:none;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.06);white-space:nowrap;padding:0 22px;display:inline-flex;align-items:center;justify-content:center;transition:all 0.2s;">语音聊天</button>
      <button id="tab-sms" class="mode-tab" data-mode="sms" style="min-width:110px;height:38px;border-radius:8px;font-size:14px;font-weight:400;font-family:inherit;color:#aaa;cursor:pointer;border:none;outline:none;background:transparent;white-space:nowrap;padding:0 22px;display:inline-flex;align-items:center;justify-content:center;transition:all 0.2s;">发送短信</button>
      <button id="tab-read" class="mode-tab" data-mode="read" style="min-width:110px;height:38px;border-radius:8px;font-size:14px;font-weight:400;font-family:inherit;color:#aaa;cursor:pointer;border:none;outline:none;background:transparent;white-space:nowrap;padding:0 22px;display:inline-flex;align-items:center;justify-content:center;transition:all 0.2s;">朗读内容</button>
    </div>
    <select id="style-dd" style="padding:7px 30px 7px 14px;border:1px solid #e6e6e0;border-radius:8px;background:#f7f7f4;color:#999;font-size:13px;cursor:pointer;outline:none;appearance:none;min-width:150px;text-align:center;background-image:url(&quot;data:image/svg+xml,%3Csvg width='8' height='5' viewBox='0 0 8 5' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L4 4L7 1' stroke='%23aaa' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E&quot;);background-repeat:no-repeat;background-position:right 12px center;">%STYLE_OPTIONS%</select>
  </div>
  <div id="panel-voice" class="mode-panel" style="display:flex;flex-direction:column;align-items:center;gap:18px;width:100%;">
    <button id="mic-btn" style="width:96px;height:96px;border-radius:50%;border:2px solid #e4e4de;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.25s;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
      <svg id="mic-icon" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="1.8" stroke-linecap="round"><rect x="8" y="2" width="8" height="13" rx="4"/><path d="M4 11a8 8 0 0 0 16 0"/><line x1="12" y1="18" x2="12" y2="22"/></svg>
    </button>
  </div>
  <div id="panel-sms" class="mode-panel" style="display:none;flex-direction:column;align-items:center;gap:18px;width:100%;">
    <div style="display:flex;gap:14px;width:80%;max-width:650px;align-items:center;"><div style="flex:1;"><input id="sms-input" type="text" placeholder="输入消息..." style="width:100%;padding:16px 22px;border:1px solid #e4e4de;border-radius:28px;background:#fff;color:#1a1a1a;font-size:16px;outline:none;box-sizing:border-box;box-shadow:0 1px 4px rgba(0,0,0,0.03);"></div>
    <button id="sms-send-btn" style="width:50px;height:50px;border-radius:50%;border:1px solid #e4e4de;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all 0.2s;box-shadow:0 1px 4px rgba(0,0,0,0.03);"><svg id="sms-send-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="2" stroke-linecap="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22,2 15,22 11,13 2,9"/></svg></button></div>
  </div>
  <div id="panel-read" class="mode-panel" style="display:none;flex-direction:column;align-items:center;gap:18px;width:100%;">
    <div style="display:flex;gap:14px;width:80%;max-width:650px;align-items:center;"><div style="flex:1;"><input id="read-input" type="text" placeholder="输入要朗读的文字..." style="width:100%;padding:16px 22px;border:1px solid #e4e4de;border-radius:28px;background:#fff;color:#1a1a1a;font-size:16px;outline:none;box-sizing:border-box;box-shadow:0 1px 4px rgba(0,0,0,0.03);"></div>
    <button id="read-send-btn" style="width:50px;height:50px;border-radius:50%;border:1px solid #e4e4de;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all 0.2s;box-shadow:0 1px 4px rgba(0,0,0,0.03);"><svg id="read-send-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="2" stroke-linecap="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22,2 15,22 11,13 2,9"/></svg></button></div>
  </div>
  <div id="response-toast" style="font-size:14px;color:#aaa;text-align:center;padding:6px 0;min-height:24px;margin-top:18px;opacity:0;transition:opacity 0.3s;"></div>
</div>
"""


JS_CODE = """
<script>
(function() {
  var obs = new MutationObserver(function(m, o) {
    var root = document.getElementById('app-root');
    if (!root || !root.offsetParent) return;
    o.disconnect();
    initApp();
  });
  obs.observe(document.body, { childList: true, subtree: true });

  function initApp() {
    var currentMode = 'voice';
    var recording = false;
    var toastTimer = null;

    var tabs = document.querySelectorAll('.mode-tab');
    tabs.forEach(function(btn) {
      btn.addEventListener('click', function() {
        var mode = this.dataset.mode;
        currentMode = mode;
        tabs.forEach(function(t) {
          t.style.background = 'transparent';
          t.style.color = '#aaa';
          t.style.boxShadow = 'none';
          t.style.fontWeight = '400';
        });
        this.style.background = '#fff';
        this.style.color = '#1a1a1a';
        this.style.fontWeight = '500';
        this.style.boxShadow = '0 1px 3px rgba(0,0,0,0.06)';
        document.getElementById('panel-voice').style.display = mode==='voice'?'flex':'none';
        document.getElementById('panel-sms').style.display = mode==='sms'?'flex':'none';
        document.getElementById('panel-read').style.display = mode==='read'?'flex':'none';
        document.getElementById('style-dd').style.visibility = mode==='read'?'hidden':'visible';
      });
    });

    function sendToPython(payload) {
      var hidden = document.querySelector('#hidden-request textarea, #hidden-request input');
      if (!hidden) return;
      var desc = hidden.tagName === 'INPUT'
        ? Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')
        : Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
      desc.set.call(hidden, '');
      hidden.dispatchEvent(new Event('input',{bubbles:true}));
      setTimeout(function() {
        desc.set.call(hidden, JSON.stringify(payload));
        hidden.dispatchEvent(new Event('input',{bubbles:true}));
        hidden.dispatchEvent(new Event('change',{bubbles:true}));
        hidden.dispatchEvent(new Event('blur',{bubbles:true}));
      }, 100);
    }

    document.getElementById('mic-btn').addEventListener('click', function() {
      recording = !recording;
      var btn = document.getElementById('mic-btn');
      var icon = document.getElementById('mic-icon');
      if (recording) {
        btn.style.borderColor = '#e04545'; btn.style.background = '#fef5f5';
        btn.style.boxShadow = '0 2px 12px rgba(224,69,69,0.15)';
        icon.style.stroke = '#e04545';
        // Trigger hidden mic recorder
        var micTrigger = document.querySelector('#mic-trigger audio') || document.querySelector('#mic-trigger button');
        if (micTrigger) micTrigger.click();
      } else {
        btn.style.borderColor = '#e4e4de'; btn.style.background = '#fff';
        btn.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)';
        icon.style.stroke = '#333';
        // Audio will be processed by Gradio's stop_recording event
      }
    });

    document.getElementById('sms-send-btn').addEventListener('click', function() {
      var input = document.getElementById('sms-input');
      var text = input.value.trim(); if(!text) return;
      sendToPython({mode:'发送短信', voice: document.getElementById('voice-dd').value, style: document.getElementById('style-dd').value, text:text});
      input.value = '';
    });
    document.getElementById('sms-input').addEventListener('keydown', function(e) {
      if(e.key==='Enter') document.getElementById('sms-send-btn').click();
    });

    document.getElementById('read-send-btn').addEventListener('click', function() {
      var input = document.getElementById('read-input');
      var text = input.value.trim(); if(!text) return;
      sendToPython({mode:'朗读内容', voice: document.getElementById('voice-dd').value, style:'', text:text});
      input.value = '';
    });
    document.getElementById('read-input').addEventListener('keydown', function(e) {
      if(e.key==='Enter') document.getElementById('read-send-btn').click();
    });

    ['sms-send-btn','read-send-btn'].forEach(function(id) {
      var b = document.getElementById(id);
      var iconId = id === 'sms-send-btn' ? 'sms-send-icon' : 'read-send-icon';
      b.addEventListener('mouseenter', function() {
        this.style.background='#1a1a1a';this.style.borderColor='#1a1a1a';
        this.style.boxShadow='0 2px 10px rgba(0,0,0,0.15)';
        document.getElementById(iconId).style.stroke='#fff';
      });
      b.addEventListener('mouseleave', function() {
        this.style.background='#fff';this.style.borderColor='#e4e4de';
        this.style.boxShadow='0 1px 4px rgba(0,0,0,0.03)';
        document.getElementById(iconId).style.stroke='#333';
      });
    });

    function checkResponse() {
      var el = document.querySelector('#hidden-response textarea, #hidden-response input');
      if (el && el.value) {
        var text = el.value;
        var toast = document.getElementById('response-toast');
        toast.textContent = text; toast.style.opacity = '1';
        clearTimeout(toastTimer);
        toastTimer = setTimeout(function(){ toast.style.opacity='0'; }, 5000);
        var desc = el.tagName==='INPUT'
          ? Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value')
          : Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value');
        desc.set.call(el,''); el.dispatchEvent(new Event('input',{bubbles:true}));
      }
      requestAnimationFrame(function(){ setTimeout(checkResponse, 300); });
    }
    setTimeout(checkResponse, 500);

    // Waveform
    (function() {
      var canvas = document.getElementById('waveform-canvas');
      var ctx = canvas.getContext('2d');
      var BAR_COUNT = 55;
      var DPR = window.devicePixelRatio || 1;
      function resizeWave() {
        var rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * DPR;
        canvas.height = rect.height * DPR;
        ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      }
      resizeWave();
      new ResizeObserver(resizeWave).observe(canvas.parentElement);
      var phases=[]; for(var i=0;i<BAR_COUNT;i++) phases.push(Math.random()*Math.PI*2);
      var speeds=[]; for(var i=0;i<BAR_COUNT;i++) speeds.push(0.5+Math.random()*1.2);
      var targetVolume=0, currentVolume=0;
      var BREATH_AMP=0.05, SPEAK_AMP=0.8, VOLUME_LERP=0.05;
      var audioCtx=null, analyser=null, freqData=null, isPlaying=false;
      function setupAudioCtx() {
        var audios = document.querySelectorAll('audio');
        for(var a=0;a<audios.length;a++) {
          var audio=audios[a];
          if(audio.dataset.wf) continue;
          audio.dataset.wf='1';
          if(!audioCtx) {
            audioCtx = new (window.AudioContext||window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize=256; analyser.smoothingTimeConstant=0.5;
            freqData = new Uint8Array(analyser.frequencyBinCount);
          }
          try { var src=audioCtx.createMediaElementSource(audio); src.connect(analyser); analyser.connect(audioCtx.destination); } catch(e){}
          audio.addEventListener('play', function(){ if(audioCtx.state==='suspended') audioCtx.resume(); isPlaying=true; });
          audio.addEventListener('ended', function(){ isPlaying=false; });
          audio.addEventListener('pause', function(){ isPlaying=false; });
        }
      }
      var time=0;
      function drawWave() {
        requestAnimationFrame(drawWave); var dt=0.016; time+=dt;
        if(Math.floor(time*10)%20===0) setupAudioCtx();
        var liveVol=0;
        if(isPlaying && analyser && freqData) {
          analyser.getByteFrequencyData(freqData);
          var sum=0; for(var i=0;i<freqData.length;i++) sum+=freqData[i];
          liveVol=(sum/freqData.length)/255; liveVol=Math.pow(liveVol,0.55);
        }
        targetVolume=liveVol>0.015?liveVol:0;
        currentVolume+=(targetVolume-currentVolume)*VOLUME_LERP;
        var W=canvas.width/DPR, H=canvas.height/DPR, CY=H/2;
        var barW=W/BAR_COUNT*0.55, gap=W/BAR_COUNT*0.45;
        ctx.clearRect(0,0,W,H);
        ctx.strokeStyle='#eaeae4'; ctx.lineWidth=1;
        ctx.beginPath(); ctx.moveTo(0,CY); ctx.lineTo(W,CY); ctx.stroke();
        for(var i=0;i<BAR_COUNT;i++) {
          var x=i*(barW+gap)+gap/2, t=time*speeds[i];
          var organic=Math.sin(t*2.5+phases[i]);
          var noise=Math.sin(t*5.1+phases[i]*1.7)*0.4+Math.sin(t*9.3+phases[i]*0.9)*0.25+Math.sin(t*15.7+phases[i]*0.4)*0.15;
          var vibrato=Math.sin(i*0.3+time*0.4)*0.06*currentVolume;
          var liveBar=0;
          if(liveVol>0.015&&freqData){var binIdx=Math.floor(i/BAR_COUNT*freqData.length);liveBar=(freqData[binIdx]||0)/255;liveBar=Math.pow(liveBar,0.55);}
          var breathe=organic*0.04+noise*0.01;
          var speakLive=liveBar*0.7+organic*0.15*liveVol;
          var speak=currentVolume>0.05?speakLive:(organic*0.55+noise*0.22+vibrato)*currentVolume;
          var amplitude=breathe+speak;
          var halfH=Math.abs(amplitude)*H*0.46, top=CY-halfH, bot=CY+halfH;
          var dist=Math.abs(i-BAR_COUNT/2)/(BAR_COUNT/2);
          var alpha=0.55-dist*0.32;
          ctx.fillStyle='rgba(30,30,30,'+alpha+')';
          var r=barW*0.45;
          ctx.beginPath();ctx.moveTo(x,top+r);ctx.arcTo(x,top,x+barW,top,r);
          ctx.arcTo(x+barW,top,x+barW,bot,r);ctx.arcTo(x+barW,bot,x,bot,r);
          ctx.arcTo(x,bot,x,top,r);ctx.closePath();ctx.fill();
        }
      }
      drawWave();
    })();
  }
})();
</script>
"""


css = """
body, .gradio-container { background: #fafaf7 !important; }
footer { display: none !important; }
.gradio-container { max-width: 100% !important; margin: 0 !important; padding: 0 !important; }
#app-root * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important; }
.contain { padding: 0 !important; }
.block.gradio-html { width: 100% !important; max-width: 100% !important; }
.mode-tab { box-sizing: border-box !important; }
"""


def create_demo():
    voices = _scan_voices()
    voice_options = "".join(f'<option>{v["name"]}</option>' for v in voices) or "<option>彪</option><option>喆</option><option>鑫</option>"
    style_options = "".join(f'<option>{s}</option>' for s in get_style_list())

    html_ui = HTML_UI.replace("%VOICE_OPTIONS%", voice_options).replace("%STYLE_OPTIONS%", style_options)

    with gr.Blocks(
        head='<meta name="viewport" content="width=device-width, initial-scale=1.0">' + JS_CODE
    ) as demo:
        gr.HTML(html_ui)

        hidden_request = gr.Textbox(value="", visible=False, elem_id="hidden-request", show_label=False)
        hidden_response = gr.Textbox(value="", visible=False, elem_id="hidden-response", show_label=False)
        output_audio = gr.Audio(autoplay=True, visible=False, show_label=False, elem_id="hidden-audio")
        mic_audio = gr.Audio(sources=["microphone"], type="filepath", visible=False, elem_id="mic-trigger", show_label=False)

        hidden_request.change(
            fn=process,
            inputs=[hidden_request],
            outputs=[output_audio, hidden_response],
        )

        def on_mic_stop(audio_path):
            if not audio_path:
                return None, None, ""
            voice = _scan_voices()
            voice_name = voice[0]["name"] if voice else "彪"
            style = "清纯男高"
            payload = json.dumps({"mode": "语音聊天", "voice": voice_name, "style": style, "text": "", "audio_path": audio_path})
            return None, payload, ""

        mic_audio.stop_recording(
            fn=on_mic_stop,
            inputs=[mic_audio],
            outputs=[mic_audio, hidden_request, hidden_response],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue(default_concurrency_limit=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(),
        css=css,
    )
