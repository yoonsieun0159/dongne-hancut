"""
동네한컷 AI 분석 서버
=====================
- 브라우저에 index.html 화면을 보여주고
- /api/analyze 로 받은 사진을 Claude(AI)에게 보내서 분석 결과를 돌려줘요.

API 키를 서버에만 두기 때문에, 웹사이트를 보는 사람에게 키가 드러나지 않아요.

실행 방법 (자세한 설명은 README.md)
  1) pip install -r requirements.txt
  2) .env 파일에  ANTHROPIC_API_KEY=내_API_키  한 줄 적기
  3) python server.py
  4) 브라우저에서 http://localhost:5000 열기
"""

import base64
import binascii
import json
import os

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

# .env 파일에 적어 둔 API 키를 불러와요
load_dotenv()

app = Flask(__name__, static_folder=None)
HERE = os.path.dirname(os.path.abspath(__file__))  # 이 파일이 있는 폴더

MODEL = "claude-opus-5-5"            # 사용할 Claude 모델
MAX_IMAGE_BYTES = 5 * 1024 * 1024    # 사진 최대 5MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# 화면의 '문제 유형' 목록과 똑같아야 해요
CATEGORIES = [
    "도로·보도블록 파손",
    "가로등 고장",
    "쓰레기 무단 방치",
    "위험한 보행로",
    "공공시설 파손",
    "기타",
]
LEVELS = ["낮음", "보통", "높음"]

# AI에게 주는 역할 설명 (분석 기준)
SYSTEM_PROMPT = """당신은 주민이 동네의 불편하거나 위험한 시설을 구청이나 안전신문고에 신고하도록 돕는 분석가입니다.
사용자가 보낸 사진 한 장을 보고, 아래 JSON 형식으로 한국어로 답하세요.

[분석 원칙]
- 사진에 실제로 보이는 것만 근거로 판단하세요. 보이지 않는 원인, 장소, 날짜는 추측하지 마세요.
- 사진이 흐리거나 문제가 잘 보이지 않으면 confidence를 "낮음"으로 하고, 그 이유를 description에 적으세요.
- 사람을 알아보거나 신원을 추측하지 마세요.

[항목 설명]
- is_issue: 사진에 신고할 만한 공공장소의 문제가 보이면 true. 음식, 셀카, 실내 사진처럼 관련이 없으면 false.
- category: 아래 중 가장 알맞은 하나. 문제가 없거나 애매하면 "기타".
  · 도로·보도블록 파손: 도로 구멍(포트홀), 깨지거나 들뜬 보도블록, 갈라진 아스팔트
  · 가로등 고장: 꺼졌거나 깨진 가로등, 기울어진 가로등 기둥
  · 쓰레기 무단 방치: 길가에 버려진 쓰레기, 대형 폐기물, 넘친 쓰레기통
  · 위험한 보행로: 보행로를 막은 장애물, 미끄럽거나 좁은 길, 사라진 안전시설
  · 공공시설 파손: 벤치, 난간, 표지판, 안내판, 놀이기구 등 공공시설의 파손
- risk: 예상 위험도
  · 높음: 지금 바로 사람이 다치거나 사고가 날 수 있음 (깊은 구멍, 노출된 전선, 쓰러질 듯한 구조물 등)
  · 보통: 당장은 아니어도 불편하고, 시간이 지나면 다칠 수 있음
  · 낮음: 주로 위생이나 미관 문제
- confidence: 이 판단을 얼마나 확신하는지 ("낮음" / "보통" / "높음")
- summary: 문제를 한 줄로 요약 (20자 안팎)
- description: 신고 담당자가 읽을 수 있게 정중한 2~3문장. 사진에서 보이는 근거(대략적인 크기, 상태 등)를 포함하세요.
- risk_reason: 위험도를 그렇게 판단한 이유 한 문장
- privacy_note: 사진에 사람 얼굴, 차량 번호판, 집 주소처럼 개인정보가 보이면 신고 전에 가리라는 안내 한 문장. 없으면 빈 문자열 ""."""

# AI 답을 이 모양(JSON)으로만 받도록 정해 둬요 → 화면에서 바로 쓸 수 있어요
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_issue": {"type": "boolean"},
        "category": {"type": "string", "enum": CATEGORIES},
        "risk": {"type": "string", "enum": LEVELS},
        "confidence": {"type": "string", "enum": LEVELS},
        "summary": {"type": "string"},
        "description": {"type": "string"},
        "risk_reason": {"type": "string"},
        "privacy_note": {"type": "string"},
    },
    "required": [
        "is_issue", "category", "risk", "confidence",
        "summary", "description", "risk_reason", "privacy_note",
    ],
    "additionalProperties": False,
}

_client = None


def get_client():
    """Claude 클라이언트를 한 번만 만들어서 재사용해요."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # .env의 ANTHROPIC_API_KEY를 자동으로 사용
    return _client


def error(message, status):
    """화면에 보여줄 오류 메시지를 JSON으로 돌려줘요."""
    return jsonify({"ok": False, "error": message}), status


# ---------- 화면 파일 보내기 ----------

@app.get("/")
def home():
    return send_from_directory(HERE, "index.html")


@app.get("/images/<path:name>")
def images(name):
    return send_from_directory(os.path.join(HERE, "images"), name)


# ---------- 서버 상태 확인 ----------

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "has_key": bool(os.environ.get("ANTHROPIC_API_KEY"))})


# ---------- 사진 분석 ----------

@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")
    media_type = data.get("media_type", "")

    # 1) 받은 사진 검사
    if media_type not in ALLOWED_TYPES:
        return error("JPG, PNG, WEBP 사진만 분석할 수 있어요.", 400)
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        return error("사진 데이터가 올바르지 않아요.", 400)
    if not image_bytes:
        return error("사진이 비어 있어요.", 400)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return error("사진이 너무 커요. 5MB 이하로 올려 주세요.", 413)

    # 2) Claude에게 사진 보내기
    try:
        response = get_client().beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            # 안전 기준 때문에 거절되면 다른 모델로 자동 재시도
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": "medium",  # 생각하는 깊이 (low / medium / high ...)
                "format": {"type": "json_schema", "schema": RESULT_SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                    },
                    {"type": "text", "text": "이 사진 속 동네 문제를 분석해 주세요."},
                ],
            }],
        )
    except anthropic.AuthenticationError:
        return error("API 키가 올바르지 않아요. .env 파일의 키를 확인해 주세요.", 500)
    except anthropic.PermissionDeniedError:
        return error("이 API 키로는 분석할 권한이 없어요.", 500)
    except anthropic.RateLimitError:
        return error("요청이 너무 많아요. 잠시 후 다시 시도해 주세요.", 429)
    except anthropic.BadRequestError as e:
        app.logger.warning("잘못된 요청: %s", e.message)
        return error("AI가 이 요청을 처리하지 못했어요. 다른 사진으로 시도해 주세요.", 400)
    except anthropic.APIStatusError as e:
        app.logger.warning("API 오류 %s: %s", e.status_code, e.message)
        return error("AI 서버에 문제가 있어요. 잠시 후 다시 시도해 주세요.", 502)
    except anthropic.APIConnectionError:
        return error("AI 서버에 연결하지 못했어요. 인터넷 연결을 확인해 주세요.", 502)
    except TypeError:
        # API 키가 아예 없을 때
        return error("API 키가 설정되지 않았어요. .env 파일을 확인해 주세요.", 500)

    # 3) 답 꺼내기
    if response.stop_reason == "refusal":
        return error("AI가 이 사진은 분석할 수 없다고 답했어요. 다른 사진으로 시도해 주세요.", 422)
    if response.stop_reason == "max_tokens":
        return error("AI 답이 너무 길어져서 끊겼어요. 다시 시도해 주세요.", 502)

    text = next((block.text for block in response.content if block.type == "text"), None)
    try:
        result = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return error("AI 답을 읽지 못했어요. 다시 시도해 주세요.", 502)

    return jsonify({"ok": True, "result": result})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY가 없어요. .env 파일에 API 키를 적어 주세요.")
    print("✅ 서버 시작! 브라우저에서 http://localhost:5000 을 열어 주세요.")
    app.run(host="127.0.0.1", port=5000, debug=False)
