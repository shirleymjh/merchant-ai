import base64
import hashlib
import hmac
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from merchant_ai.config import Settings


def identity_token(secret: str, merchant_id: str = "100", role: str = "merchant_owner") -> str:
    def encode(payload):
        return base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).decode("ascii").rstrip("=")

    header = encode({"alg": "HS256", "typ": "JWT"})
    claims = encode({"sub": "api-user", "merchantId": merchant_id, "role": role})
    signing_input = "%s.%s" % (header, claims)
    signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    return "%s.%s" % (signing_input, signature)


def authenticated_client(tmp_path: Path) -> tuple[TestClient, str]:
    secret = "merchant-api-auth-test-secret"
    app = create_app(
        Settings(
            merchant_id="100",
            allowed_merchant_ids="100,200",
            identity_auth_required=True,
            identity_jwt_secret=secret,
            harness_workspace_path=str(tmp_path),
        )
    )
    return TestClient(app), secret


def test_merchant_scoped_read_and_upload_endpoints_require_identity(tmp_path: Path) -> None:
    client, _secret = authenticated_client(tmp_path)

    responses = [
        client.post(
            "/api/attachments?merchantId=100&name=note.txt&type=text/plain",
            content=b"safe attachment",
        ),
        client.get("/api/runs?merchantId=100"),
        client.get("/api/runs/dashboard?merchantId=100"),
        client.get("/api/daily-report?merchantId=100"),
        client.get("/api/merchant-profile/100"),
    ]

    assert [response.status_code for response in responses] == [401, 401, 401, 401, 401]


def test_authenticated_identity_cannot_select_another_merchant(tmp_path: Path) -> None:
    client, secret = authenticated_client(tmp_path)
    headers = {"Authorization": "Bearer %s" % identity_token(secret, merchant_id="100")}

    responses = [
        client.post(
            "/api/attachments?merchantId=200&name=note.txt&type=text/plain",
            content=b"safe attachment",
            headers=headers,
        ),
        client.get("/api/runs?merchantId=200", headers=headers),
        client.get("/api/runs/dashboard?merchantId=200", headers=headers),
        client.get("/api/daily-report?merchantId=200", headers=headers),
        client.get("/api/merchant-profile/200", headers=headers),
    ]

    assert [response.status_code for response in responses] == [403, 403, 403, 403, 403]


def test_verified_same_merchant_identity_can_access_run_listing_and_upload(tmp_path: Path) -> None:
    client, secret = authenticated_client(tmp_path)
    headers = {"Authorization": "Bearer %s" % identity_token(secret)}

    runs = client.get("/api/runs?merchantId=100", headers=headers)
    uploaded = client.post(
        "/api/attachments?merchantId=100&name=note.txt&type=text/plain",
        content=b"safe attachment",
        headers=headers,
    )

    assert runs.status_code == 200
    assert uploaded.status_code == 200
    assert uploaded.json()["success"] is True


def test_current_profile_scope_comes_from_verified_identity_not_frontend_default(tmp_path: Path) -> None:
    client, secret = authenticated_client(tmp_path)
    headers = {"Authorization": "Bearer %s" % identity_token(secret, merchant_id="200")}

    response = client.get("/api/merchant-profile", headers=headers)

    assert response.status_code == 200
    assert response.json()["profile"]["merchantId"] == "200"


def test_explicitly_disabled_identity_auth_keeps_local_merchant_access(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            Settings(
                merchant_id="100",
                allowed_merchant_ids="100",
                identity_auth_required=False,
                harness_workspace_path=str(tmp_path),
            )
        )
    )

    assert client.get("/api/runs?merchantId=100").status_code == 200


def test_operator_session_cookie_authenticates_internal_governance(tmp_path: Path) -> None:
    client, secret = authenticated_client(tmp_path)
    client.cookies.set(
        "yshopping_ops_session",
        identity_token(secret, role="platform_operator"),
    )

    response = client.get("/api/topics")

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_merchant_session_cookie_cannot_open_internal_governance(tmp_path: Path) -> None:
    client, secret = authenticated_client(tmp_path)
    client.cookies.set(
        "yshopping_ops_session",
        identity_token(secret, role="merchant_owner"),
    )

    response = client.get("/api/topics")

    assert response.status_code == 403


def test_unconfigured_grounded_data_plane_is_typed_503_and_control_plane_stays_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("YSHOPPING_LLM_API_KEY", "")
    application = create_app(
        Settings(
            merchant_id="100",
            allowed_merchant_ids="100",
            identity_auth_required=False,
            llm_api_key="",
            harness_workspace_path=str(tmp_path),
        )
    )

    assert application.state.runtime.runtime_trace()["onlineReady"] is False
    response = TestClient(application).post(
        "/api/chat",
        json={"message": "最近7天订单量", "merchantId": "100"},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "GROUNDED_ONLINE_RUNTIME_UNAVAILABLE"
    )


def test_browser_bundle_does_not_embed_or_send_ops_credentials() -> None:
    source = (Path(__file__).parents[3] / "frontend" / "src" / "api" / "client.js").read_text(encoding="utf-8")

    assert "VITE_OPS_TOKEN" not in source
    assert "X-Ops-Token" not in source
