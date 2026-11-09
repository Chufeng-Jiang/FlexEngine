from fastapi.testclient import TestClient

from flexengine.api.server import create_app


def test_completion_and_stream(engine, conf):
    app = create_app(conf, engine)
    with TestClient(app) as client:
        response = client.post("/v1/completions", json={"prompt":"Hello","max_tokens":3, "model":"test"})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["usage"]["completion_tokens"] == 3
        assert payload["choices"][0]["finish_reason"] == "length"
        with client.stream("POST", "/v1/completions", json={"prompt":"Hi", "max_tokens":3, "stream": True}) as r:
            assert r.status_code == 200
            data = "\n".join(r.iter_lines())
            assert "[DONE]" in data
            assert '"text"' in data
        assert client.get("/healthz").json()["fault"] is None
        assert client.get("/metrics").status_code == 200


def test_invalid_prompt_and_model(engine, conf):
    app = create_app(conf, engine)
    with TestClient(app) as c:
        assert c.post("/v1/completions", json={"prompt":"", "max_tokens":1}).status_code == 400
        assert c.post("/v1/completions", json={"prompt":"x", "model":"other"}).status_code == 400
        assert c.post("/v1/completions", json={"prompt":"x", "max_tokens":999}).status_code == 400
        assert c.delete("/v1/requests/absent").status_code == 404
