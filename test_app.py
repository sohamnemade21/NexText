import sys
from fastapi.testclient import TestClient
from app import app


def test_api_suite():
    print("\n" + "=" * 60)
    print("RUNNING API TEST SUITE")
    print("=" * 60)

    # Use TestClient with lifespan context
    with TestClient(app) as client:
        # 1. Test GET / (Frontend UI)
        print("\n[Test 1] Testing GET / (Frontend UI)...")
        res = client.get("/")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "Abstractive Text Summarizer" in res.text
        assert "NexText" in res.text
        print("[PASS] GET / returned 200 OK and rendered HTML properly.")

        # 2. Test GET /health
        print("\n[Test 2] Testing GET /health...")
        res = client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert data["model"] == "Pegasus-SamSum"
        print(f"[PASS] GET /health returned healthy status on device: {data['device']}")

        # 3. Test POST /predict with Valid Input
        print("\n[Test 3] Testing POST /predict with Valid Dialogue Input...")
        valid_payload = {
            "text": "Hannah: Hey, are you coming to the dinner tonight?\nRob: Yes, what time should I be there?\nHannah: Around 7:30 PM at Luigi's restaurant.\nRob: Sounds good, see you there!"
        }
        res = client.post("/predict", json=valid_payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert "summary" in data
        assert isinstance(data["summary"], str)
        assert len(data["summary"]) > 0
        print(f"[PASS] Valid input returned summary successfully:\n  Summary: '{data['summary']}'")

        # 4. Test POST /predict with Empty Text
        print("\n[Test 4] Testing POST /predict with Empty Text...")
        res = client.post("/predict", json={"text": ""})
        assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.text}"
        data = res.json()
        assert "detail" in data
        print(f"[PASS] Empty input correctly rejected with 400: {data['detail']}")

        # 5. Test POST /predict with Whitespace-only Text
        print("\n[Test 5] Testing POST /predict with Whitespace-only Text...")
        res = client.post("/predict", json={"text": "    \n\t   "})
        assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.text}"
        data = res.json()
        assert "detail" in data
        print(f"[PASS] Whitespace input correctly rejected with 400: {data['detail']}")

        # 6. Test POST /predict with Missing 'text' key
        print("\n[Test 6] Testing POST /predict with Missing 'text' key...")
        res = client.post("/predict", json={"invalid_key": "some text"})
        assert res.status_code in (400, 422), f"Expected 400/422, got {res.status_code}"
        print("[PASS] Missing text key properly rejected.")

        # 7. Test POST /predict with Invalid Data Type (e.g., number instead of string)
        print("\n[Test 7] Testing POST /predict with Non-string Type...")
        res = client.post("/predict", json={"text": 12345})
        assert res.status_code in (400, 422), f"Expected 400/422, got {res.status_code}"
        print("[PASS] Non-string input properly rejected.")

        # 8. Test POST /predict with Long Input (within allowed limit)
        print("\n[Test 8] Testing POST /predict with Long Input...")
        long_text = "Doctor: Good morning, how are your symptoms today?\nPatient: I still have a mild fever and cough.\n" * 15
        res = client.post("/predict", json={"text": long_text})
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert "summary" in data
        assert len(data["summary"]) > 0
        print(f"[PASS] Long input summarized successfully:\n  Summary: '{data['summary']}'")

        # 9. Test POST /predict exceeding max limit (>20,000 characters)
        print("\n[Test 9] Testing POST /predict with Excessive Length (>20,000 chars)...")
        excessive_text = "Too long text pattern! " * 2000
        res = client.post("/predict", json={"text": excessive_text})
        assert res.status_code in (400, 422), f"Expected 400/422, got {res.status_code}"
        print("[PASS] Excessively long input correctly rejected with 400.")

    print("\n" + "=" * 60)
    print("ALL 9 TEST CASES PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_api_suite()
