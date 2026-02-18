"""Simple test client for the RAG API endpoint."""

import json
import sys

import requests


def test_api_endpoint(url: str = "http://127.0.0.1:8000/query") -> None:
    """Test the RAG API endpoint.

    Args:
        url: Full URL to the query endpoint
    """
    print(f"Testing endpoint: {url}\n")

    # Test query
    payload = {
        "query": "What is RAG?",
        "top_k": 5,
        "threshold": 0.3,
    }

    print("Sending query:", json.dumps(payload, indent=2))
    print("\n" + "=" * 60 + "\n")

    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        response.raise_for_status()

        print("Streaming response:\n")

        full_text = ""
        chunks_received = False

        for line in response.iter_lines():
            if line:
                # Remove 'data: ' prefix
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove 'data: '
                    try:
                        event = json.loads(data_str)

                        if event["type"] == "status":
                            print(f"[STATUS] {event['message']}")

                        elif event["type"] == "chunks":
                            chunks_received = True
                            print(f"\n[CHUNKS] Retrieved {len(event['data'])} chunks:")
                            for i, chunk in enumerate(event["data"][:3], 1):  # Show first 3
                                preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                                print(f"  {i}. Score: {chunk['score']:.4f}")
                                print(f"     {preview}\n")

                        elif event["type"] == "text":
                            chunk_text = event["chunk"]
                            print(chunk_text, end="", flush=True)
                            full_text += chunk_text

                        elif event["type"] == "done":
                            print("\n\n[COMPLETE]")
                            break

                        elif event["type"] == "error":
                            print(f"\n[ERROR] {event['message']}")
                            break

                    except json.JSONDecodeError:
                        print(f"Failed to parse: {data_str}")

        if full_text and chunks_received:
            print("\n" + "=" * 60)
            print("\nSUCCESS! API endpoint is working correctly.")
            print(f"Response length: {len(full_text)} characters")
        elif not chunks_received:
            print("\nWARNING: No chunks were retrieved.")

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed: {e}")
        print("\nMake sure:")
        print("1. The Streamlit app is running")
        print("2. API endpoint is enabled in the Query step")
        print("3. Documents are uploaded and embeddings are generated")
        sys.exit(1)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/query"
    test_api_endpoint(url)
