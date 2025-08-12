import pytest
from unittest.mock import Mock, patch
from openai import RateLimitError, APITimeoutError
from integrations.direct_openai import DirectOpenAIChat
from httpx import Response, Request


def _setup_chat(mock_openai, mock_embeddings, memory_engine, max_retries=3):
    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_embeddings.return_value = Mock()
    chat = DirectOpenAIChat(
        api_key="test-key",
        memory_engine=memory_engine,
        max_retries=max_retries,
        backoff_factor=1,
    )
    return chat, mock_client


def test_retry_succeeds_after_rate_limit(memory_engine):
    with patch("integrations.direct_openai.OpenAI") as mock_openai, \
         patch("integrations.direct_openai.OpenAIEmbeddings") as mock_embeddings, \
         patch("integrations.direct_openai.time.sleep", return_value=None):
        chat, mock_client = _setup_chat(mock_openai, mock_embeddings, memory_engine)

        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "AI response"
        http_response = Response(429, request=Request("POST", "http://test"))
        mock_client.chat.completions.create.side_effect = [
            RateLimitError("rate limit", response=http_response, body=None),
            response,
        ]

        result, _ = chat.chat("Hello", thread_id="t1", remember_response=False)

        assert result == "AI response"
        assert mock_client.chat.completions.create.call_count == 2


def test_retry_stops_after_max_attempts(memory_engine):
    with patch("integrations.direct_openai.OpenAI") as mock_openai, \
         patch("integrations.direct_openai.OpenAIEmbeddings") as mock_embeddings, \
         patch("integrations.direct_openai.time.sleep", return_value=None):
        chat, mock_client = _setup_chat(mock_openai, mock_embeddings, memory_engine, max_retries=2)

        http_request = Request("POST", "http://test")
        mock_client.chat.completions.create.side_effect = APITimeoutError(http_request)

        with pytest.raises(RuntimeError) as err:
            chat.chat("Hello", thread_id="t1", remember_response=False)

        assert "after 2 attempts" in str(err.value)
        assert mock_client.chat.completions.create.call_count == 2
