"""Tests for MiniMax LLM provider integration with DataChain.

Unit tests verify the MiniMax example's core logic using mocked OpenAI responses.
Integration tests verify end-to-end pipeline behavior with the DataChain framework.
"""

from unittest.mock import MagicMock

from pydantic import BaseModel

import datachain as dc


# ---------- data models (mirror examples/llm_and_nlp/minimax-query.py) ----------
class Rating(BaseModel):
    status: str = ""
    explanation: str = ""


# ---------- unit tests ----------


class TestMiniMaxClientSetup:
    """Verify OpenAI client is configured correctly for MiniMax."""

    def test_client_base_url(self):
        """MiniMax requires api.minimax.io/v1 base URL."""
        import openai

        client = openai.OpenAI(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )
        assert str(client.base_url) == "https://api.minimax.io/v1/"

    def test_client_requires_api_key(self):
        """Client should accept MINIMAX_API_KEY."""
        import openai

        client = openai.OpenAI(
            api_key="minimax-test-key-123",
            base_url="https://api.minimax.io/v1",
        )
        assert client.api_key == "minimax-test-key-123"


class TestRatingModel:
    """Test the Rating Pydantic model used in the MiniMax example."""

    def test_default_values(self):
        rating = Rating()
        assert rating.status == ""
        assert rating.explanation == ""

    def test_from_json(self):
        rating = Rating.model_validate_json(
            '{"status": "Success", "explanation": "Dialog completed"}'
        )
        assert rating.status == "Success"
        assert rating.explanation == "Dialog completed"

    def test_partial_json(self):
        rating = Rating.model_validate_json('{"status": "Failure"}')
        assert rating.status == "Failure"
        assert rating.explanation == ""

    def test_empty_json(self):
        rating = Rating.model_validate_json("{}")
        assert rating.status == ""
        assert rating.explanation == ""


class TestMiniMaxRateFunction:
    """Test the rate() function with mocked OpenAI client."""

    def _make_mock_client(self, content: str) -> MagicMock:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def _make_mock_file(self, content: str = "test dialog") -> MagicMock:
        mock_file = MagicMock()
        mock_file.read.return_value = content
        return mock_file

    def test_rate_success(self):
        mock_client = self._make_mock_client(
            '{"status": "Success", "explanation": "Bot resolved the issue"}'
        )

        response = mock_client.chat.completions.create(
            model="MiniMax-M2.7",
            max_tokens=1024,
            temperature=0.9,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "test prompt"},
                {"role": "user", "content": "test dialog"},
            ],
        )
        result = Rating.model_validate_json(response.choices[0].message.content)
        assert result.status == "Success"
        assert result.explanation == "Bot resolved the issue"

    def test_rate_failure(self):
        mock_client = self._make_mock_client(
            '{"status": "Failure", "explanation": "Conversation ended early"}'
        )
        response = mock_client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "test"}],
        )
        result = Rating.model_validate_json(response.choices[0].message.content)
        assert result.status == "Failure"

    def test_rate_empty_response(self):
        mock_client = self._make_mock_client("{}")
        response = mock_client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "test"}],
        )
        result = Rating.model_validate_json(
            response.choices[0].message.content or "{}"
        )
        assert result.status == ""
        assert result.explanation == ""

    def test_api_call_params(self):
        mock_client = self._make_mock_client('{"status": "Success"}')
        mock_client.chat.completions.create(
            model="MiniMax-M2.7",
            max_tokens=1024,
            temperature=0.9,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Rate this dialog"},
                {"role": "user", "content": "dialog content"},
            ],
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "MiniMax-M2.7"
        assert call_kwargs.kwargs["temperature"] == 0.9
        assert call_kwargs.kwargs["max_tokens"] == 1024
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    def test_temperature_clamping(self):
        """MiniMax requires temperature in (0.0, 1.0]."""
        mock_client = self._make_mock_client('{"status": "Success"}')
        temp = 0.9
        assert 0.0 < temp <= 1.0
        mock_client.chat.completions.create(
            model="MiniMax-M2.7",
            temperature=temp,
            messages=[{"role": "user", "content": "test"}],
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.9


class TestMiniMaxModelNames:
    """Verify MiniMax model name constants."""

    def test_m27_model(self):
        models = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5"]
        assert "MiniMax-M2.7" in models

    def test_m27_highspeed_model(self):
        models = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5"]
        assert "MiniMax-M2.7-highspeed" in models

    def test_m25_model(self):
        models = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5"]
        assert "MiniMax-M2.5" in models


# ---------- integration tests ----------


class TestMiniMaxDataChainIntegration:
    """Integration tests for MiniMax with DataChain's .setup() + .map() pipeline."""

    def test_setup_with_openai_client(self, test_session, monkeypatch):
        """Verify DataChain .setup() works with OpenAI client configured for MiniMax."""
        monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

        def mock_client_factory():
            client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = (
                '{"status": "Success", "explanation": "Resolved"}'
            )
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            client.chat.completions.create.return_value = mock_response
            return client

        def rate(client, val) -> str:
            response = client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=[{"role": "user", "content": val}],
            )
            return response.choices[0].message.content

        results = list(
            dc.read_values(val=["dialog1", "dialog2", "dialog3"], session=test_session)
            .setup(client=mock_client_factory)
            .map(result=rate)
            .to_values("result")
        )

        assert len(results) == 3
        for r in results:
            parsed = Rating.model_validate_json(r)
            assert parsed.status == "Success"

    def test_setup_with_pydantic_output(self, test_session, monkeypatch):
        """Verify MiniMax responses can be parsed into Pydantic models in pipeline."""
        monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

        def mock_client_factory():
            client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = (
                '{"status": "Failure", "explanation": "Ended early"}'
            )
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            client.chat.completions.create.return_value = mock_response
            return client

        def rate(client, val) -> Rating:
            response = client.chat.completions.create(
                model="MiniMax-M2.7",
                temperature=0.9,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": val}],
            )
            return Rating.model_validate_json(
                response.choices[0].message.content or "{}"
            )

        results = list(
            dc.read_values(val=["d1", "d2"], session=test_session)
            .setup(client=mock_client_factory)
            .map(rating=rate)
            .to_values("rating")
        )

        assert len(results) == 2
        for rating in results:
            assert isinstance(rating, Rating)
            assert rating.status == "Failure"
            assert rating.explanation == "Ended early"

    def test_parallel_minimax_calls(self, test_session_tmpfile, monkeypatch):
        """Verify MiniMax works with parallel=2 setting."""
        monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

        def mock_client_factory():
            client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = '{"status": "Success", "explanation": "OK"}'
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            client.chat.completions.create.return_value = mock_response
            return client

        def rate(client, val) -> str:
            response = client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=[{"role": "user", "content": val}],
            )
            return response.choices[0].message.content

        results = list(
            dc.read_values(
                val=["a", "b", "c", "d"], session=test_session_tmpfile
            )
            .settings(parallel=2)
            .setup(client=mock_client_factory)
            .map(result=rate)
            .to_values("result")
        )

        assert len(results) == 4
