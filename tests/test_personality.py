from unittest.mock import patch, Mock
from integrations.direct_openai import DirectOpenAIChat
from core.personality import PersonalityProfile


def test_personality_prompt_included(memory_engine):
    with patch("integrations.direct_openai.OpenAI") as mock_openai, \
         patch("integrations.direct_openai.OpenAIEmbeddings") as mock_embeddings:
        mock_openai.return_value = Mock()
        mock_embeddings.return_value = Mock()
        chat = DirectOpenAIChat(api_key="test", memory_engine=memory_engine)
        profile = PersonalityProfile(
            extraversion=0.8,
            agreeableness=0.2,
            conscientiousness=0.9,
            neuroticism=0.1,
            openness=0.7,
        )
        messages = chat._build_messages_array(
            thread_id="t1",
            user_message="Hello",
            include_memories=0,
            personality=profile,
        )
        assert any(
            "extraversion 0.80" in m["content"] for m in messages if m["role"] == "system"
        )
