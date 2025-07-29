
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from openai import RateLimitError

# Make sure the app path is added to sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_manager import LLMManager, managed_chat_request

# Mock response object that the chat engine would return
class MockResponse:
    def __init__(self, text):
        self.response = text

@pytest.fixture
def mock_llm_manager():
    """Fixture to create a mock LLMManager for testing."""
    with patch('src.llm_manager.os.getenv') as mock_getenv:
        # Mock environment variables for keys and models
        mock_getenv.side_effect = lambda key: {
            "OPENROUTER_API_KEY_1": "key1",
            "OPENROUTER_API_KEY_2": "key2",
            "OPENROUTER_MODELS": "model1,model2"
        }.get(key)

        manager = LLMManager()
        # Ensure it has more than one configuration to test fallback
        assert len(manager.configurations) > 1
        return manager

@pytest.mark.asyncio
async def test_managed_chat_request_handles_rate_limit_and_succeeds(mock_llm_manager):
    """ 
    Verify that the managed_chat_request function can handle a RateLimitError,
    switch to a new configuration, and succeed on the second attempt.
    """
    # 1. Mock the 'source' object (which would be a VectorStoreIndex)
    mock_index = MagicMock()
    
    # 2. Mock the chat engine that the index would create
    mock_chat_engine = MagicMock()
    
    # 3. Set up the behavior for the achat method
    # First call: Raise a RateLimitError
    # Second call: Return a successful response
    mock_chat_engine.achat = AsyncMock(side_effect=[
        RateLimitError("Rate limit exceeded", response=MagicMock(), body=None),
        MockResponse("Successful response")
    ])
    
    # 4. Mock the 'as_chat_engine' method on the index to return our mock engine
    mock_index.as_chat_engine.return_value = mock_chat_engine

    # 5. Spy on the 'switch_to_next_config' method to ensure it's called
    with patch.object(mock_llm_manager, 'switch_to_next_config', wraps=mock_llm_manager.switch_to_next_config) as spy_switch:
        # --- Execute the function under test ---
        result = await managed_chat_request(mock_index, "test question", mock_llm_manager)

        # --- Assertions ---
        # Verify the final result is correct
        assert result["response"] == "Successful response"

        # Verify that the fallback mechanism was triggered
        spy_switch.assert_called_once()

        # Verify that as_chat_engine was called twice (once for each attempt)
        assert mock_index.as_chat_engine.call_count == 2

        # Verify that the LLM was passed directly, not monkey-patched
        # The call_args_list stores the arguments of each call
        first_call_kwargs = mock_index.as_chat_engine.call_args_list[0].kwargs
        second_call_kwargs = mock_index.as_chat_engine.call_args_list[1].kwargs

        assert 'llm' in first_call_kwargs
        assert 'llm' in second_call_kwargs
        # Ensure the LLM instance was different between the two calls
        assert first_call_kwargs['llm'] is not second_call_kwargs['llm']
