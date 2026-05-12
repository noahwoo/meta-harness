"""Smoke tests for OpenRouter LLM providers."""

from __future__ import annotations

import os
import unittest

from text_classification.llm import ProviderLLM


@unittest.skipUnless(
    os.environ.get("OPENROUTER_API_KEY"),
    "OPENROUTER_API_KEY not set, skipping live API test",
)
class TestOpenRouterGptOss(unittest.TestCase):
    """Verify openrouter/openai/gpt-oss-120b returns valid responses."""

    MODEL = "openrouter/openai/gpt-oss-120b"

    def setUp(self):
        self.llm = ProviderLLM(model=self.MODEL, max_concurrent=1)

    def test_generate_returns_nonempty_string(self):
        result = self.llm.generate("What is 2+2? Reply with just the number.")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        content = result[0][0]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content.strip()), 0)

    def test_generate_tracks_token_usage(self):
        self.llm.generate("Say hello.")
        self.assertGreater(self.llm.total_input_tokens, 0)
        self.assertGreater(self.llm.total_output_tokens, 0)

    def test_generate_math_answer(self):
        result = self.llm.generate("What is 7 * 6? Reply with just the number.")
        content = result[0][0].strip()
        self.assertIn("42", content)


@unittest.skipUnless(
    os.environ.get("OPENROUTER_API_KEY"),
    "OPENROUTER_API_KEY not set, skipping live API test",
)
class TestOpenRouterDeepSeekV3(unittest.TestCase):
    """Verify openrouter/baidu/deepseek-v3.2 (Baidu Qianfan BYOK) returns valid responses."""

    MODEL = "openrouter/baidu/deepseek-v3.2"

    def setUp(self):
        self.llm = ProviderLLM(model=self.MODEL, max_concurrent=1)

    def test_generate_returns_nonempty_string(self):
        result = self.llm.generate("What is 2+2? Reply with just the number.")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        content = result[0][0]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content.strip()), 0)

    def test_generate_tracks_token_usage(self):
        self.llm.generate("Say hello.")
        self.assertGreater(self.llm.total_input_tokens, 0)
        self.assertGreater(self.llm.total_output_tokens, 0)

    def test_generate_math_answer(self):
        result = self.llm.generate("What is 7 * 6? Reply with just the number.")
        content = result[0][0].strip()
        self.assertIn("42", content)

    def test_generate_chinese(self):
        result = self.llm.generate("用一个词回答：中国的首都是哪里？")
        content = result[0][0].strip()
        self.assertIn("北京", content)


@unittest.skipUnless(
    os.environ.get("QIANFAN_API_KEY"),
    "QIANFAN_API_KEY not set, skipping live API test",
)
class TestQianfanDeepSeekV3(unittest.TestCase):
    """Verify qianfan/deepseek-v3.2 (Baidu Qianfan Anthropic-compatible API) returns valid responses."""

    MODEL = "qianfan/deepseek-v3.2"

    def setUp(self):
        self.llm = ProviderLLM(model=self.MODEL, max_concurrent=1)

    def test_normalized_model(self):
        self.assertEqual(self.llm._normalized_model(), "anthropic/deepseek-v3.2")

    def test_generate_returns_nonempty_string(self):
        result = self.llm.generate("What is 2+2? Reply with just the number.")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        content = result[0][0]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content.strip()), 0)

    def test_generate_tracks_token_usage(self):
        self.llm.generate("Say hello.")
        self.assertGreater(self.llm.total_input_tokens, 0)
        self.assertGreater(self.llm.total_output_tokens, 0)

    def test_generate_math_answer(self):
        result = self.llm.generate("What is 7 * 6? Reply with just the number.")
        content = result[0][0].strip()
        self.assertIn("42", content)

    def test_generate_chinese(self):
        result = self.llm.generate("用一个词回答：中国的首都是哪里？")
        content = result[0][0].strip()
        self.assertIn("北京", content)

    def test_cost_is_zero(self):
        self.llm.generate("Say hi.")
        self.assertEqual(self.llm.total_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
