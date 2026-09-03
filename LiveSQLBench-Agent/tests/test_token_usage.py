import unittest
from types import SimpleNamespace

from system_agent.adk_runtime import AdkRuntime


class TokenUsageTests(unittest.TestCase):
    def test_provider_usage_is_normalized(self):
        usage = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            total_token_count=125,
            cached_content_token_count=60,
            thoughts_token_count=5,
            tool_use_prompt_token_count=3,
        )
        result = AdkRuntime._serialize_usage(usage)
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 20)
        self.assertEqual(result["cached_input_tokens"], 60)
        self.assertEqual(result["uncached_input_tokens"], 40)
        self.assertEqual(result["total_tokens"], 125)

    def test_usage_is_summed_across_model_events(self):
        result = AdkRuntime._sum_usage([
            {"usage": {"input_tokens": 100, "output_tokens": 10,
                       "total_tokens": 110, "cached_input_tokens": 0,
                       "uncached_input_tokens": 100}},
            {"usage": {"input_tokens": 120, "output_tokens": 20,
                       "total_tokens": 140, "cached_input_tokens": 80,
                       "uncached_input_tokens": 40}},
            {"type": "user_message"},
        ])
        self.assertEqual(result["input_tokens"], 220)
        self.assertEqual(result["total_tokens"], 250)
        self.assertEqual(result["cached_input_tokens"], 80)
        self.assertEqual(result["usage_event_count"], 2)


if __name__ == "__main__":
    unittest.main()
