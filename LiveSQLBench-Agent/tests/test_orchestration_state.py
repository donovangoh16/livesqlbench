import unittest
from unittest.mock import AsyncMock, patch

from experiment.variants import get_variant
from orchestrator.single_turn import init_agent_session


class OrchestrationStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_variant_four_sends_multi_orchestration_to_service(self):
        task = {
            "instance_id": "example_1",
            "selected_database": "example",
            "query": "Return one row",
        }
        with patch("orchestrator.single_turn._post", new=AsyncMock(return_value={})) as post:
            await init_agent_session("example_1", task, get_variant(4))
        state = post.await_args.args[1]["state"]
        self.assertEqual(state["requested_orchestration_profile"], "multi")
        self.assertEqual(state["active_orchestration_profile"], "multi")

    async def test_variant_one_remains_single_orchestration(self):
        task = {
            "instance_id": "example_1",
            "selected_database": "example",
            "query": "Return one row",
        }
        with patch("orchestrator.single_turn._post", new=AsyncMock(return_value={})) as post:
            await init_agent_session("example_1", task, get_variant(1))
        state = post.await_args.args[1]["state"]
        self.assertEqual(state["active_orchestration_profile"], "single")


if __name__ == "__main__":
    unittest.main()
