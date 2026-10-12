# -*- coding: utf-8 -*-
import unittest

from apps.charsheet import jobs as jobs_module
from apps.charsheet.jobs import _init_job, _normalize_view_keys, _view_prompt_for_size


class CharsheetSizePromptTests(unittest.TestCase):
    def test_selected_views_are_deduplicated_in_request_order(self):
        self.assertEqual(
            _normalize_view_keys("front,left,front"),
            ["front", "left"],
        )

    def test_unknown_view_is_rejected(self):
        with self.assertRaises(ValueError):
            _normalize_view_keys("front,unknown")

    def test_job_total_matches_only_requested_views(self):
        job_id = "selected-view-test"
        _init_job(job_id, 1, generation_size=768, view_keys=["front", "left"])
        try:
            self.assertEqual(jobs_module.jobs[job_id]["total"], 2)
            self.assertEqual(
                [view["key"] for view in jobs_module.jobs[job_id]["views"]],
                ["front", "left"],
            )
        finally:
            jobs_module.jobs.pop(job_id, None)

    def test_default_size_keeps_medium_shot(self):
        prompt = "<sks> front view eye-level shot medium shot"
        self.assertEqual(_view_prompt_for_size(prompt, None), prompt)
        self.assertEqual(_view_prompt_for_size(prompt, 1024), prompt)

    def test_small_size_uses_supported_wide_shot_tag(self):
        prompt = "<sks> front view eye-level shot medium shot"
        for size in (512, 624, 768):
            result = _view_prompt_for_size(prompt, size)
            self.assertTrue(result.startswith("<sks> front view eye-level shot wide shot."))
            self.assertIn("top of the head to both feet", result)
            self.assertIn("clear margin on every side", result)

    def test_natural_language_prompt_gets_full_body_framing(self):
        prompt = "Show the character from the front"
        result = _view_prompt_for_size(prompt, 512)
        self.assertIn("wide full-body composition", result)
        self.assertIn("both feet", result)


if __name__ == "__main__":
    unittest.main()
