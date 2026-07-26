import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class DashboardImportTest(unittest.TestCase):
    def test_dashboard_import_is_safe_outside_streamlit(self):
        sys.modules.pop("dashboard", None)
        import dashboard

        self.assertTrue(callable(getattr(dashboard, "render_dashboard", None)))


if __name__ == "__main__":
    unittest.main()
