import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gpu_handoff import probe
from gpu_handoff.cli import Check, render_table


class UnreadableXmlTest(unittest.TestCase):
    @unittest.skipIf(os.geteuid() == 0, "root reads everything")
    def test_unreadable_xml_is_counted_and_rendered_as_warn(self):
        with TemporaryDirectory() as d:
            f = Path(d, "vm1.xml")
            f.write_text("<domain/>")
            f.chmod(0)
            self.assertEqual(probe.unreadable_xml(Path(d)), 1)
            self.assertIn("WARN", render_table([Check("vm references device", None, "x")]))


if __name__ == "__main__":
    unittest.main()
