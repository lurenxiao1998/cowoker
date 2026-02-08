import unittest

from main import list_files, read_file

class TestRepoTools(unittest.TestCase):
    def test_read_a_json(self):
        # read existing a.json in repo
        out = read_file.invoke({"path": "a.json"})
        self.assertIsNotNone(out)
        self.assertIn("\"messages\"", out)

    def test_list_root_contains_a_json(self):
        out = list_files.invoke({"glob": "**/*", "max_files": 100})
        self.assertIsInstance(out, str)
        self.assertIn("a.json", out)


if __name__ == '__main__':
    unittest.main()
