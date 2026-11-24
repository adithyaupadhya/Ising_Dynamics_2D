import json
import tempfile
import unittest
from pathlib import Path
from ising.experiments import first_passage, coalescence, run
from ising import IsingModel

def scientific(rows):
    return [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows]

class TestExperiments(unittest.TestCase):

    def test_parallel_matches_serial(self):
        for fn in (first_passage, coalescence):
            options = dict(sizes=[2, 3], replicates=2, limit=3, backend='python', seed=17)
            self.assertEqual(scientific(fn(**options).rows), scientific(fn(**options, workers=2).rows))

    def test_resume_after_interruption(self):
        options = dict(sizes=[2, 3], replicates=2, limit=3, backend='python', seed=17)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'study'

            def interrupt(done, total, row):
                if done == 2:
                    raise RuntimeError('simulated interruption')
            with self.assertRaises(RuntimeError):
                first_passage(**options, output=directory, progress=interrupt)
            with (directory / 'checkpoint.jsonl').open('a') as handle:
                handle.write('{partial')
            result = first_passage(**options, output=directory, resume=True)
            self.assertEqual(scientific(result.rows), scientific(first_passage(**options).rows))
            self.assertEqual(len((directory / 'checkpoint.jsonl').read_text().splitlines()), 4)
            self.assertTrue((directory / 'trials.csv').exists())
            self.assertIn('source_sha256', json.loads((directory / 'parameters.json').read_text())['environment'])
            with self.assertRaises(FileExistsError):
                first_passage(**options, output=directory)
            with self.assertRaises(ValueError):
                first_passage(**{**options, 'limit': 4}, output=directory, resume=True)

    def test_optional_trajectory_observables(self):
        _, rows = run(IsingModel(3), 2, record_every=1, observables={'contours': True, 'interface_heights': True})
        self.assertEqual(len(rows), 3)
        self.assertIn('droplet_area', rows[-1])
        self.assertIn('interface_heights', rows[-1])
