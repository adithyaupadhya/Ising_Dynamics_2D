import contextlib
import io
import unittest
from unittest.mock import patch
from ising.cli import build_parser, main

class TestCLI(unittest.TestCase):

    def test_readable_options(self):
        a = build_parser().parse_args(['passage', '--size', '8', '16', '--field', '3', '--workers', '2'])
        self.assertEqual(a.N, [8, 16])
        self.assertEqual(a.lam, 3)
        self.assertEqual(a.workers, 2)

    def test_temperature(self):
        with patch('ising.cli.cmd_series') as handler:
            main(['series', '--temperature', '2'])
            self.assertEqual(handler.call_args.args[0].beta, 0.5)

    def test_invalid_options(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for args in [['series', '--size', '0'], ['series', '--beta', '.5', '--temperature', '2'], ['check', '--size', '5'], ['passage', '--threshold', '2']]:
                with self.assertRaises(SystemExit):
                    main(args)

    def test_no_command_help(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main([])
        self.assertIn('coalescence', out.getvalue())
