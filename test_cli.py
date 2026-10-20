import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from preannotate_cli import (
    parse_ssh_target,
    parser,
    read_env,
    resolve_remote_config,
)


class PreannotateCliTests(unittest.TestCase):
    def test_env_permissions(self):
        with TemporaryDirectory() as d:
            path = Path(d) / ".env"
            path.write_text(
                'PREANNOTATE_SSH_TARGET=gpu-server\nPREANNOTATE_WORKSPACE_USER="operador"\n',
                encoding="utf-8",
            )
            path.chmod(0o600)
            env = read_env(path)
            self.assertEqual(env["PREANNOTATE_SSH_TARGET"], "gpu-server")
            self.assertEqual(env["PREANNOTATE_WORKSPACE_USER"], "operador")
            if os.name == "posix":
                path.chmod(0o644)
                with self.assertRaises(ValueError):
                    read_env(path)

    def test_ssh_target_parsing(self):
        alias = parse_ssh_target("gpu-host")
        self.assertEqual(alias["host"], "gpu-host")
        self.assertEqual(alias["port"], "")

        with_flags = parse_ssh_target("operador@10.0.0.5 -p 2222")
        self.assertEqual(with_flags["host"], "operador@10.0.0.5")
        self.assertEqual(with_flags["port"], "2222")

        url = parse_ssh_target("ssh://operador@10.0.0.5:2200")
        self.assertEqual(url["host"], "operador@10.0.0.5")
        self.assertEqual(url["port"], "2200")

        with self.assertRaises(ValueError):
            parse_ssh_target("")

    def test_remote_workspace_resolution(self):
        with TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "PREANNOTATE_SSH_TARGET=gpu-server",
                        "PREANNOTATE_REMOTE_ROOT=/home/cluster",
                        "PREANNOTATE_WORKSPACE_USER=operador",
                        "PREANNOTATE_WORKSPACE_DIR=pasta_do_anotador",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            env_path.chmod(0o600)
            args = parser().parse_args(["--env", str(env_path), "remote-check"])
            cfg = resolve_remote_config(args)
            self.assertEqual(
                cfg["workspace_dir"], "/home/cluster/operador/pasta_do_anotador"
            )
            self.assertEqual(
                cfg["remote_venv"], "/home/cluster/operador/pasta_do_anotador/.venv"
            )

    def test_command_reference_covers_every_subcommand(self):
        choices = next(
            action.choices
            for action in parser()._actions
            if hasattr(action, "choices") and action.choices
        )
        root = Path(__file__).parent
        reference = (root / "docs" / "CLI.md").read_text(encoding="utf-8")
        for command in choices:
            with self.subTest(command=command):
                self.assertIn(f"`{command}", reference)

    def test_no_proprietary_brand_or_credentials_in_repo(self):
        forbidden_terms = ("get" + "ter", "jeand" + "erson", "samar" + "co")
        root = Path(__file__).parent
        checked_files = [
            root / "README.md",
            root / "docs" / "CLI.md",
            root / "perfil.env.example",
            root / "pyproject.toml",
            root / "preannotate_cli.py",
            root / "preannotate_core.py",
            root / "preannotate_worker.py",
            root / "test_cli.py",
            root / "test_core.py",
        ]
        for file_path in checked_files:
            with self.subTest(file=file_path.name):
                content = file_path.read_text(encoding="utf-8").casefold()
                for term in forbidden_terms:
                    self.assertNotIn(term, content)


if __name__ == "__main__":
    unittest.main()
