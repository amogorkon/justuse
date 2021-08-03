import os
import sys
import warnings
from pathlib import Path
from setuptools import _find_all_simple
	
from unittest import skip

import pytest
import requests
	
from yarl import URL

from .unit_test import log, reuse

not_local = "GITHUB_REF" in os.environ


# Add in-progress tests here


def test_template(reuse):
    pass


def test_is_platform_compatible_macos(reuse):
    platform_tags = reuse.use.get_supported()
    platform_tag = next(iter(platform_tags))
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": f"numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "source",
        "requires_python": ">=3.6",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": "https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert reuse._is_platform_compatible(info, platform_tags)


def test_is_platform_compatible_win(reuse):
    platform_tags = reuse.use.get_supported()
    platform_tag = next(iter(platform_tags))
    info = {
        "comment_text": "",
        "digests": {
            "md5": "baf1bd7e3a8c19367103483d1fd61cfc",
            "sha256": "dbd18bcf4889b720ba13a27ec2f2aac1981bd41203b3a3b27ba7a33f88ae4827",
        },
        "downloads": -1,
        "filename": f"numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "has_sig": False,
        "md5_digest": "baf1bd7e3a8c19367103483d1fd61cfc",
        "packagetype": "bdist_wheel",
        "python_version": f"cp3{sys.version_info[1]}",
        "requires_python": f">=3.{sys.version_info[1]}",
        "size": 13227547,
        "upload_time": "2021-01-05T17:24:53",
        "upload_time_iso_8601": "2021-01-05T17:24:53.052845Z",
        "url": f"https://files.pythonhosted.org/packages/ea/bc/da526221bc111857c7ef39c3af670bbcf5e69c247b0d22e51986f6d0c5c2/numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert reuse._is_platform_compatible(info, platform_tags, include_sdist=False)


def load_mod(package, version):
  import shlex, subprocess, sys, importlib
  venv_root = \
    Path.home() / ".justuse-python" / "venv" / package / version
  if not venv_root.exists():
    venv_root.mkdir(parents=True)
  venv_bin = venv_root / "bin"
  if not venv_bin.exists():
    print(
      subprocess.check_output(
        [Path(sys.executable).stem, "-m", "venv", venv_root],
        shell=False, encoding="UTF-8"
      )
    )
  current_path = os.environ.get("PATH")
  pip_args = [
    "env",
    f"PATH={venv_bin}{os.path.pathsep}{current_path}",
    Path(sys.executable).stem,
    "-m", "pip",
    "--no-python-version-warning",
    "--disable-pip-version-check",
    "--no-color",
    "--no-cache-dir",
    "--isolated",
    "install",
    "--progress-bar", "ascii",
    "--prefer-binary",
    "--only-binary", ":all:",
    "--no-build-isolation",
    "--no-use-pep517",
    "--no-compile",
    "--no-warn-script-location",
    "--no-warn-conflicts",
    f"{package}=={version}",
  ]
  pkg_path = subprocess.check_output(
    [
      """
      package="{package}"; version="{version}"
      venv_root="{venv_root}"; venv_bin="{venv_bin}"
      for attempt in 1 2; do
        output="$( {pip_args} 2>&1 \
          | sed -u -e "w /dev/stderr" )"
        output_req="${{output##*satisfied: $package==$version in }}"
        [ "x$output" != "x$output_req" ] \
          && pkg_path="${{output_req%% \\(*}}" && break # "\\)"
      done
      test -n "$pkg_path" && test -d "$pkg_path" \
          && echo "$pkg_path" \
          || exit 255
      """.format(
        pip_args=shlex.join(pip_args),
        package=package,
        version=version,
        venv_root=venv_root,
        venv_bin=venv_bin,
      )
    ], shell=True, encoding="UTF-8"
  ).strip()
  if not Path(pkg_path).is_dir():
    raise OSError(f"Expected a directory at '{pkp_path}'")
  mod = None
  try:
    sys.path.insert(0, pkg_path)
    mod = importlib.import_module(package)
    return mod
  finally:
    sys.path.remove(pkg_path)

@pytest.mark.xfail(True, reason="in testing")
def test_load_venv_mod():
  mod = load_mod("numpy", "1.19.3")
  assert mod.__version__ == "1.19.3"

