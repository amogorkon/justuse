import sys
from logging import getLogger

from furl import furl as URL

from ._parse_name import _parse_name

from ._filtered_and_ordered_data import _filtered_and_ordered_data
from ._ensure_path import _ensure_path
from ...pypi_model import Version

log = getLogger(__name__)


def _find_or_install(
    name, version=None, artifact_path=None, url=None, out_info=None, force_install=False
):
    log.debug(
        "_find_or_install(name=%s, version=%s, artifact_path=%s, url=%s)",
        name,
        version,
        artifact_path,
        url,
    )
    if out_info is None:
        out_info = {}
    info = out_info
    package_name, rest = _parse_name(name)

    if isinstance(url, str):
        url = URL(url)
    filename = artifact_path.name if artifact_path else None
    if url:
        filename = url.asdict()["path"]["segments"][-1]
    else:
        filename = artifact_path.name if artifact_path else None
    if filename and not artifact_path:
        artifact_path = sys.modules["use"].home / "packages" / filename

    if not url or not artifact_path or (artifact_path and not artifact_path.exists()):
        info.update(
            _filtered_and_ordered_data(_get_package_data(package_name), version)[
                0
            ].dict()
        )
        url = URL(str(info["url"]))
        filename = url.asdict()["path"]["segments"][-1]
        artifact_path = sys.modules["use"].home / "packages" / filename
    out_info["artifact_path"] = artifact_path

    # todo: set info
    as_dict = info
    url = URL(as_dict["url"])
    filename = url.path.segments[-1]
    info["filename"] = filename
    # info.update(_parse_filename(filename))
    info = {**info, "version": Version(version)}
    if not artifact_path.exists():
        artifact_path = _ensure_path(_download_artifact(name, version, filename, url))

    out_info.update(info)
    install_item = artifact_path
    meta = archive_meta(artifact_path)
    import_parts = re.split("[\\\\/]", meta["import_relpath"])
    if "__init__.py" in import_parts:
        import_parts.remove("__init__.py")
    import_name = ".".join(import_parts)
    name = f"{package_name}.{import_name}"
    relp = meta["import_relpath"]
    out_info["module_path"] = relp
    out_info["import_relpath"] = relp
    out_info["import_name"] = import_name
    if not force_install and _pure_python_package(artifact_path, meta):
        log.info(f"pure python pkg: {package_name, version, use.home}")
        return out_info

    venv_root = _venv_root(package_name, version, use.home)
    out_info["installation_path"] = venv_root
    python_exe = Path(sys.executable)
    env = _get_venv_env(venv_root)
    module_paths = venv_root.rglob(f"**/{relp}")
    if force_install or (not python_exe.exists() or not any(module_paths)):
        log.info("calling pip to install install_item=%s", install_item)

        # If we get here, the venv/pip setup is required.
        output = _process(
            python_exe,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "--no-color",
            "install",
            "--pre",
            "--root",
            PureWindowsPath(venv_root).drive
            if isinstance(venv_root, (WindowsPath, PureWindowsPath))
            else "/",
            "--prefix",
            str(venv_root),
            "--progress-bar",
            "ascii",
            "--prefer-binary",
            "--exists-action",
            "i",
            "--ignore-installed",
            "--no-warn-script-location",
            "--force-reinstall",
            "--no-warn-conflicts",
            install_item,
        )
        sys.stderr.write("\n\n".join((output.stderr, output.stdout)))

    site_dir, module_path = _find_module_in_venv(package_name, version, relp)
    module_paths = []
    if module_path:
        module_paths.append(module_path)
        installation_path = site_dir

    out_info.update(**meta)
    assert module_paths

    log.info("installation_path = %s", installation_path)
    log.info("module_path = %s", module_path)
    out_info.update(
        {
            **info,
            "artifact_path": artifact_path,
            "installation_path": installation_path,
            "module_path": module_path,
            "import_relpath": ".".join(relp.split("/")[0:-1]),
            "info": info,
        }
    )
    return _delete_none(out_info)
