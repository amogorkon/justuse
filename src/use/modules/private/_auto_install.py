def _auto_install(
    func=None,
    *,
    name,
    version,
    package_name,
    rest,
    hash_algo=Hash.sha256.name,
    self=None,
    **kwargs,
) -> Union[ModuleType, BaseException]:
    package_name, rest = _parse_name(name)

    if func:
        result = func(**all_kwargs(_auto_install, locals()))
        if isinstance(result, ModuleType):
            return result

    query = self.execute_wrapped(
        f"""
        SELECT
            artifacts.id, import_relpath,
            artifact_path, installation_path, module_path
        FROM distributions
        JOIN artifacts ON artifacts.id = distributions.id
        WHERE name='{package_name}' AND version='str(version)'
        ORDER BY artifacts.id DESC
        """
    ).fetchone()

    if not query or not _ensure_path(query["artifact_path"]).exists():
        query = _find_or_install(package_name, version)

    artifact_path = _ensure_path(query["artifact_path"])
    if _ensure_path(query["module_path"]) == Path(
        "/data/media/0/src/use/src/twisted/__init__.py"
    ):
        raise BaseException("module_path")
    module_path = _ensure_path(query["module_path"])
    # trying to import directly from zip
    _clean_sys_modules(rest)
    try:
        importer = zipimport.zipimporter(artifact_path)
        return importer.load_module(query["import_name"])
    except BaseException as zerr:
        pass
    orig_cwd = Path.cwd()
    mod = None
    if "installation_path" not in query:
        query = _find_or_install(package_name, version, force_install=True)
        artifact_path = _ensure_path(query["artifact_path"])
        module_path = _ensure_path(query["module_path"])
    assert "installation_path" in query
    assert query["installation_path"]
    installation_path = _ensure_path(query["installation_path"])
    try:
        module_path = _ensure_path(query["module_path"])
        os.chdir(installation_path)
        import_name = (
            str(module_path.relative_to(installation_path))
            .replace("\\", "/")
            .replace("/__init__.py", "")
            .replace("-", "_")
        )
        return (
            mod := _load_venv_entry(
                package_name,
                import_name,
                module_path=module_path,
                installation_path=installation_path,
            )
        )

    finally:
        os.chdir(orig_cwd)
        if "fault_inject" in config:
            config["fault_inject"](**locals())
        if mod:
            use._save_module_info(
                name=package_name,
                import_relpath=str(
                    _ensure_path(module_path).relative_to(installation_path)
                ),
                version=version,
                artifact_path=artifact_path,
                hash_value=hash_algo.value(artifact_path.read_bytes()).hexdigest(),
                module_path=module_path,
                installation_path=installation_path,
            )
