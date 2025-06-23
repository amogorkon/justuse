from justuse.pimp import _is_version_satisfied


def test_is_version_satisfied(reuse):
    sys_version = reuse.Version("3.6.0")
    # google.protobuf 1.19.5 normal case
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": "google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "cp36",
        "requires_python": ">=3.6",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": "https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert _is_version_satisfied(info.get("requires_python", ""), sys_version)

    # requires >= python 4!
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": "google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "cp36",
        "requires_python": ">=4",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": "https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert False == _is_version_satisfied(
        info.get("requires_python", ""), reuse.Version(sys_version)
    )

    # pure python
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": "google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "source",
        "requires_python": ">=3.6",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": "https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert _is_version_satisfied(info.get("requires_python", ""), sys_version)
