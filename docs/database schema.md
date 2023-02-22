This is the schema of the registry database for reference.

* Packages have a name and version, but are released as platform-dependent (or independent, as pure-python) distributions.
* Each distribution corresponds to a single downloadable and installable file, called an artifact.
* Each artifact has hashes that are generated using certain algorithms, which map to a very large number (no matter how this number is represented otherwise - as hexdigest or JACK or something else).
* Once a distribution has been installed the artifact could be removed (since everything is now unpacked, compiled etc), but it also can be kept for further P2P sharing.
* The distribution is installed in a venv, isolated.

```mermaid

  erDiagram

    artifacts {
    INTEGER id
    INTEGER distribution_id
    TEXT import_relpath
    TEXT artifact_path
    TEXT module_path
    }

    distributions {
      INTEGER id
      TEXT name
      TEXT version
      TEXT installation_path
      INTEGER date_of_installation
      INTEGER number_of_uses
      INTEGER date_of_last_use
      INTEGER pure_python_package
    }

    hashes {
      TEXT algo
      INTEGER value
      INTEGER artifact_id
    }

    hashes ||--o{ artifacts : "foreign key"
    artifacts ||--o{ distributions : "foreign key"

```

# use(URL)
In case of a single module being imported from a web-source, the module is cached in <home>/web-modules as a file with a 
random but valid module name. We keep track of the mapping via DB: artifact.artifact_path -> web-URI used to fetch the 
module, artifact.module_path -> module-file