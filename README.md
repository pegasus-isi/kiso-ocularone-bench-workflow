The Kiso repository is a template that can be used to start a new Kiso experiment project. The repository defines the directory structure, best practices, etc. to be used while developing an experiment in Kiso.

# Getting Started

## Prerequisites

```sh
pip install kiso
# Install the resource provider specific dependencies you want to use
pip install kiso[vagrant]
pip install kiso[chameleon]
```

## Defining the experiment

Define your experiments and the resources it needs in a YAML file named `experiment.yml`.

## Running the experiment

Place any required credentials files in the `secrets` directory. These credentials are referenced in the experiment configuration YAML file.

Place any required config files in the `config` directory.

Place any required data files in the `data` directory.

```sh
# Check Kiso experiment configuration
kiso check

# Provision and setup the resources
kiso up

# Run the experiments defined in the experiment configuration YAML file
kiso run

# Destroy the provisioned resources
kiso down

# Pegasus workflow submit directories will be placed in the output directory at the end of the experiment. The submit directories will also have a statistics directory with the pegasus-statistics output.
# Outputs defined in the experiment configuration will be placed to the destination specified in the experiment configuration.
```

# Versioning

```sh
pip install commitizen

# Committing changes
# Use git cz c or cz c to commit changes

# Tagging a new version, updating the changelog
cz bump
git push --tags

# Use GitHub CLI to create a new release
# gh release create --repo <user-or-org>/<repo> <tag-name>
```

# References

- [Pegasus Workflow Management System](https://pegasus.isi.edu)
- [EnOSlib](https://discovery.gitlabpages.inria.fr/enoslib/)
- [Chameleon Cloud](https://www.chameleoncloud.org)
- [FABRIC](https://portal.fabric-testbed.net)

# Acknowledgements

Kiso is funded by National Science Foundation (NSF) under award [2403051](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2403051).
