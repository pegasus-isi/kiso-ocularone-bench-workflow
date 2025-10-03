VIP navigation requires multiple DNN models for identification, posture analysis, and depth estimation to ensure safe mobility. Using a hazard vest as a unique identifier enhances visibility while selecting the right DNN model and computing device balances accuracy and real-time performance. We present Ocularone-Bench, which is a benchmark suite designed to address the lack of curated datasets for uniquely identifying individuals in crowded environments and the need for benchmarking DNN inference times on resource-constrained edge devices. The suite evaluates the accuracy-latency trade-offs of YOLO models retrained on this dataset and benchmarks inference times of situation awareness models across edge accelerators and high-end GPU workstations.

> [!IMPORTANT]
> The experiment can be run only on ARM-based machines.

# Getting Started

## Prerequisites

```sh
pip install kiso
pip install kiso[vagrant]
```

## Running the experiment

Create application credentials for CHI@TACC as tacc-app-cred-oac-edge-openrc.sh and place the files in the secrets directory.

## Running the experiment

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

# References

- [Pegasus Workflow Management System](https://pegasus.isi.edu)
- [EnOSlib](https://discovery.gitlabpages.inria.fr/enoslib/)
- [Chameleon Cloud](https://www.chameleoncloud.org)
- [FABRIC](https://portal.fabric-testbed.net)

# Citation

[Suman Raj, Bhavani A Madhabhavi, Kautuk Astu, Arnav A Rajesh, Pratham M and Yogesh Simmhan (2025). Ocularone-Bench: Benchmarking DNN Models on GPUs to Assist the Visually Impaired. Ocularone Dataset](https://www.arxiv.org/pdf/2504.03709)

# Acknowledgements

Kiso is funded by National Science Foundation (NSF) under award [2403051](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2403051).
