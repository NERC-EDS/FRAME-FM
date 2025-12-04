# FRAME-FM December Sprint (04/12/2025)

This README covers: 
- The Agenda
- Logistics - working on JASMIN

## Agenda

Here is our plan for the sprint:


The notes of about the plans are in this spreadsheet:

    ​FRAME-FM-WP3-Roadmap.xlsx​

## Logistics - working on JASMIN

We have chosen JASMIN as the platform for the sprint because it is quite straightforward to bring everyone 
on to the same platform. Here are details to navigate around JASMIN:

**General JASMIN Help pages** 

See: https://help.jasmin.ac.uk/

JASMIN has two main ways that you can interact, via an SSH (terminal) system, or through a Jupyter Notebook 
Service. 

**SSH Login**

See instructions: https://help.jasmin.ac.uk/docs/getting-started/how-to-login/

Note that you will start by connecting to a login (gateway) machine, and then you can make an onward 
connection to one of the `sci` (scientific analysis) servers:

https://help.jasmin.ac.uk/docs/interactive-computing/sci-servers/

Once, on a `sci` server, you can access data and run code. See **GPU Access** below if you want to run 
on the GPU nodes.

**Notebook Service**

Login here: https://notebooks.jasmin.ac.uk

NOTE: You should select the "GPU" option if you require access to GPUs/CUDA.

**Working directories for the Sprint**

You should all have read/write access to the `eds_ai` Group Workspace. This is mounted across the `sci` 
servers, the Notebook Service and the Slurm cluster (including "ORCHID" GPU nodes). Please check you 
can access it at:

```
/gws/ssde/j25b/eds_ai/frame-fm
```

You can set yourself up a working directory for the sprint, using:

```
mydir=/gws/ssde/j25b/eds_ai/frame-fm/users/$USER
mkdir -p $mydir
cd $mydir/
```

**GPU Access**

If you want 

**JASMIN Accounts**

Once you have a JASMIN account, please check that you have access to the `orchid` and `eds_ai` roles in 
the JASMIN accounts portal, at: https://accounts.jasmin.ac.uk/services/my_services/



Through both SSH access and the Jupyter Notebook access, you will arrive within your $HOME directory. When working on the sprint we plan to use the eds_ai Group Workspace which currently has 10TB of free space. It is located here on the file system:

    

    Please check you can navigate in to this directory, and check you can create your own user directory within it using:

    mkdir /gws/ssde/j25b/eds_ai/frame-fm/users/$USER

    Diane and Adrian have been busy pulling in the required datasets to this sub-directory:

    /gws/ssde/j25b/eds_ai/frame-fm/data/inputs/

If you are working in the SSH environment, you can can use the srun command on a sci server to connect interactively to a GPU node on the "ORCHID" cluster, using this command:

    srun --gres=gpu:1 --mem=64g --partition=orchid --account=orchid --qos=orchid --time=01:00:00 --pty /bin/bash

    This will typically take about a minute to queue an interactive slot, and log you in to one of the GPU hosts. The settings mean: 1 GPU, 64GB RAM, for a duration of 1 hour.

Software environment: we are working on setting up a software environment from this recipe:

    https://github.com/NERC-EDS/FRAME-FM/blob/feature/toms-examples/pyproject.toml

    We will use the UV package to install it, following Tom's excellent instructions here: 

    https://github.com/NERC-EDS/FRAME-FM/blob/feature/toms-examples/uv-intro.md

We are working out of this repository:

    https://github.com/NERC-EDS/FRAME-FM

    We will create a branch for the sprint and install the software environment before Thursday.
