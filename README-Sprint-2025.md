# FRAME-FM December Sprint (04/12/2025)

This README covers: 
- The Agenda
- Logistics - working on JASMIN

## Agenda

Here is our plan for the sprint.

**Morning session (0900-1230)**

1. Introduction to the sprint (this document)
2. Quick review of an example PyTorch-Lightning workflow (as a visual cue)
3. Plan sprint teams, include a lead and scribe for each:
   - Models: Spatial auto-encoders (static data) & Temporal auto-encoders (temporal and spatial)
   - Data: Building the data loaders
   - Monitoring (later): Testing and deciding on monitoring/visualisation tools
4. Inter-team discussions						

**Afternoon session (1330-1600)**

1. Recap what was learnt
2. Plan the afternoon
3. Regroup at 1500:
  - Record key findings
  - Write down plans and next steps

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

**Software environments: in the SSH world**

If you have connected via SSH, then you will use a Python virtual environment which can be set up with 
by running:

```bash
source /gws/ssde/j25b/eds_ai/frame-fm/setup-sprint.sh
```

This command will give you access to the environment listed in the [pyproject.toml](https://github.com/NERC-EDS/FRAME-FM/blob/sprint-dec-2025/pyproject.toml#L8-L30) file.

**Software environments: in the Notebook world**

In the Notebook Service, you have to set up the environment as an `ipython kernel` within your own 
Notebook configuration. Here is how you do it:

1. Select `File --> New --> Terminal` from the menu.
2. Within your `bash` terminal, run these commands:

```bash
source /gws/ssde/j25b/eds_ai/frame-fm/code/envs/sprint_env_nb/bin/activate
python -m ipykernel install --user --name sprint_env_nb
```

It should look like this:

<img width="1016" height="337" alt="image" src="https://github.com/user-attachments/assets/98c2e098-f863-41f4-885c-a0cb8ad5f3b3" />

You may need to reload the web page at this point.

Now, when you click the blue "+" button, you should see `sprint_env_nb` offered as kernel that you can 
select to start a Notebook:

<img width="1010" height="616" alt="image" src="https://github.com/user-attachments/assets/d9e7041c-3d78-416a-b5b9-9837e842eb03" />

**Datasets**

The input data for the sprint is found at:

```
/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/
```

**GPU Access: interactive**

If you want to work on an interactive GPU node in the SSH world, make sure you are on a `sci` server and 
type this command:

```bash
srun --gres=gpu:1 --mem=64g --partition=orchid --account=orchid --qos=orchid --time=01:00:00 --pty /bin/bash
```

This will typically take about a minute to queue an interactive slot, and log you in to one of the GPU hosts. 
The settings mean: 1 GPU, 64GB RAM, for a duration of 1 hour.

**JASMIN Accounts**

Once you have a JASMIN account, please check that you have access to the `orchid` and `eds_ai` roles in 
the JASMIN accounts portal, at: https://accounts.jasmin.ac.uk/services/my_services/

**Note about creating your own environments**

You may want to explore and build your own software environments. If so, we recommend using the UV package, 
as documented in Tom's excellent instructions here: 

    https://github.com/NERC-EDS/FRAME-FM/blob/feature/toms-examples/uv-intro.md


** Teams for the sprint:**
Data Loaders:
.csv loading with metadata - Colin
transformations - Matt
intermediary cached data - Matthew
shapefile loading - Michael


Modelling:
logging intermediary classes: Adam 
Modelling from Nature paper: Anastasia and Jeremy
Geospatial data sampling research : Tom

