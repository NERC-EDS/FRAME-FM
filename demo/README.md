# Environmental Mini Foundation Model (Soil Moisture)

This project builds a small **environmental foundation model** trained on multiple environmental datasets and then
fine-tuned on **COSMOS soil moisture measurements**.

The goal is to learn general environmental representations first, and then adapt them to predict soil moisture.

The repo is structured as a simple pipeline:

```
ingest datasets
    ↓
build token dataset
    ↓
train foundation model
    ↓
fine-tune on COSMOS
    ↓
generate predictions
```

This README explains what each script does and how to run the pipeline.

---

# Quick start

Install dependencies:

```
pip install -r requirements.txt
```

Then run the pipeline in this order (scroll to Notes before running):

```
python ingest_environmental_tokens.py
python build_store.py
python train_mini_foundation.py
python finetune_cosmos_foundation_focused.py
```

Each stage writes outputs used by the next stage.

---

# Repository structure

```
ingest_environmental_tokens.py
build_store.py
framefm_core.py
train_mini_foundation.py
finetune_cosmos_foundation_focused.py
requirements.txt
```

---

# Script descriptions

## ingest_environmental_tokens.py

First stage of the pipeline.

Loads environmental datasets and converts them into **feature tokens** for each spatial location.

Typical inputs:

* environmental rasters
* gridded climate data
* terrain / soil variables

Main steps:

* load datasets
* align them spatially
* extract feature values
* build a per-location feature vector

Outputs:

* processed environmental feature arrays
* location/index metadata

These outputs become the training corpus for the foundation model.

---

## build_store.py

Takes the processed feature outputs and converts them into an **efficient training dataset**.

Purpose:

* make training faster
* avoid loading everything into memory

Main steps:

* read processed environmental tokens
* reshape into training format
* write memmapped arrays / dataset shards
* write index metadata

Outputs:

* memmapped training arrays
* dataset index files

These are used directly by the training script.

---

## framefm_core.py

Contains the **core model architecture**.

Defines:

* environmental feature encoder
* backbone layers
* prediction heads

Both training and fine-tuning scripts import the model from here.

---

## train_mini_foundation.py

Pretrains the environmental model.

This stage learns general relationships between environmental variables before using any soil moisture labels.

Main steps:

* load dataset store
* initialise model
* train with the pretraining objective
* save checkpoints

Outputs:

* pretrained model weights
* training logs

---

## finetune_cosmos_foundation_focused.py

Fine-tunes the pretrained model using **COSMOS soil moisture observations**.

Main steps:

* load pretrained foundation model
* load COSMOS data
* match observations to environmental features
* train prediction head
* evaluate predictions

Outputs:

* fine-tuned model
* prediction outputs
* evaluation metrics
* CSV/JSON results

---

# Model idea (brief)

Each spatial location is represented as a vector of environmental variables:

```
x = [climate, terrain, soil, remote sensing, ...]
```

The foundation model learns a representation of these environmental conditions during pretraining.

During fine-tuning the model learns:

```
soil_moisture = f(environmental_features)
```

using COSMOS measurements.

---

# Notes

Things that may need adjustment depending on where this is run:

* file paths inside scripts
* dataset locations
* grid resolution / dataset size
* training hyperparameters

Paths are currently set directly inside the scripts rather than via config. 
TODO: change this. 

---

# Future improvements 

Improvements to still make:

* move configuration to a config file
* add CLI arguments to scripts
* add logging instead of print statements
* add small example dataset [??]
* add more learning objectives to FM training
* show two downstream tasks to demonstrate multiple use cases from one FM
* add notebook for exploring outputs - might be nice
* fix warnings from different scripts
* automate reading parameter dependencies between scripts

---