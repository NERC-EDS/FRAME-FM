from big_geo_loader.datasets import BigGeoDataset
from big_geo_loader.utils import dump_selectors_to_yaml, load_selectors_from_yaml
import torch
from torch.utils.data import DataLoader 

# Need this because of the way PyTorch's DataLoader works with multiple workers and the fact 
# that we're using multiprocessing in our dataset caching. Setting the start method to 'spawn' 
# is often necessary to avoid issues with multiprocessing on certain platforms (like Windows)
#  or when using certain libraries that don't play well with the default 'fork' method on 
# Unix-based systems. By setting it to 'spawn', we ensure that each worker process starts fresh, 
# which can help avoid issues related to shared memory and state that can arise with 'fork'.
from torch import multiprocessing as mp


def main():

    selectors = [
            {   
                "uri": "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json",
                "common": {
                    "pre_transforms": [
                        {"type": "roll", "dim": "longitude", "shift": None},
                        {"type": "reverse_axis", "dim": "latitude"},
                        {"type": "subset", "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"], "latitude": [-30, 60], "longitude": [-40, 100]},
                        {"type": "rename", "var_id": "d2m", "new_name": "dewpoint_temperature"},
                    ],
                    "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
                    "chunks": {"time": 48}  # Example chunking strategy, can be adjusted as needed
                },
                "variables": {
                    "d2m": {},
                }
            },
            {   
                "uri": "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2t_repack.kr1.0.json",
                "common": {
                    "pre_transforms": [
                        {"type": "roll", "dim": "longitude", "shift": None},
                        {"type": "reverse_axis", "dim": "latitude"},
                        {"type": "subset", "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"], "latitude": [-30, 60], "longitude": [-40, 100]},
                        {"type": "rename", "var_id": "t2m", "new_name": "dewpoint_temperature"},
                    ],
                    "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
                    "chunks": {"time": 48}  # Example chunking strategy, can be adjusted as needed
                },
                "variables": {
                    "t2m": {},
                }
            },
            # {
            #     "uri": "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/chess-met/aggregations/chess-met_precip_gb_1km_daily_19610101-20191231.nca",
            #     "common": {
            #         "subset": {
            #             "time": ["2000-01-01 00:00:00", "2000-01-06 00:00:00"],
            #             "y": [912500, 1041500],
            #             "x": [535500, 655500],
            #         },
            #         "pre_transforms": [
            #             {"type": "rename", "var_id": "precip", "new_name": "precipitation"},
            #         ],
            #         "pre_transform_rule": "append",
            #         "chunks": {"time": 3, "y": 64, "x": 64}  # Example chunking strategy, can be adjusted as needed
            #     },
            #     "variables": {
            #         "precip": {},
            #      }
            # },
        ]
    
    print("\nWe have defined the selectors...let's write the to YAML, then reload them to show that works...")
    # Show how we can save and load selectors from YAML
    yaml_path = "selectors.yaml"
    dump_selectors_to_yaml(selectors, yaml_path)
    loaded_selectors = load_selectors_from_yaml(yaml_path)

    print("\nLoaded selectors:\n", loaded_selectors)

    # Example usage
    dataset = BigGeoDataset(
        selectors=loaded_selectors,
        cache_dir="DATASET_CACHE",
        pre_transforms=None,
        runtime_transforms=[{"type": "to_tensor"}],
        generate_stats=True,
        force_recache=False
    )

    print("\nCreated BigGeoDataset instance. Now let's check the dataset length before pre-caching data (should raise an error because the dataset is not yet cached)...")
    try:
        print(f"Dataset length: {len(dataset)}")
    except Exception as e:
        print(f"Expected error getting dataset length: {e}")

    print("\nPre-caching data...calling: `dataset.precache_data()` - this caches the tranformed subsets in Zarr format.")
    dataset.precache_data()

    print(f"\nDataset length is now accessible (because the dataset is cached): {len(dataset)}")

    sample = dataset[0]
    print(f"\nWe loaded the first sample in dataset: {sample.shape = }, {sample.dtype = }, {sample.min() = }, {sample.max() = }")  

    print("\nNow let's create a PyTorch DataLoader to iterate over the dataset in batches...")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8) 

    for counter, batch in enumerate(dataloader): 
        print(f"\nBatch counter: {counter + 1}: Batch shape: {batch.shape}, dtype: {batch.dtype}") 
        # print("Exiting loop after one iteration.")
        if counter > 2: 
            break  # Just show the first batch for demonstration

    # Now let's demonstrate loading the dataset into a PyTorch Model with a basic training loop.
    # This is just a placeholder example to show how the dataset can be used in a training loop.
    print("\nNow let's demonstrate a simple training loop using the dataset with a PyTorch model...")

    # The batch size, including the batch dimension is: [8, 2, 641, 604], so let's create a simple
    # PyTorch model that uses the 2 channels (dewpoint_temperature and surface_temperature).
    class SimpleModel(torch.nn.Module):
        input_shape = (2, 361, 561)

        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(2, 16, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc1 = torch.nn.Linear(32 * self.input_shape[1] * self.input_shape[2], 128)
            self.fc2 = torch.nn.Linear(128, 16)  # Example output size

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            # Add some dropout for regularization
            x = torch.nn.functional.dropout(x, p=0.5)

            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 5

    NUM_WORKERS = 1
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS)

    for epoch in range(n_epochs):  # Training for 5 epochs
        for batch_number, batch in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(batch)
            
            # Because all pretend, need to match shapes (because this isn't done properly)
            resized_outputs = outputs[:, :2].unsqueeze(-1).unsqueeze(-1).repeat(1,1,361, 561)
            loss = criterion(resized_outputs, batch)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Batch: {batch_number + 1}, Loss: {loss.item()}")


    print("\nFinished training loop demonstration.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
