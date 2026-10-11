## Point Cloud Autoencoder 
A PyTorch autoencoder that compresses a 3D point cloud (1024 points) into a 512-dim latent vector and reconstructs it. Trained on ModelNet40

### Architecture 
   * Encoder(```encoder.py```): PointNet-style. Shared per-point ```Conv1d``` layers (3 -> 64 -> 128 -> 258 -> latent) with BatchNorm, then max pooling over points for an order-invariant latent vector. 
   * Decoder(```decoder.py```): FoldingNet-style. A fixed 32x32 2D grid is concatenated with the latent vector and "folded" into 3D twice (fold 1 gives a first guess, fold 2 refines it), producing 1024 points
   * AutoEncoder (```autoencoder.py```): Wraps the encoder and decoder

### Loss 

```loss_function.py```  contains : 
   * ```chamfer_distance```: standard nearest-neighbour Chamfer distance 
   * ```chamfer_distance_knnn```: Chamfer averaged over the k nearest neighbours (default k=8). **Used for training**. It discourages the model from collapsing onto a thin wireframe.
   * ```repulsion_loss```: optional penalty on clustered predicted points (not used by default). 

### Project Structure 

| File | Purpose |
| --- | --- | 
|```input.py```| Downloads ModelNet40 and samples 1024 surface points per mesh into ```.npy``` files (centred and scaled to the unit sphere)
|```dataset.py```| ```PointCloudDataset``` and train/validation dataloaders with a 90/10 split, seed = 42, batch size = 32 |
|```encoder.py```,```decoder.py```, ```autoencoder.py```| Model | 
|```loss_function.py```| Loss functions |
|```training.py```| Training loop | 
|```visualise_reconstructions.py```|  Plots original vs reconstructed clouds | 

### Setup 

```sh 
pip install torch numpy trimesh matplotlib 
``` 

### Usage 

Run all commands from the project root, since all data paths are relative to it (```data/```, ```checkpoints/```).
1. **Preprocess** (downloads ModelNet40 and creates the point clouds, skipped if the output folder exists): 
    ```bash
      python input.py
    ```
2. **Train**:
    ```bash
      python training.py
   ```
   * Adam, lr = 1e-5, weight decay = 1e-6, up to 200 epochs
    * ```ReduceLROnPlateau``` scheduler (factor 0.5, patience 2). 
    * Validates every 300 batches, saves the best model to ```checkpoint/checkpoints.pt```, and stops early after 15 validations without improvement. 
    * Resumes from the checkpoint if one exists (layers with mismatched are skipped)
3. **Visualise**:  
    ```bash
    python visualise_reconstructions.py
    ```
    Saves a comparison plot to the path set in ```SAVE_PATH```. Top row is the original, bottom row is the reconstruction. 


### Configuration 
   * Edit ```CHECKPOINT_PATH```, ```DATA_DIR```, ```N_SAMPLES``` and ```SAVE_PATH``` at the top of ```visualise_reconstructions.py```
   * To train or test on specific categories, use the commented-out category filter in ```dataset.py``` and ```visualise_reconstructions.py```.
   * Change ```latent_dim``` to trade compression against detail. Set the same value in ```training.py``` and ```LATENT_DIM``` in ```visualise_reconstructions.py```, and retrain, since a checkpoint only loads with the latent size it was trained with. 
