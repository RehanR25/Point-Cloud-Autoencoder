import torch 
from torch import optim 
import os 
import time 
import input
from autoencoder import AutoEncoder
from dataset import _get_dataloader_
from loss_function import chamfer_distance_knn

start = time.time()

def main(): 

    # Creating an instance of the autoencoder 
    autoencoder = AutoEncoder(latent_dim=512, n_points=1024)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    autoencoder.to(device)
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoints_dir, 'checkpoint.pt')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_state = checkpoint['model_state_dict']
        model_state = autoencoder.state_dict()

        filtered_state = { 
           k : v for k,v in checkpoint_state.items()
           if k in model_state and v.shape == model_state[k].shape
        }

        model_state.update(filtered_state)
        autoencoder.load_state_dict(model_state)

        skipped = checkpoint_state.keys() - filtered_state.keys()

        if skipped: 
            best_val_loss = float('inf')
            print("Resumed from checkpoint with new final layer, resetting loss value")
        else :
            best_val_loss = checkpoint['val_loss']
            print(f'Resumed from checkpoint, best_val_loss = {best_val_loss:.4f}')
    else : 
        best_val_loss = float('inf')
        print(f'No checkpoint found')

    train_loader, valid_loader = _get_dataloader_()


    optimiser = optim.Adam(autoencoder.parameters(), lr = 0.0001, weight_decay=1e-5)

    # Learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=2
    )

    i = 0 
    valid_interval = 300 
    epochs = 200
    # too many epochs is a bad thing if the training data does not learn 
    # since the model constantly updates its weights, it overfits ultimately 
    patience_counter = 0
    patience_threshold = 15

    # Periodic validation checks
    # to track accuracy and can be used to decide when to save checkpoints
    # This allows the model to save the best model till now

    def evaluation(i): 
        nonlocal best_val_loss, patience_counter

        if i % valid_interval == 0 : 
            autoencoder.eval()
            valid_loss = 0
            with torch.no_grad(): 
                for pts in valid_loader:
                    pts = pts.to(device)

                    recon = autoencoder(pts)
                    #loss = chamfer_distance(recon, pts) + 0.1 * repulsion_loss(recon)

                    loss = chamfer_distance_knn(recon, pts)

                    valid_loss += loss.item()

            avg_val_loss = valid_loss/ len(valid_loader)
            print(f'Validation loss :{avg_val_loss}')

            scheduler.step(avg_val_loss)

            autoencoder.train()

            if avg_val_loss < best_val_loss : 
                best_val_loss = avg_val_loss
                save_path = os.path.join(checkpoints_dir, 'checkpoint.pt')
                checkpoint = {
                    'model_state_dict': autoencoder.state_dict(), 
                    'val_loss': avg_val_loss 
                }
                torch.save(checkpoint, save_path)
                print(f'New best model saved (val_loss = {avg_val_loss:.4f}) -> {save_path}')
                patience_counter = 0 
            else : 
                patience_counter += 1

            if patience_counter >= patience_threshold: 
                print("Exited early")
                return True 
        return False 

    for epoch in range(epochs): 
        autoencoder.train()

        for pts in train_loader: 
            pts = pts.to(device)

            recon = autoencoder(pts)

            # loss = chamfer_distance(recon, pts)

            loss = chamfer_distance_knn(recon, pts)

            # Clears old gradients 
            optimiser.zero_grad()

            # Computes new gradients 
            loss.backward()

            # Apply the update onto the optimiser 
            optimiser.step()

            i += 1 
            if i % 100 == 0 : 
                print(f'Batch {i}, Train loss: {loss.item():.4f}')
            stop = evaluation(i)
            if stop: 
                break 
        if stop: 
            break
        print(f' Epoch [{epoch} / {epochs}] done')

if __name__ == '__main__': 
    main()
    print(f'Execution time: {time.time() - start:.6f} seconds')