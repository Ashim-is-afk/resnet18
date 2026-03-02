import torch
from data import get_loaders
from model import EndometrialResNet
from train import train_model
from eval import evaluate

def main():
    TSV_PATH = "endometrial_data.tsv"
    WEIGHTS_PATH = "resnet_18_23dataset.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    USE_MASK = True
    PERFORM_CROP = True
    
    EPOCHS = 50
    BATCH_SIZE = 2

    print(f"Using device: {DEVICE}")
    print(f"Settings: USE_MASK={USE_MASK}, PERFORM_CROP={PERFORM_CROP}")

    data_bundle = get_loaders(
        TSV_PATH, 
        batch_size=BATCH_SIZE, 
        use_mask=USE_MASK, 
        perform_crop=PERFORM_CROP
    )
    
    loaders = {
        "train": data_bundle["train"],
        "val":   data_bundle["val"],
        "test":  data_bundle["test"],
        "maps":  data_bundle["maps"]   # FIX: include maps so train.py can save checkpoint
    }
    status_map, figo_map = data_bundle["maps"]

    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches:   {len(loaders['val'])}")
    print(f"Test batches:  {len(loaders['test'])}")

    model = EndometrialResNet(
        num_status=len(status_map), 
        num_figo=len(figo_map), 
        weights_path=WEIGHTS_PATH,
        in_channels=2 if USE_MASK else 1
    ).to(DEVICE)

    print("\nStarting training phase...")
    train_model(model, loaders, DEVICE, epochs=EPOCHS, use_mask=USE_MASK)

    print("\nTraining complete. Starting final evaluation on test set...")
    evaluate(
        model, 
        loaders["test"], 
        DEVICE, 
        status_map, 
        figo_map, 
        ckpt_path="best_model.pth",
        use_mask=USE_MASK
    )

if __name__ == "__main__":
    main()