import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, loaders, device, epochs=50, use_mask=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')

    train_history, val_history = [], []

    print("\n================ TRAINING STARTED ================\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("-" * 40)

        train_bar = tqdm(loaders['train'], desc="Training", leave=False)

        for batch in train_bar:
            if use_mask:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                inputs = torch.cat([img, msk], dim=1)
            else:
                inputs = batch["image"].to(device)

            s_lab = batch["status"].to(device).long()
            f_lab = batch["figo"].to(device).long()

            optimizer.zero_grad()

            s_out, f_out = model(inputs)

            loss_status = criterion(s_out, s_lab)
            loss_figo = criterion(f_out, f_lab)
            total_loss = loss_status + loss_figo

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            train_bar.set_postfix(loss=total_loss.item())

        avg_train_loss = running_loss / len(loaders['train'])
        train_history.append(avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in loaders['val']:
                if use_mask:
                    img = batch["image"].to(device)
                    msk = batch["mask"].to(device)
                    inputs = torch.cat([img, msk], dim=1)
                else:
                    inputs = batch["image"].to(device)

                s_lab = batch["status"].to(device).long()
                f_lab = batch["figo"].to(device).long()

                s_out, f_out = model(inputs)
                v_loss = criterion(s_out, s_lab) + criterion(f_out, f_lab)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(loaders['val'])
        val_history.append(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

            maps = loaders.get("maps", (None, None))  # FIX: avoid KeyError
            checkpoint = {
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': best_loss,
                'status_map': maps[0],
                'figo_map': maps[1]
            }
            torch.save(checkpoint, "best_model.pth")
            print(f"✅ Model Checkpoint Saved (Best Val Loss: {avg_val_loss:.4f})")

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f}")
        print("=" * 40)

    print("\n================ TRAINING COMPLETE ================\n")

    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label="Train Loss")
    plt.plot(val_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss (Multi-Task)")
    plt.tight_layout()
    plt.savefig("loss_metrics.png", dpi=200)
    plt.close()