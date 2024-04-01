import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import VoiceDataset
from utils.model_utils import RVCModel
from utils.train_utils import train_epoch, validate, collate_fn

def main():
    # Load configuration
    with open("configs/config.json", "r") as f:
        config = json.load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and dataloader
    dataset = VoiceDataset(**config["data"])
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    
    # Create model, criterion, and optimizer
    model = RVCModel(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    
    # Training loop
    for epoch in range(config["train"]["num_epochs"]):
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        
        print(f"Epoch [{epoch+1}/{config['train']['num_epochs']}], Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % config["train"]["save_interval"] == 0:
            torch.save(model.state_dict(), f"models/rvc_model_epoch{epoch+1}.pth")
    
    # Save the final model
    torch.save(model.state_dict(), "models/rvc_model_final.pth")

if __name__ == "__main__":
    main()