import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
from simple_ae import Unet
from tqdm import tqdm as std_tqdm
from functools import partial
from glob import glob


tqdm = partial(std_tqdm, dynamic_ncols=True)


accelerator = Accelerator()

train_images = glob(os.path.join("rna_data", "train", "*.npy"))
test_images = glob(os.path.join("rna_data", "test", "*.npy"))

train_images = np.asarray([np.load(x) for x in train_images])
test_images = np.asarray([np.load(x) for x in test_images])

epochs = 100
lr = 1e-3
in_chans = train_images[0].shape[0]

train_images = torch.from_numpy(train_images).float()
test_images = torch.from_numpy(test_images).float()

model = Unet(
    in_chans=in_chans,
    num_classes=in_chans,
    int_chans=16,
    latent_dim=16
)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

train_dataset = torch.utils.data.TensorDataset(train_images)
test_dataset = torch.utils.data.TensorDataset(test_images)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

for epoch in range(epochs):

    model.train()
    progress = tqdm(total=len(train_loader), desc="Training")
    for source in train_loader:

        optimizer.zero_grad()
        source = source[0]

        output = model(source)
        # loss = F.cross_entropy(output, targets)
        loss = F.mse_loss(output, source)

        accelerator.backward(loss)

        optimizer.step()

        progress.set_postfix({"Epoch": epoch, "train_loss": loss.item()})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
        progress.update()

    model.eval()
    progress = tqdm(total=len(train_loader), desc="Testing")
    with torch.no_grad():
        for source in test_loader:

            source = source[0]

            output = model(source)
            loss = F.mse_loss(output, source)

        progress.set_postfix({"test_loss": loss.item()})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
        progress.update()

