import torch
from torchvision.utils import save_image

import config


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5 # remove normalization
        save_image(y_fake, folder + f"y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"label_{epoch}")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoints")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, learning_rate):
    print("=> Loading checkpoints")
    checkpoints = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoints['state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer'])

    # Mandatory step:
    for params in optimizer.param_groups:
        params['lr'] = learning_rate



