from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(x)

def train_model(audio_frames):
    encoder_model = Encoder()
    decoder_model = Decoder()

    encoder_model.train()
    decoder_model.train()

    # Training the model
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import Adam

    # frames = audio_to_frames(audio, len(audio)).unsqueeze(dim=1)

    dataset = TensorDataset(audio_frames)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 10
    loss_fn = nn.L1Loss()  # L1 loss against the original audio sample works fine
    optimizer = Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=0.01)

    print("Beginning Training...")
    for epoch in range(epochs):
        print(f"---- Epoch {epoch} ----")
        encoder_model.train()
        decoder_model.train()

        train_loss = 0
        for batch_tuple in dataloader:
            batch = batch_tuple[0]  # shape (batch_size, 1, frame_size)
            latent = encoder_model(batch)
            reconstructed = decoder_model(latent)

            loss = loss_fn(reconstructed, batch)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(dataloader)

        print(f"Train Loss: {train_loss:.5f}")

    from torch import save
    save(encoder_model.state_dict(), "model_state/encoder_state")
    save(decoder_model.state_dict(), "model_state/decoder_state")


if __name__ == "__main__":
    import argparse
    from utils.utils import audio_to_frames

    parser = argparse.ArgumentParser(
        prog='code',
        description='Trains Encoder/Decoder model on a sample of audio')

    parser.add_argument('filename') #positional argument
    parser.add_argument('-t', '--train',
                        action='store_true')  # on/off flag

    args = parser.parse_args()
    print(args.filename, args.train)

    if args.train and args.filename:
        from librosa import load
        from torch import from_numpy, float32, cat
        import os

        audio_frames = None
        if os.path.isdir(args.filename):
            for file in os.listdir(args.filename):
                if file.endswith("mp3"):
                    audio, _ = load(os.path.join(args.filename, file), sr=None)
                    tensor = from_numpy(audio).type(float32)
                    frames = audio_to_frames(tensor, len(audio)).unsqueeze(dim=1)
                    audio_frames = frames if audio_frames is None else cat((audio_frames, frames))
        else:
            audio, _ = load(args.filename, sr=None)
            tensor = from_numpy(audio).type(float32)
            audio_frames = audio_to_frames(tensor, len(audio)).unsqueeze(dim=1)
        train_model(audio_frames)