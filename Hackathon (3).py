
# Imports
from pathlib import Path
import nibabel as nib
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from IPython.display import HTML
import torch
import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Paths
root = Path("Task03_Liver_rs/imagesTr/")
label = Path("Task03_Liver_rs/labelsTr/")

# Helper Functions

def change_img_to_label_path(path):
    parts = list(path.parts)
    parts[parts.index("imagesTr")] = "labelsTr"
    return Path(*parts)

def get_mid_slice(img_path: Path):
    img = nib.load(img_path).get_fdata()
    z = img.shape[2]
    slice2d = img[:, :, z // 2]
    return slice2d

# Sample Paths
sample_path = list(root.glob("liver*"))[0]
sample_path_label = change_img_to_label_path(sample_path)

data = nib.load(sample_path)
label_data = nib.load(sample_path_label)
ct = data.get_fdata()
mask = label_data.get_fdata().astype(int)

# Plot Animation
fig = plt.figure()
camera = Camera(fig)
for i in range(ct.shape[2]):
    plt.imshow(ct[:, :, i], cmap="bone")
    mask_ = np.ma.masked_where(mask[:, :, i] == 0, mask[:, :, i])
    plt.imshow(mask_, alpha=0.5)
    camera.snap()
plt.tight_layout()
animation = camera.animate()
HTML(animation.to_html5_video())

# UNet Model
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
            torch.nn.ReLU()
        )
    def forward(self, X):
        return self.step(X)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DoubleConv(1, 32)
        self.layer2 = DoubleConv(32, 64)
        self.layer3 = DoubleConv(64, 128)
        self.layer4 = DoubleConv(128, 256)
        self.layer5 = DoubleConv(256 + 128, 128)
        self.layer6 = DoubleConv(128+64, 64)
        self.layer7 = DoubleConv(64+32, 32)
        self.layer8 = torch.nn.Conv3d(32, 3, 1)
        self.maxpool = torch.nn.MaxPool3d(2)
    def forward(self, x):
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        x4 = self.layer4(x3m)
        x5 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)
        x6 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)
        x7 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)
        ret = self.layer8(x7)
        return ret

# Test Model
model = UNet()
random_input = torch.randn(1, 1, 128, 128, 128)
with torch.no_grad():
    output = model(random_input)
assert output.shape == torch.Size([1, 3, 128, 128, 128])

# Training Setup
path = Path("Task03_Liver_rs/imagesTr/")
subjects_paths = list(path.glob("liver_*"))
subjects = []
for subject_path in subjects_paths:
    label_path = change_img_to_label_path(subject_path)
    subject = tio.Subject({"CT":tio.ScalarImage(subject_path), "Label":tio.LabelMap(label_path)})
    subjects.append(subject)
for subject in subjects:
    assert subject["CT"].orientation == ("R", "A", "S")
process = tio.Compose([
    tio.CropOrPad((256, 256, 200)),
    tio.RescaleIntensity((-1, 1))
])
augmentation = tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10))
val_transform = process
train_transform = tio.Compose([process, augmentation])
train_dataset = tio.SubjectsDataset(subjects[:105], transform=train_transform)
val_dataset = tio.SubjectsDataset(subjects[105:], transform=val_transform)
sampler = tio.data.LabelSampler(patch_size=96, label_name="Label", label_probabilities={0:0.2, 1:0.3, 2:0.5})
train_patches_queue = tio.Queue(
    train_dataset,
    max_length=40,
    samples_per_volume=5,
    sampler=sampler,
    num_workers=4,
)
val_patches_queue = tio.Queue(
    val_dataset,
    max_length=40,
    samples_per_volume=5,
    sampler=sampler,
    num_workers=4,
)
batch_size = 2
train_loader = torch.utils.data.DataLoader(train_patches_queue, batch_size=batch_size, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_patches_queue, batch_size=batch_size, num_workers=0)

class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self, data):
        pred = self.model(data)
        return pred
    def training_step(self, batch, batch_idx):
        img = batch["CT"]["data"]
        mask = batch["Label"]["data"][:,0]
        mask = mask.long()
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        self.log("Train Loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Train")
        return loss
    def validation_step(self, batch, batch_idx):
        img = batch["CT"]["data"]
        mask = batch["Label"]["data"][:,0]
        mask = mask.long()
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        self.log("Val Loss", loss)
        self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Val")
        return loss
    def log_images(self, img, pred, mask, name):
        pred = torch.argmax(pred, 1)
        axial_slice = 50
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][:,:,axial_slice]==0, mask[0][:,:,axial_slice])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth")
        axis[1].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][:,:,axial_slice]==0, pred[0][:,:,axial_slice])
        axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].set_title("Pred")
        self.logger.experiment.add_figure(f"{name} Prediction vs Label", fig, self.global_step)
    def configure_optimizers(self):
        return [self.optimizer]

model = Segmenter()
checkpoint_callback = ModelCheckpoint(
    monitor='Val Loss',
    save_top_k=10,
    mode='min')
gpus = 0 # Change to number of available GPUs
t
rainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1,
                     callbacks=checkpoint_callback,
                     max_epochs=100)
trainer.fit(model, train_loader, val_loader)

# Prediction and Visualization
model = Segmenter.load_from_checkpoint("weights/epoch=97-step=25773.ckpt")
model = model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
IDX = 4
mask = val_dataset[IDX]["Label"]["data"]
imgs = val_dataset[IDX]["CT"]["data"]
grid_sampler = tio.inference.GridSampler(val_dataset[IDX], 96, (8, 8, 8))
aggregator = tio.inference.GridAggregator(grid_sampler)
patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
with torch.no_grad():
    for patches_batch in patch_loader:
        input_tensor = patches_batch['CT']["data"].to(device)
        locations = patches_batch[tio.LOCATION]
        pred = model(input_tensor)
        aggregator.add_batch(pred, locations)
output_tensor = aggregator.get_output_tensor()
fig = plt.figure()
camera = Camera(fig)
pred = output_tensor.argmax(0)
for i in range(0, output_tensor.shape[3], 2):
    plt.imshow(imgs[0,:,:,i], cmap="bone")
    mask_ = np.ma.masked_where(pred[:,:,i]==0, pred[:,:,i])
    label_mask = np.ma.masked_where(mask[0,:,:,i]==0, mask[0,:,:,i])
    plt.imshow(mask_, alpha=0.1, cmap="autumn")
    plt.imshow(label_mask, alpha=0.5, cmap="jet")
    camera.snap()
animation = camera.animate()
HTML(animation.to_html5_video())

def show_slice(image_path):
    img = nib.load(image_path).get_fdata()
    mid = img.shape[2] // 2
    slice2d = img[:, :, mid]
    plt.imshow(slice2d.T, cmap="gray")
    plt.axis("off")
    plt.show()
    return slice2d

# Gradio App

def predict_volume(file):
    img = nib.load(file).get_fdata()
    img = (img - img.min()) / (img.max() - img.min())
    volume_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(volume_tensor)
        mask = torch.argmax(pred, dim=1).cpu().squeeze().numpy()
    return img, mask

def predict_mid_slice(file):
    img, mask = predict_volume(file)
    z = img.shape[2] // 2
    slice2d, mask2d = img[:, :, z], mask[:, :, z]
    fig, ax = plt.subplots()
    ax.imshow(slice2d, cmap="gray")
    ax.imshow(mask2d, alpha=0.4, cmap="jet")
    ax.axis("off")
    fig.canvas.draw()
    overlay_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay_img = overlay_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return overlay_img

def predict_slice_at_index(file, slice_index):
    img, mask = predict_volume(file)
    slice2d, mask2d = img[:, :, slice_index], mask[:, :, slice_index]
    fig, ax = plt.subplots()
    ax.imshow(slice2d, cmap="gray")
    ax.imshow(mask2d, alpha=0.4, cmap="jet")
    ax.axis("off")
    fig.canvas.draw()
    overlay_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay_img = overlay_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return overlay_img

with gr.Blocks() as demo:
    gr.Markdown("## 🩻 Liver Anomaly Segmentation")
    gr.Markdown("Upload a CT scan (.nii/.nii.gz) to view segmentation results.")
    with gr.Tab("Mid Slice Preview"):
        gr.Interface(
            fn=predict_mid_slice,
            inputs=gr.File(file_types=[".nii", ".nii.gz"]),
            outputs=gr.Image(type="numpy")
        )
    with gr.Tab("Full Volume Viewer"):
        file_input = gr.File(type="filepath", label="Upload CT scan (.nii/.nii.gz)")
        slice_slider = gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Slice Index")
        output_img = gr.Image(type="numpy")
        def wrapped_predict(file, slice_index):
            return predict_slice_at_index(file, slice_index)
        slice_slider.change(fn=wrapped_predict, inputs=[file_input, slice_slider], outputs=output_img)
        file_input.change(fn=wrapped_predict, inputs=[file_input, slice_slider], outputs=output_img)
demo.launch(debug=True)
