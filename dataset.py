#"custom Telugu handwritten digits dataset"
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

transform = transforms.Compose([
                                transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))
                              ])

def collate_fn(batch):
    images, labels = zip(*batch)
    return {
        'pixel_values': torch.stack(images),
        'labels': torch.tensor(labels)
    }
  
class num_dataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=transform):
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform
    
  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
    image  = Image.open(img_path).convert('L')
    label = torch.tensor(int(self.annotations.iloc[index,1]))

    if self.transform:
            image = self.transform(image)
      
    return (image, label)


def create_dataset(csv_file = '/drive/MyDrive/labels.csv',img_dir = '/drive/MyDrive/dataset/'):
  
  dataset = num_dataset(csv_file = csv_file, root_dir = img_dir)
  train_size = int(0.7 * len(dataset))   # 70% for training
  test_size = int(0.20 * len(dataset))     # 20% for test
  val_size = len(dataset) - train_size - test_size  # Remaining 10% for val


  train_dataset,test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

  # Create DataLoaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers = 4)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
  
  return train_loader,test_loader,val_loader
