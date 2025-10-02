import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from wilds import get_dataset
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import StratifiedKFold

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]  # Get image index
        return img, label, path
    
def load_data(dataset_path, dataset_name, dataset_is_testset=False, batch_size=32, num_workers=8):

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_denoise = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.filter(ImageFilter.MedianFilter(size=11))),  # Median Filter
        transforms.GaussianBlur(kernel_size=21, sigma=1),  # Gaussian Blur
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == 'camelyon17-wilds':
        # Load Camelyon17-wilds dataset from given path
        dataset = get_dataset(dataset="camelyon17", root_dir=dataset_path, download=False) # dataset_path
        class_names = ['Tumor', 'Normal']
        num_classes = len(class_names)

        # Split dataset by the default of WILDS
        train_data = dataset.get_subset('train', transform=transform)
        val_data = dataset.get_subset('val', transform=transform)
        test_data = dataset.get_subset('test', transform=transform)

        # Take 10% of each
        # train_size = int(0.1 * len(train_data))
        # val_size = int(0.1 * len(val_data))
        # test_size = int(0.1 * len(test_data))
        # torch.manual_seed(0) # for reproducibility
        # train_data, _ = random_split(train_data, [train_size, len(train_data) - train_size])
        # torch.manual_seed(0)
        # val_data, _ = random_split(val_data, [val_size, len(val_data) - val_size])
        # torch.manual_seed(0)
        # test_data, _ = random_split(test_data, [test_size, len(test_data) - test_size])

        # Create data loaders
        generator = torch.Generator().manual_seed(0) # for reproducibility
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    if dataset_name == 'bbbc':
        if dataset_is_testset:
            dataset = ImageFolder(root=dataset_path, transform=transform)
            train_loader = []
            val_loader = []
            test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        else:
            train_data = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
            val_data = ImageFolder(root=os.path.join(dataset_path, 'val'), transform=transform)
            test_data = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform)
            class_names = train_data.classes
            num_classes = len(class_names)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
            test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    if dataset_name in ['manual_classification_6classes', 'zeiss_all_labeled', 'olympus']:
        dataset = ImageFolder(root=dataset_path, transform=transform_denoise)
        class_names = ['1-ER','2-MR','3-LR','4-Troph','5-Schizont','6-Gametocyte']
        num_classes = 6

        if dataset_is_testset:
            train_loader = []
            val_loader = []
            test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        else:
            # Use Stratified K-Fold to split data (5 folds) and only use the first fold
            labels = [label for _, label in dataset]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            train_idx, test_idx = next(skf.split(labels, labels))

            # Create data loaders
            train_data = torch.utils.data.Subset(dataset, train_idx)
            test_data = torch.utils.data.Subset(dataset, test_idx)            

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
            val_loader = test_loader
            
    return train_loader, val_loader, test_loader, num_classes, class_names
    
    
