import torch
import numpy as np
import pandas as pd
from isic_metric import score
from tqdm import tqdm
from PIL import Image
class pAUC:
    def __init__(self, device, model, model_transform, X_test, y_test, metadata_test):
        self.device = device
        self.model = model
        self.model_transform = model_transform
        self.X_test = X_test
        self.y_test = y_test # ground truth labels
        self.metadata_test = metadata_test
        self.partial_aucs = []
        self.test_predictions = []
        self.test_isic_id = []
        
    def compute_pAUC(self):
        with torch.no_grad():
            i = 0
            for img in tqdm(self.X_test):
                if isinstance(img, np.ndarray):
                    # First ensure it's uint8
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                # Apply the transformation (PIL → normalized tensor)
                img_tensor = self.model_transform(img).unsqueeze(0).to(self.device)  # shape: [1, C, H, W]

                # Get prediction
                logits = self.model(img_tensor)
                prob = torch.sigmoid(logits).cpu().item()  # probability for class 1 (malignant)
                self.test_predictions.append(prob)
                self.test_isic_id.append(self.metadata_test[i]['isic_id'])
                i += 1
                
        df_gt = pd.DataFrame({
            "image_id": list(range(len(self.test_isic_id))),
            "target": self.y_test  # ground truth labels
        })
        
        df_pred = pd.DataFrame({
            "image_id": list(range(len(self.test_predictions))),
            "target": self.test_predictions  # prediction probabilities
        })
        
        pAUC = score(df_gt, df_pred, row_id_column_name="image_id", min_tpr=0.80)
        return pAUC
