"""DistilBERT multi-head classifier for merchant text."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class TransactionClassifier(nn.Module):

    def __init__(self, model_name, num_categories, num_subcategories, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_categories)
        )

        self.subcategory_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_subcategories)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.category_head(cls_embedding), self.subcategory_head(cls_embedding)


class TransactionClassifierInference:

    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(f"{model_path}/model.pt", map_location=self.device, weights_only=False)
        self.label_mappings = checkpoint['label_mappings']
        config = checkpoint['config']

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = TransactionClassifier(
            config['model_name'], config['num_categories'], config['num_subcategories']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def classify(self, text):
        encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                   max_length=64, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            cat_logits, subcat_logits = self.model(input_ids, attention_mask)

        cat_probs = F.softmax(cat_logits, dim=1)
        subcat_probs = F.softmax(subcat_logits, dim=1)
        cat_id = cat_logits.argmax(dim=1).item()
        subcat_id = subcat_logits.argmax(dim=1).item()

        return {
            'category': self.label_mappings['id_to_category'][str(cat_id)],
            'category_confidence': cat_probs[0, cat_id].item(),
            'subcategory': self.label_mappings['id_to_subcategory'][str(subcat_id)],
            'subcategory_confidence': subcat_probs[0, subcat_id].item(),
        }

