import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm

# Load the dataset
file_path = 'evaluation.csv'  # Change to the correct path of your dataset
df = pd.read_csv(file_path, delimiter=';')

# Display a portion of the dataset to ensure it is loaded correctly
print(df.head())

# Prepare the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom dataset class
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Tokenize the texts
        self.input_ids = []
        self.attention_masks = []
        
        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.input_ids.append(encoding['input_ids'].squeeze(0))
            self.attention_masks.append(encoding['attention_mask'].squeeze(0))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].clone().detach().to(torch.long),
            'attention_mask': self.attention_masks[idx].clone().detach().to(torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Parameters
MAX_LEN = 512
BATCH_SIZE = 16

# Create the dataset and the dataloader
train_texts = df['text'].values  # Assuming the column with the texts is named 'text'
train_labels = df['label'].values  # Assuming the column with the labels is named 'label'

train_dataset = FakeNewsDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer, max_len=MAX_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the pre-trained BERT model for binary classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the GPU (if available)
model.to(device)

# Set the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training function
def train_model(model, train_dataloader, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc="Training"):
        # Move the data to the device (CPU or GPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Pass the data through the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # The loss is the first element returned by the model
        
        # Perform backpropagation
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        # Add the loss to the total
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)  # Calculate the average loss
    print(f"Average Training Loss: {avg_loss}")

# Model training
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_model(model, train_dataloader, optimizer, device)

# Save the model
model.save_pretrained('./fake_news_model')
