# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.

## Problem Statement and Dataset
Develop a deep learning model for image classification using transfer learning. Utilize the pre-trained VGG19 model as the feature extractor, fine-tune it, and adapt it to classify images into specific categories.

## DESIGN STEPS
### **Step 1: Import Libraries and Load Dataset**
- Import the necessary libraries.
- Load the dataset.
- Split the dataset into training and testing sets.

### **Step 2: Initialize Model, Loss Function, and Optimizer**
- Define the model architecture.
- Use `CrossEntropyLoss` for multi-class classification.
- Choose the `Adam` optimizer for efficient training.

### **Step 3: Train the Model**
- Train the model using the training dataset.
- Optimize the model parameters to minimize the loss.

### **Step 4: Evaluate the Model**
- Test the model using the testing dataset.
- Measure performance using appropriate evaluation metrics.

### **Step 5: Make Predictions on New Data**
- Use the trained model to predict outcomes for new inputs.

## PROGRAM

### Load Pretrained Model and Modify for Transfer Learning
```
model = models.vgg19(pretrained=True)
```

### Modify the final fully connected layer to match the dataset classes
```
model.classifier[6] = nn.Linear(4096, num_classes)
```

### Include the Loss function and optimizer
```
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Train the model
```
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Convert labels to one-hot encoding
            labels = nn.functional.one_hot(labels, num_classes=num_classes).float().to(device) # Change this line
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Convert labels to one-hot encoding
                labels = nn.functional.one_hot(labels, num_classes=num_classes).float().to(device) # Change this line
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Yogeshvar M")
    print("Register Number:212222230180")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/394f2115-c091-41a0-8ce5-862cd2a6e42e)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/8066d963-f11f-494a-89b7-c7c980fd9fc3)

### Classification Report
![image](https://github.com/user-attachments/assets/0dedc10c-c79c-41bf-a959-de7931debe31)

### New Sample Prediction
![image](https://github.com/user-attachments/assets/974ddd53-580d-4206-8076-868d7f6a8623)


## RESULT
Thus, the transfer Learning for classification using VGG-19 architecture has succesfully implemented.
