import sys
sys.path.append("../")
import pkbar
import time
import numpy as np
import torch

print('=> using pbar (progress bar)')
pbar = pkbar.Pbar(name='loading and processing dataset', target=10)

for i in range(10):
    time.sleep(0.1)
    pbar.update(i)


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 3
learning_rate = 0.001
train_per_epoch = 100

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
model = torch.nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print()

print('=> using kbar (keras bar)')
# Train the model
for epoch in range(num_epochs):
    ################################### Initialization ########################################
    kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=num_epochs, width=8, always_stateful=False)
    # By default, all metrics are averaged over time. If you don't want this behavior, you could either:
    # 1. Set always_stateful to True, or
    # 2. Set stateful_metrics=["loss", "rmse", "val_loss", "val_rmse"], Metrics in this list will be displayed as-is.
    # All others will be averaged by the progbar before display.
    ###########################################################################################

    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # training
    for i in range(train_per_epoch):
        # Forward pass
        outputs = model(inputs)
        train_loss = criterion(outputs, targets)
        train_rmse = torch.sqrt(train_loss)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        time.sleep(0.1)

        ############################# Update after each batch ##################################
        kbar.update(i, values=[("loss", train_loss), ("rmse", train_rmse)])
        ########################################################################################

    # validation
    outputs = model(inputs)
    val_loss = criterion(outputs, targets)
    val_rmse = torch.sqrt(val_loss)

    ################################ Add validation metrics ###################################
    kbar.add(1, values=[("val_loss", val_loss), ("val_rmse", val_rmse)])
    ###########################################################################################
