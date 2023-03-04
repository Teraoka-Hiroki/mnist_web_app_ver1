# 以下を「model.py」に書き込み
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes = ["「0」", "「1」", "「2」", "「3」", "「4」", "「5」", "「6」", "「7」", "「8」", "「9」"]
#classes_en = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
n_class = len(classes)
img_size = 28

# CNNのモデル
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3)  # 畳み込み層:(入力チャンネル数, フィルタ数、フィルタサイズ)
 #       self.pool = nn.MaxPool2d(2, 2)  # プーリング層:（領域のサイズ, ストライド）
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)  # プーリング層:（領域のサイズ, ストライド）
        self.dropout1 = nn.Dropout(p=0.5)  # ドロップアウト:(p=ドロップアウト率)
        self.fc1 = nn.Linear(12*12*64, 128)  # 全結合層
        self.dropout2 = nn.Dropout(p=0.5)  # ドロップアウト:(p=ドロップアウト率)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12*12*64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def predict(img):
    # モデルへの入力
    img = img.convert("L")
    img = img.resize((img_size, img_size))
#    img = 255 - img
    normalize = transforms.Normalize((0.0), (1.0))  # 平均値を0、標準偏差を1に
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose([to_tensor, normalize])


    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.0, 0.0), (1.0, 1.0))  # 平均値を0、標準偏差を1に
    #                            ])

    img = transform(img)
    x = img.reshape(1, 1, img_size, img_size)

    # 訓練済みモデル
    net = Net()
    net.load_state_dict(torch.load(
        "model_cnn_mnist3.pth", map_location=torch.device("cpu")
        ))
    
    # 予測
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [( classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
