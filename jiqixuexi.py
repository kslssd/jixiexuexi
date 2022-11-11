import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset

# ......读取数据集......
#使用pandas读取训练集
df = pd.read_csv('train_dataset.csv')
#去除训练集中重复的元素，将训练集转换为列表
user_ids = df["user_id"].unique().tolist()

# 对user和book重新编码
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

book_ids = df["item_id"].unique().tolist()
book2book_encoded = {x: i for i, x in enumerate(book_ids)}
book_encoded2book = {i: x for i, x in enumerate(book_ids)}

# 编码映射
df["user"] = df["user_id"].map(user2user_encoded)
df["book"] = df["item_id"].map(book2book_encoded)

num_users = len(user2user_encoded)
num_books = len(book_encoded2book)

user_book_dict = df.iloc[:].groupby(['user'])['book'].apply(list)

#构建负样本
neg_df = []
book_set = set(list(book_encoded2book.keys()))
for user_idx in user_book_dict.index:
    book_idx = book_set - set(list(user_book_dict.loc[user_idx]))
    book_idx = list(book_idx)
    neg_book_idx = np.random.choice(book_idx, 100)
    for x in neg_book_idx:
        neg_df.append([user_idx, x])

#负样本的标签为0
neg_df = pd.DataFrame(neg_df, columns=['user', 'book'])
neg_df['label'] = 0

# 正样本的标签为1
df['label'] = 1

# 正负样本合并为数据集
train_df = pd.concat([df[['user', 'book', 'label']],
                      neg_df[['user', 'book', 'label']]], axis=0)

train_df = train_df.sample(frac=1)
del df;

# ......自定义数据集......
#数据集需要继承paddle.io.Dataset类
class SelfDefinedDataset(Dataset):
    # 实现构造函数
    def __init__(self, data_x, data_y, mode = 'train'):
        super(SelfDefinedDataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'predict':
            return self.data_x[idx]
        else:
            return self.data_x[idx], self.data_y[idx]

    # 获取元素的长度
    def __len__(self):
        return len(self.data_x)

#划分数据集，得到训练集和验证集
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_df[['user', 'book']].values,
                                        train_df['label'].values.astype(np.float32).reshape(-1, 1))
traindataset = SelfDefinedDataset(x_train, y_train)
# 对自定义数据集进行迭代读取
for data, label in traindataset:
    print(data.shape, label.shape)
    print(data, label)
    break
# paddle.io.DataLoader迭代读取数据集
train_loader = paddle.io.DataLoader(traindataset, batch_size = 1280*4, shuffle = True)
for batch_id, data in enumerate(train_loader):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break

val_dataset = SelfDefinedDataset(x_val, y_val)
val_loader = paddle.io.DataLoader(val_dataset, batch_size = 1280*4, shuffle = True)
for batch_id, data in enumerate(val_loader):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break

EMBEDDING_SIZE = 32

# ......定义模型......
class RecommenderNet(nn.Layer):
    #构造函数
    def __init__(self, num_users, num_books, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        weight_attr_user = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size,
            weight_attr=weight_attr_user
        )
        self.user_bias = nn.Embedding(num_users, 1)

        weight_attr_book = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.book_embedding = nn.Embedding(
            num_books,
            embedding_size,
            weight_attr=weight_attr_book
        )
        self.book_bias = nn.Embedding(num_books, 1)

    #向前计算函数
    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_user_book = paddle.dot(user_vector, book_vector)
        x = dot_user_book + user_bias + book_bias
        x = nn.functional.sigmoid(x)
        return x
model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE)
#模型封装
model = paddle.Model(model)

optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.003)#优化器
loss = nn.BCELoss() #损失函数
metric = paddle.metric.Precision()#评价指标

# 设置visualdl路径
log_dir = './visualdl'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)

# 模型训练与验证
model.prepare(optimizer, loss, metric)
model.fit(train_loader, val_loader, epochs=5, save_dir='./checkpoints', verbose=1, callbacks=callback)

# ......预测测试集......
test_df = []
with open('sub.csv', 'w') as up:
    up.write('user_id,item_id\n')

# 模型预测步骤
book_set = set(list(book_encoded2book.keys()))
for idx in range(int(len(user_book_dict) / 1000) + 1):
    # 对于所有的用户，需要预测其与其他图书的打分
    test_user_idx = []
    test_book_idx = []
    for user_idx in user_book_dict.index[idx * 1000:(idx + 1) * 1000]:
        book_idx = book_set - set(list(user_book_dict.loc[user_idx]))
        book_idx = list(book_idx)
        test_user_idx += [user_idx] * len(book_idx)
        test_book_idx += book_idx

    # 从剩余图书中筛选出标签为正的样本
    test_data = np.array([test_user_idx, test_book_idx]).T
    test_dataset = SelfDefinedDataset(test_data, data_y=None, mode='predict')
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=1280, shuffle=False)

    test_predict = model.predict(test_loader, batch_size=1024)
    test_predict = np.concatenate(test_predict[0], 0)

    test_data = pd.DataFrame(test_data, columns=['user', 'book'])
    test_data['label'] = test_predict
    for gp in test_data.groupby(['user']):
        with open('sub.csv', 'a') as up:
            u = gp[0]
            b = gp[1]['book'].iloc[gp[1]['label'].argmax()]
            up.write(f'{userencoded2user[u]}, {book_encoded2book[b]}\n')

    del test_data, test_dataset, test_loader
