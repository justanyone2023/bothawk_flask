
import pickle
import sklearn
print(sklearn.__version__)

model_path = '/root/bot_hawk_flask/training/model/baggingRandomForest.pickle'
# 加载预训练的模型
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
