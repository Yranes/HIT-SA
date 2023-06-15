2023春情感分析实验 情感分类

实验环境：colab

词向量使用：[Chinese Word Vectors 中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)

数据处理：分词使用[jieba](https://github.com/fxsjy/jieba)，变长数据处理全部粗暴加\<PAD\>符，不在预训练词向量中出现的均设置为\<UNK\>

CNN模型参考：<https://www.cnblogs.com/ttdeveloping/p/10668621.html>

LSTM模型直接用最后一个输出接FC层做判断（没有使用注意力机制）

尝试使用了老师下发的./数据/参考模型/LSTM.ppt中的方法自定义collate_fn函数对数据进行补全，以求不引入噪音\<PAD\>，但跑的又慢最后效果也不好……（模型实现在SA-LSTM.ipynb的代码最后）

test集上结果：
|模型|F1(micro)|Acc|P|R|
|:--:|:--:|:--:|:--:|:--:|
|CNN|0.715|0.780|0.741|0.707|
|BiLSTM|0.742|0.790|0.746|0.741|
