## Repo for addressing knowledge conflicts in PLM(s)

### Requirements

HF and pytorch. Other versions would probably work as well - but just in case you run into issues.

```
torch==1.7.1
transformers=4.18.0
```

## New Dataloader
    只使用原始jsonl data，预处理均在tune_gpt_qa.py内完成，使用验证集选出的4000条eval，最终完全正确593条，输出的correct_answered.jsonl包含了数据的几乎全部属性
    

## ShuffleGen.ipynb
    使用correct_answered.jsonl生成错误答案，其中把数据load成pandas dataframe，然后按照Pred类型分组计数，组内少于5个的过滤，其余的在组内随意挑选不同的答案，生成wrong answer，数据比较有用的几个attr：
    query：包含MASK的回答
    answer: 多个答案
    question：问题主体
    oneanswer: answer[0]
    group_ans: 相同pred类型的oneanswer集合
    target：query.replace('[MASK]',answer[0])
    text: [BOS]+question+[SEP]+target+<|endoftext|>
    wrongans: randomly selected from group_ans different from oneanswer