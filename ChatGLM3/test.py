# !/user/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(r"D:\1-PythonProjects\ChatGLM3\chatglm3-6b", trust_remote_code=True)

model = AutoModel.from_pretrained(r"D:\1-PythonProjects\ChatGLM3\chatglm3-6b", trust_remote_code=True).quantize(4).cuda()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)