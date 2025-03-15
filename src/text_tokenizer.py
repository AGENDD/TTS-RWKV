class TextTokenizer:
    def __init__(self):
        # 创建字母表、空格、结束标记、常见标点符号和数字的词表
        self.vocab = {chr(i): i - 97 for i in range(97, 123)}  # 'a' 到 'z' 对应 0 到 25
        self.vocab.update({str(i): 26 + i for i in range(10)})  # '0' 到 '9' 对应 26 到 35
        self.vocab['<unk>'] = 36  # 未知字符对应 36
        self.vocab['<bot>'] = 37  # 开始标记对应 37
        self.vocab['<eot>'] = 38  # 结束标记对应 38
        self.vocab[' '] = 39  # 空格对应 39
        self.vocab['.'] = 40  # 句号对应 40
        self.vocab[','] = 41  # 逗号对应 41
        self.vocab['!'] = 42  # 感叹号对应 42
        self.vocab['?'] = 43  # 问号对应 43
        self.vocab['\''] = 44  # 单引号对应 44
        self.vocab['"'] = 45  # 双引号对应 45
        self.vocab['-'] = 46  # 连字符对应 46
        
        # 添加额外的字符
        additional_chars = ['”', 'â', ')', ']', '“', 'è', '£', ';', 'à', '$', '[', '’', 'ü', 'ê', ':', '(', 'é', '&']
        for i, char in enumerate(additional_chars):
            self.vocab[char] = 47 + i

        # 将词表大小扩展到 128
        self.vocab_size = 128
        self.vocab.update({f'<unk{47 + len(additional_chars) + i}>': 47 + len(additional_chars) + i for i in range(self.vocab_size - len(self.vocab))})

    def tokenize(self, text):
        # 将输入文本转换为小写
        text = text.lower()
        # 添加 <bot> token
        tokens = [self.vocab['<bot>']]
        # 将每个字符转换为对应的 token ID，如果字符不在词表中则使用 <unk>
        tokens.extend(self.vocab.get(char, self.vocab['<unk>']) for char in text)
        # 添加结束标记
        tokens.append(self.vocab['<eot>'])
        return tokens

    def detokenize(self, tokens):
        # 创建从 token ID 到字符的映射
        id_to_char = {v: k for k, v in self.vocab.items()}
        # 将 token ID 转换回字符
        text = ''.join(id_to_char.get(token, '<unk>') for token in tokens if token not in {self.vocab['<bot>'], self.vocab['<eot>']})
        return text