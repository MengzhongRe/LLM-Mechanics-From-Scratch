# 手撕字符级(character)BPE（简单适用于教学面试）

import re
import unicodedata
from collections import defaultdict,Counter
from typing import List,Dict,Tuple,Iterable

class BPETokenizer:
    def __init__(self):
        # 核心数据结构
        self.vocab:Dict[int,str] = {} # ID -> subword
        self.inverse_vocab:Dict[str,int] = {}  # subword -> ID
        self.merges: Dict[Tuple[str,str],int] = {} # 合并规则(pair)

        # 特殊控制字符
        self.special_tokens = {
            '<|endoftext|>': 1000,
            '<think>': 1001,
            '</think>': 1002,
            '<UNK>': 1003,
        }
        
        self.special_tokens_inv = {v:k for k,v in self.special_tokens.items()}

        # 预分词正则:匹配缩写('s、't)、单词（可能带前导空格）、标点、连续空格
        # 这样处理可以保证decode时不需要手动补空格
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""")

        # 编译用于提取特殊字符的正则表达式
        # 由于特殊字符中包含类似于'<'、'|'等正则敏感符号，如果不加\转义符，会报错，我们直接re.escape把正则敏感符号
        # 全部转义
        escaped_special = [re.escape(k) for k in self.special_tokens.keys()]
        # 以下正则表达式必须加(),以便能够分组捕获，不丢失捕获到的特殊字符
        self.special_pattern = re.compile(f"({'|'.join(escaped_special)})")
        
    def _get_stats(self,vocab_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str,str], int]:
        """统计相邻Pair 的频率"""
        pairs = defaultdict(int)
        for word_tuple,freq in vocab_freqs.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i],word_tuple[i + 1])] += freq
        
        return pairs

    def _merge_tuple(self,word_tuple: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """在元组中合并指定的pair,比正则更高效且安全"""
        new_word = []
        i = 0
        first,second = pair
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        return tuple(new_word) # 需要把列表转换为元组，元组作为不可变对象适合作为字典的key

    def train(self,text: str, num_merges: int):
        """训练 BPE 模型"""
        # 1.先进行规范化和预分词
        text = unicodedata.normalize('NFC',text) # 把乱码、重音、奇怪的 Unicode -》 标准干净的字符
        words = self.pat.findall(text) # 用正则把一个完整的长文本切分成 缩写、单词（可能带前导空格）、标点、空格得到最终的words列表
        # 给后面的分词模型使用

        # 2.构建初始频率字典：将词切分为字符元组
        # ' hello' -> (' ','h','e','l','l','o')
        # 需要把字符串转换为元组，元组是不可变对象，才可以作为字典的key
        vocab_freqs = Counter(tuple(list(w)) for w in words)

        # 3.初始化词表(所有出现的单字符)
        # 先将所有ASCII可见字符加入到词表当中去
        for i in range(256):
            char = chr(i)
            self.vocab[i] = char
            self.inverse_vocab[char] = i

        # unique_chars = set()
        # for word_tuple in vocab_freqs.keys():
        #     for char in word_tuple:
        #         unique_chars.add(char)
        
        # # 按照字符的Unicode编码顺序赋值
        # # sorted函数按照所有单字符的unicode编码排序，例如ord('A') = 65
        # for i,char in enumerate(sorted(list(unique_chars))):
        #     self.vocab[i + 256] = char
        #     self.inverse_vocab[char] = i + 256
        
        current_id = len(self.vocab)
        print(f'[*] 初始字符数： {current_id}')

        # 4.核心训练循环
        for i in range(num_merges):
            # 先调用函数统计词频中每个词对出现的频率
            pairs = self._get_stats(vocab_freqs)
            if not pairs:
                break

            # 找到频率最高的那个词对
            best_pair = max(pairs,key=pairs.get)
            # 记录合并规则（也就是pair合并的优先级）
            self.merges[best_pair] = i 

            # 将当前频率最高词对注册到词表当中
            new_symbol = best_pair[0] + best_pair[1]
            # 更新ID -> 字符词表
            self.vocab[current_id] = best_pair[0] + best_pair[1]
            # 更新字符 -> ID词表
            self.inverse_vocab[new_symbol] = current_id
            current_id += 1

            # 在词频表中合并最高频词对
            new_vocab_freqs = {}
            for word_tuple,freq in vocab_freqs.items():
                new_tuple = self._merge_tuple(word_tuple,best_pair)
                new_vocab_freqs[new_tuple] = freq
            vocab_freqs = new_vocab_freqs

            if (i + 1) % 100 == 0:
                print(f'[*] 已完成{i}轮合并！')
        print(f'[*] 已完成{num_merges}轮合并,最终词表大小为: {len(self.vocab)}')
    
    def _encode_chunk(self,text: str) -> List[int]:
        """对非特殊字符文本块进行编码级 字符 -> ID"""
        # 1.预分词:先提取文本串中的缩写、单词、标点符号和空格
        words = self.pat.findall(text)
        # 初始化索引序列
        ids = []
        # 开始编码,外层循环遍历每个单词
        for word in words:
            # 内层循环1.对每个单词进行合并操作
            # 先把每个字符串（单词为主）转换为字符列表方便后续统计字符对以及合并操作
            symbols = list(word)
            # 循环应用合并规则，直到字符列表中只有一个字符串，此时不能再合并
            while len(symbols) > 1:
                # 先寻找当前单词中所有存在的pair,并找出rank最小的那个
                pairs = [(symbols[i],symbols[i + 1])for i in range(len(symbols) - 1)]
                # 获取在merges中rank最小的那个pair
                best_pair = min(pairs,key=lambda p: self.merges.get(p,float('inf')))

                # 如果得到的pair不在合并规则中，则说明已经没有可以合并的了,直接break掉当前单词的合并，即内层循环
                if best_pair not in self.merges:
                    break
                # 执行合并操作，合并后需要转换为列表，方便后续操作
                symbols = list(self._merge_tuple(tuple(symbols),best_pair))

            # 内层循环2.在某个单词合并之后，对其进行编码并加入到结果列表当中
            # 转为ID 
            for sym in symbols:
                if sym in self.inverse_vocab:
                    ids.append(self.inverse_vocab[sym])
                else:
                    # 这里的处理可能更复杂，比如转为byte或跳过
                    print(f'[*] 警告： 字符"{sym}" 不在词表当中，已被替换为<UNK>! ')
                    ids.append(self.special_tokens['<UNK>'])
        return ids


    def encode(self,text: str) -> List[int]:
        """文本 -》ID序列"""
        # 先用正则表达式匹配文本的特殊字符，将其与其他字符切分开，同时保留特殊字符
        chunks = self.special_pattern.split(text) # str -> List[str]
        final_ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                final_ids.append(self.special_tokens[chunk])
            elif chunk:
                # 由于self._encode_chunk返回的是ID列表，所以必须用extend函数将ID
                # 平铺在结果列表当中
                final_ids.extend(self._encode_chunk(chunk))
        return final_ids
    
    def decode(self,ids: List[int]) -> str:
        """ID -> 字符串文本"""
        # 初始化字符列表，用于暂时存储从ID转换来的字符
        parts = []
        for idx in ids:
            if idx in self.special_tokens_inv:
                parts.append(self.special_tokens_inv[idx])
            elif idx in self.vocab:
                parts.append(self.vocab[idx])
        return ''.join(parts)

if __name__ == '__main__':
    # 1.准备语料库
    train_text = 'low ' * 5 + 'lower ' * 2 + 'newest ' * 6 + 'widest ' * 3

    # 2.初始化分词器
    tokenizer = BPETokenizer()
    # 3.训练十次合并
    tokenizer.train(train_text,num_merges=10)
    print(f'[*] 以下为词表内容：{tokenizer.vocab}')

    # 4.测试文本
    # 包含训练过的词，没见过的词以及特殊字符、标点符号
    test_text = "<think> I'm lower than the newestest! </think> \n 你好!我叫孟志泉"

    print(f'\n[原始文本]： {test_text}')
    
    # 5.编码：字符 -> ID
    ids = tokenizer.encode(test_text)
    print(f'[编码 ID]: {ids}')

    # 6.解码：ID -> 字符
    decoded_text = tokenizer.decode(ids)
    print(f'\n[解码字符]: {decoded_text}')
    # 7.判断是否实现无损压缩
    assert test_text == decoded_text,'测试失败，编解码不一致！'
    print(f'\n[*] 验证通过：编解码完全一致！！！')

    # 8.打印词表内容
    
