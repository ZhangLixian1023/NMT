import torch
import numpy
class Demo:
    """Show translateion demo."""
    
    def __init__(self, model, src_vocab, tgt_vocab, examples,device,strategy='beam'):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.examples = examples
        self.device = device
        self.strategy=strategy
        print("Demo initialized with strategy:",self.strategy)
    
    def translate_sentence(self, sentence: str) -> str:
        """Translate a single sentence."""
        self.model.eval()
        # 预处理源语言句子
        src_tokens = sentence.split()
        src_indices = [self.src_vocab.word_to_idx(word) for word in src_tokens]
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device)  # ( 1 ,seq_len)
        # 翻译
        output_indices = self.model.predict(src_tensor,strategy=self.strategy)
        # 将索引转换为单词
        output_words = []
        for idx in output_indices:
            word = self.tgt_vocab.idx_to_word(idx)
            output_words.append(word)
            if word == '<eos>':
                break
        return ' '.join(output_words)

    def generate_translation_examples(self,noref=False,show=False) -> None:
        """Generate translations for a list of example sentences."""
        output=[]
        if show:
            print(f'\n===翻译示例 ===') 
        for (idx,item) in enumerate(self.examples):
            if noref:
                src= item
                translation = self.translate_sentence(src)
                if show:
                    print(f'\n{idx}. {src}')
                    print(f'模型: {translation}')
                output.append({'src':src,'translation':translation})
            else:
                (src,tgt)= item
                translation = self.translate_sentence(src)
                if show:
                    print(f'\n{idx}. {src}')
                    print(f'参考: {tgt}')
                    print(f'模型: {translation}')
                output.append({'src':src,'tgt':tgt,'translation':translation})
        if show:
            print('='*20+'\n')
        return output
    