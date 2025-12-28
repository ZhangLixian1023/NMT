import torch
class Demo:
    """Show translateion demo."""
    
    def __init__(self, model, src_vocab, tgt_vocab, examples,device):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.examples = examples
        self.device = device
    
    def translate_sentence(self, sentence: str) -> str:
        """Translate a single sentence."""
        self.model.eval()

        # 预处理源语言句子
        src_tokens = sentence.split()
        src_indices = [self.src_vocab.word_to_idx(word) for word in src_tokens]
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(self.device)  # (seq_len, 1)
        src_lengths = torch.tensor([len(src_indices)], dtype=torch.long).to(self.device)
        
        # 翻译
        output_indices = self.model.predict(src_tensor, src_lengths)

        # 将索引转换为单词
        output_words = []
        for idx in output_indices:
            word = self.tgt_vocab.idx_to_word(idx)
            output_words.append(word)
            if word == '<eos>':
                break
        return ' '.join(output_words)

    def generate_translation_examples(self,noref=False) -> None:
        """Generate translations for a list of example sentences."""
        print(f'\n===翻译示例 ===')
        output=[]

        if noref:
            for src in self.examples:
                translation = self.translate_sentence(src)
                print(f'\n源语言: {src}')
                print(f'模型翻译: {translation}')
                output.append({'src':src,'translation':translation})
            print('====================')
            return output
        
        for (src,tgt) in self.examples:
            translation = self.translate_sentence(src)
            print(f'\n源语言: {src}')
            print(f'目标语言: {tgt}')
            print(f'模型翻译: {translation}')
            output.append({'src':src,'tgt':tgt,'translation':translation})
        print('====================')
        return output