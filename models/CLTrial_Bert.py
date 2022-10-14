from transformers.modeling_utils import PreTrainedModel
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
)
from models.row_transformer import RowEmbeddings
from models.tabformer_bert import TabFormerBertForMaskedLM, TabFormerBertConfig


class ClTrial_LM(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = RowEmbeddings(self.config)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    def forward(self, input_ids, **input_args):
        #Get features from Row Embeddings
        inputs_embeds = self.tab_embeddings(input_ids)
        #Pass the row embedding feature to the Bert model
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class ClTrial:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, field_hidden_size=768):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        hidden_size = field_hidden_size if flatten else (field_hidden_size * self.ncols)

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=self.ncols)
        
        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        
        self.model = self.get_model()

    def get_model(self):

        model = ClTrial_LM(self.config, self.vocab)

        return model