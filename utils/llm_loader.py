from transformers import LlamaConfig, LlamaModel, AutoTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, OPTConfig, OPTModel
import transformers

def load_llm(configs):
    """
    Load the specified LLM model and tokenizer, and perform corresponding initialization configurations
    (such as setting the pad token and freezing parameters).
    """
    transformers.logging.set_verbosity_error()

    if configs.llm_model == 'LLAMA':
        llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-3.2-1B')
        llama_config.num_hidden_layers = configs.llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        llm_model = LlamaModel.from_pretrained(
            'meta-llama/Llama-3.2-1B',
            trust_remote_code=True,
            local_files_only=False,
            config=llama_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-3.2-1B',
            trust_remote_code=True,
            local_files_only=False
        )
    elif configs.llm_model == 'LLAMA-3B':
        print(configs.llm_model)
        llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-3.2-3B')
        llama_config.num_hidden_layers = configs.llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        llm_model = LlamaModel.from_pretrained(
            'meta-llama/Llama-3.2-3B',
            trust_remote_code=True,
            local_files_only=False,
            config=llama_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-3.2-3B',
            trust_remote_code=True,
            local_files_only=False
        )
    elif configs.llm_model == 'OPT':
        opt_config = OPTConfig.from_pretrained('facebook/opt-2.7b')
        llm_model = OPTModel.from_pretrained(
            'facebook/opt-2.7b',
            trust_remote_code=True,
            local_files_only=False,
            config=opt_config,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'facebook/opt-2.7b',
            trust_remote_code=True,
            local_files_only=False
        )
    elif configs.llm_model == 'GPT2':
        gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
        gpt2_config.num_hidden_layers = configs.llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        llm_model = GPT2Model.from_pretrained(
            'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=False,
            config=gpt2_config,
        )
        tokenizer = GPT2Tokenizer.from_pretrained(
            'openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=False
        )
    elif configs.llm_model == 'BERT':
        bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
        bert_config.num_hidden_layers = configs.llm_layers
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        llm_model = BertModel.from_pretrained(
            'google-bert/bert-base-uncased',
            trust_remote_code=True,
            local_files_only=False,
            config=bert_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'google-bert/bert-base-uncased',
            trust_remote_code=True,
            local_files_only=False,
            use_fast=True,
        )
    else:
        raise Exception('LLM model is not defined')

    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        pad_token = '[PAD]'
        tokenizer.add_special_tokens({'pad_token': pad_token})
        tokenizer.pad_token = pad_token

    for param in llm_model.parameters():
        param.requires_grad = False

    return llm_model, tokenizer
