import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, BertConfig
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertModel



class MultiTaskBertConfig(PretrainedConfig):

    model_type = "multi_task_bert_encoder_decoder"

    def __init__(self, vocab_size=32000, hidden_size=768, num_hidden_layers=6, num_attention_heads=12,
                 intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 max_position_embeddings=768, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12,
                 pad_token_id=0, num_pos_labels=17, mlm_weight=1.0, translation_weight=1.0, pos_weight=0.5, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_pos_labels = num_pos_labels
        self.mlm_weight = mlm_weight
        self.translation_weight = translation_weight
        self.pos_weight = pos_weight



class MultiTaskBertEncoderDecoder(PreTrainedModel):
    config_class = MultiTaskBertConfig


    def __init__(self, config: MultiTaskBertConfig):
        super().__init__(config)


        encoder_config = BertConfig.from_dict(config.to_dict())
        self.encoder = BertModel(config=encoder_config)

        decoder_config = BertConfig.from_dict(config.to_dict())
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        self.decoder = BertModel(config=decoder_config)

        self.decoder.embeddings = self.encoder.embeddings


        self.lm_head = BertLMPredictionHead(config)


        self.lm_head.decoder.weight = self.encoder.embeddings.word_embeddings.weight



        self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pos_classifier = nn.Linear(config.hidden_size, config.num_pos_labels)


        from transformers.models.bert.modeling_bert import BertEmbeddings
        self.embeddings_orig = BertEmbeddings(encoder_config)
        self.embeddings_norm = BertEmbeddings(encoder_config)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))


    def get_encoder(self):

        return self.encoder

    def get_decoder(self):

        return self.decoder


    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            norm_input_ids: torch.LongTensor = None,
            norm_attention_mask: torch.FloatTensor = None,
            norm_mlm_labels: torch.LongTensor = None,
            norm_tlm_labels: torch.LongTensor = None,
            fusion_mode: str = "none",
            decoder_input_ids: torch.LongTensor = None,
            decoder_attention_mask: torch.FloatTensor = None,

            mlm_labels: torch.LongTensor = None,
            tlm_labels: torch.LongTensor = None,

            translation_labels: torch.LongTensor = None,
            pos_labels: torch.LongTensor = None,
            return_dict: bool = True,
            **kwargs,
    ):


        if fusion_mode == "alpha" and norm_input_ids is not None:

            emb_orig = self.embeddings_orig(input_ids=input_ids)
            emb_norm = self.embeddings_norm(input_ids=norm_input_ids)


            mask_o = attention_mask.unsqueeze(-1).type_as(emb_orig)
            emb_orig = emb_orig * mask_o
            if norm_attention_mask is not None:
                mask_n = norm_attention_mask.unsqueeze(-1).type_as(emb_norm)
                emb_norm = emb_norm * mask_n


            alpha = torch.sigmoid(self.alpha_logit)
            fused_embeddings = alpha * emb_orig + (1.0 - alpha) * emb_norm


            fused_attn = attention_mask
            if norm_attention_mask is not None:
                fused_attn = torch.clamp(attention_mask + norm_attention_mask, max=1).to(attention_mask.dtype)


            encoder_outputs = self.encoder(inputs_embeds=fused_embeddings,
                                           attention_mask=fused_attn,
                                           return_dict=True)
            encoder_hidden = encoder_outputs.last_hidden_state


            logits_orig = self.lm_head(encoder_hidden)
            logits_norm = None


            translation_logits = None
            if decoder_input_ids is not None:
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=fused_attn,
                    return_dict=True
                )
                decoder_last_hidden_state = decoder_outputs.last_hidden_state
                translation_logits = self.lm_head(decoder_last_hidden_state)

            pos_logits = self.pos_classifier(self.pos_dropout(encoder_hidden))


            loss_dict = {}
            loss_fct = nn.CrossEntropyLoss()
            if mlm_labels is not None:
                mlm_loss = loss_fct(logits_orig.view(-1, self.config.vocab_size), mlm_labels.view(-1))
                loss_dict["mlm_loss"] = mlm_loss
            if tlm_labels is not None:
                tlm_loss = loss_fct(logits_orig.view(-1, self.config.vocab_size), tlm_labels.view(-1))
                loss_dict["tlm_loss"] = tlm_loss
            if translation_labels is not None and translation_logits is not None:
                trans_loss = loss_fct(translation_logits.view(-1, self.config.vocab_size), translation_labels.view(-1))
                loss_dict["translation_loss"] = trans_loss
            if pos_labels is not None:
                pos_loss = loss_fct(pos_logits.view(-1, self.config.num_pos_labels), pos_labels.view(-1))
                loss_dict["pos_loss"] = pos_loss

            if not return_dict:
                return (None, logits_orig, pos_logits)

            return {
                "loss_dict": loss_dict,
                "logits_orig": logits_orig,
                "logits_norm": logits_norm,
                "pos_logits": pos_logits,
                "translation_logits": translation_logits,
                "alpha": alpha.detach()
            }

        encoder_outputs_orig = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        encoder_hidden_orig = encoder_outputs_orig.last_hidden_state
        logits_orig = self.lm_head(encoder_hidden_orig)

        logits_norm = None
        if norm_input_ids is not None:
            encoder_outputs_norm = self.encoder(input_ids=norm_input_ids, attention_mask=norm_attention_mask,
                                                return_dict=True)
            encoder_hidden_norm = encoder_outputs_norm.last_hidden_state
            logits_norm = self.lm_head(encoder_hidden_norm)


        translation_logits = None
        if decoder_input_ids is not None:

            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_orig,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )
            decoder_last_hidden_state = decoder_outputs.last_hidden_state
            translation_logits = self.lm_head(decoder_last_hidden_state)


        pos_logits = self.pos_classifier(self.pos_dropout(encoder_hidden_orig))


        loss_dict = {}
        loss_fct = nn.CrossEntropyLoss()


        if mlm_labels is not None and not torch.isnan(
                loss := loss_fct(logits_orig.view(-1, self.config.vocab_size), mlm_labels.view(-1))):
            loss_dict["mlm_loss"] = loss
        if tlm_labels is not None and not torch.isnan(
                loss := loss_fct(logits_orig.view(-1, self.config.vocab_size), tlm_labels.view(-1))):
            loss_dict["tlm_loss"] = loss

        if norm_mlm_labels is not None and logits_norm is not None and not torch.isnan(
                loss := loss_fct(logits_norm.view(-1, self.config.vocab_size), norm_mlm_labels.view(-1))):
            loss_dict["norm_mlm_loss"] = loss
        if norm_tlm_labels is not None and logits_norm is not None and not torch.isnan(
                loss := loss_fct(logits_norm.view(-1, self.config.vocab_size), norm_tlm_labels.view(-1))):
            loss_dict["norm_tlm_loss"] = loss




        if translation_labels is not None and translation_logits is not None:
            trans_loss = loss_fct(translation_logits.view(-1, self.config.vocab_size), translation_labels.view(-1))
            loss_dict["translation_loss"] = trans_loss

        if pos_labels is not None:
            pos_loss = loss_fct(pos_logits.view(-1, self.config.num_pos_labels), pos_labels.view(-1))
            loss_dict["pos_loss"] = pos_loss

        if not return_dict:
            return (None, logits_orig, pos_logits)

        return {
            "loss_dict": loss_dict,
            "logits_orig": logits_orig,
            "logits_norm": logits_norm,
            "pos_logits": pos_logits,
            "translation_logits": translation_logits,
        }
