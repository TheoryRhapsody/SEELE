import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from chinesebert.modeling_chinesebert import ChineseBertModel
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.roformer import RoFormerModel, RoFormerPreTrainedModel
from utils.args import parse_args
from utils.get_transformer_encoder import get_transformer_encoder, Type_aware_MultiHeadedAttention
import time

args = parse_args()
cur_device = torch.device(args.my_device)
model_name2model_cls = {
    "bert": (BertPreTrainedModel, BertModel),
    # "chinesebert": (BertPreTrainedModel, ChineseBertModel),
    "roformer": (RoFormerPreTrainedModel, RoFormerModel),
}

INF = 1e13
EPSILON = 1e-5

M_EPSILON = 1e-13

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, unsqueeze_dim=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("n , d -> n d", t, inv_freq)
        if unsqueeze_dim:
            freqs = freqs.unsqueeze(unsqueeze_dim)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)

    def forward(self, t, seqlen=-2, past_key_value_length=0):
        # t shape [bs, dim, seqlen, seqlen]
        sin, cos = (
            self.sin[past_key_value_length: past_key_value_length + seqlen, :],
            self.cos[past_key_value_length: past_key_value_length + seqlen, :],
        )
        t1, t2 = t[..., 0::2], t[..., 1::2]
        return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(
            -2, -1
        )
    

# head_&tail contrastive_loss
def decoder_head_tail_contrastive_loss(scores, labels_mat, mask,position=None)  -> torch.FloatTensor:
    # scores: bs,seq_len, types,types
    # labels_mat: bs, types, each_sample_label_num, 2
    # mask: bs, seq_len
    batch_size, seq_length, types = scores.size(0), scores.size(1),scores.size(2)
    mask_ = mask.unsqueeze(-1)
    mask__ = mask_.unsqueeze(-1)
    mask = (torch.ones_like(mask__) - mask__)*(-INF)
    scores = scores + mask
    scores = scores.view(batch_size, -1)
    # mask = mask__ .view(batch_size, -1)
    
    # scores = scores + mask
    scores = F.log_softmax(scores, dim=-1)
    indice_bs, indice_types, indice_label_num = torch.where(labels_mat[:,:,:,0]!=0)
    assert len(indice_bs) == len(indice_types) and len(indice_types) == len(indice_label_num), "Dimension Error"
    indice_contrastive = [ types*types*position[inds]+types*indice_types[inds]+indice_types[inds] for inds in range(len(indice_bs))]
    if len(indice_contrastive) == 0:
        loss_log_softmax = M_EPSILON
        print("Current batch has none event")
        return loss_log_softmax
        
    else:
        loss_log_softmax = scores [indice_bs,indice_contrastive]
        return -loss_log_softmax.mean()

# span-level contrastive learning
def span_contrastive_loss(scores, labels_mat, mask,position=None)  -> torch.FloatTensor:
    # scores: bs,num_types, seq_len,seq_len
    # labels_mat: bs, num_types, each_sample_label_num, 2
    # mask: bs, seq_len,seq_len
    batch_size, types,seq_length= scores.size(0), scores.size(1),scores.size(2)
    mask = mask.unsqueeze(1)
    mask = (torch.ones_like(mask) - mask)*(-INF)
    scores = scores + mask
    scores = scores.view(batch_size, -1)
    # mask = mask__ .view(batch_size, -1)
    # scores = scores + mask
    scores = F.log_softmax(scores, dim=-1)
    indice_bs, indice_types, indice_label_num = torch.where(labels_mat[:,:,:,0]!=0)
    assert len(indice_bs) == len(indice_types) and len(indice_types) == len(indice_label_num) and len(position) == len(indice_bs), "Dimension Error"

    indice_contrastive = [ seq_length*seq_length*indice_types[inds]+seq_length*position[inds][0]+position[inds][1] for inds in range(len(indice_bs))]
    if len(indice_contrastive) == 0:
        loss_log_softmax = M_EPSILON
        print("Current batch has none event")
        return loss_log_softmax
        
    else:
        loss_log_softmax = scores [indice_bs,indice_contrastive]
        return -loss_log_softmax.mean()

def encoder_head_tail_contrastive_loss(scores, labels_mat, mask,position=None)  -> torch.FloatTensor:
    # scores: # batch_size, num_labels, seq_length
    # labels_mat: batch_size, num_labels, each_sample_label_num, 2
    # mask: batch_size, seq_length
    batch_size, num_labels, seq_length = scores.size(0), scores.size(1),scores.size(2)
    mask = mask.unsqueeze(1)
    mask = (torch.ones_like(mask) - mask)*(-INF)
    scores = scores + mask
    scores = scores.view(batch_size, -1)
    # mask = mask__ .view(batch_size, -1)
    
    # scores = scores + mask
    scores = F.log_softmax(scores, dim=-1)
    indice_bs, indice_num_labels, indice_each_sample_label_num = torch.where(labels_mat[:,:,:,0]!=0)
    assert len(indice_bs) == len(indice_num_labels) and len(indice_num_labels) == len(indice_each_sample_label_num) and len(position) == len(indice_bs), "维度出错"
    indice_contrastive = [ seq_length*indice_num_labels[inds]+position[inds] for inds in range(len(indice_bs))]
    
    if len(indice_contrastive) == 0:
        loss_log_softmax = M_EPSILON
        print("Current batch has none event")
        return loss_log_softmax
        
    else:
        loss_log_softmax = scores [indice_bs,indice_contrastive]
        return -loss_log_softmax.mean()

        
class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, heads=12, head_size=64, RoPE=True, tril_mask=True, max_length=512,if_contrast=False):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.init_temperature =1.0
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * 2 * head_size)
        self.if_contrast = if_contrast

        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length, unsqueeze_dim=-2) # n1d
        if self.if_contrast:
            self.role_start_dense = nn.Linear(hidden_size, head_size)
            self.role_end_dense = nn.Linear(hidden_size, head_size)
            self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / self.init_temperature))
            self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / self.init_temperature))

    def forward(self, inputs, type_inputs=None, attention_mask=None):
        inputs = self.dense(inputs)
        bs, seqlen = inputs.shape[:2]

        # method 1
        inputs = inputs.reshape(bs, seqlen, self.heads, 2, self.head_size)
        # bs,seqlen,types,head_size
        qw, kw = inputs.unbind(axis=-2)

        if self.if_contrast:

            role_start_representation = F.normalize(self.role_start_dense(type_inputs),dim=-1)# types,head_size
            role_end_representation = F.normalize(self.role_end_dense(type_inputs),dim =-1)

            sequence_start_representations = F.normalize(qw,dim=-1) # bs,seqlen,types,head_size
            sequence_end_representations = F.normalize(kw,dim=-1)

            start_scores_mat = self.start_logit_scale.exp() * role_start_representation.unsqueeze(0) @ sequence_start_representations.transpose(2, 3) # bs,seqlen,types,types
            end_scores_mat = self.end_logit_scale.exp() * role_end_representation.unsqueeze(0) @ sequence_end_representations.transpose(2, 3)

        # method 2
        # inputs = inputs.reshape(bs, seqlen, self.heads, 2 * self.head_size)
        # qw, kw = inputs.chunk(2, axis=-1)

        # original
        # inputs = inputs.chunk(self.heads, axis=-1)
        # inputs = torch.stack(inputs, axis=-2)
        # qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE
        if self.RoPE:
            qw, kw = self.rotary(qw, seqlen), self.rotary(kw, seqlen)

        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # remove padding loss
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * INF

        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * INF


        if self.if_contrast:
            return logits / self.head_size ** 0.5, start_scores_mat, end_scores_mat
        else:
            return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):

    def __init__(self, hidden_size, heads=12, head_size=64, RoPE=True, tril_mask=True, max_length=512):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2)
        self.dense2 = nn.Linear(head_size * 2, heads * 2)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length)

    def forward(self, inputs, attention_mask=None):
        seqlen = inputs.shape[1]
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE
        if self.RoPE:
            qw, kw = self.rotary(qw, seqlen), self.rotary(kw, seqlen)
            
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size ** 0.5
        bias = self.dense2(inputs).transpose(1, 2) / 2  #'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # remove padding loss
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * INF

        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * INF

        return logits

# Sparse cross entropy for multi label classification
def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + INF
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=EPSILON, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss


def globalpointer_loss(y_pred, y_true):
    shape = y_pred.shape
    # bs, nclass, max_spo_num
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    # bs, nclass, seqlen * seqlen
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=True)
    return loss.sum(dim=1).mean()


# def get_auto_model(model_type):
#     parent_cls, base_cls = model_name2model_cls[model_type]
#     exist_add_pooler_layer = model_type in ["bert"]

class AutoModelGPLinker4EE(nn.Module):
    def __init__(self, config, head_size=64, use_efficient=False):
        super(AutoModelGPLinker4EE,self).__init__()
        # if exist_add_pooler_layer:
        #     setattr(
        #         self,
        #         self.base_model_prefix,
        #         base_cls(config, add_pooling_layer=False),
        #     )
        # else:
        #     setattr(self, self.base_model_prefix, base_cls(config))
        self.config = config
        self.text_encoder = BertModel.from_pretrained(
            config["bert_path"],
            cache_dir = config["cache_dir"],
            add_pooling_layer=False
        )
        self.type_encoder = BertModel.from_pretrained(
            config["bert_path"],
            cache_dir = config["cache_dir"],
            add_pooling_layer=False
        )
        self.init_temperature = 0.2
        self.attention_threshold = 0.5
        self.dropout = torch.nn.Dropout(config["dropout_rate"])
        # Construct a virtual event representation for each event type for contrastive learning at the event level
        self.event_type_embeddings = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(config["event_type_num"],config["hidden_size"]), mode='fan_in', nonlinearity='relu'))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / self.init_temperature))
        if config["contrastive_method"] == 'event_description' or config["contrastive_method"] == 'No':
            self.role_description_start_linear = torch.nn.Linear(config["hidden_size"], config["hidden_size"])
            self.role_description_end_linear = torch.nn.Linear(config["hidden_size"], config["hidden_size"])

        self.event_level_huberloss = torch.nn.HuberLoss(reduction='mean')


        if use_efficient:
            gpcls = EfficientGlobalPointer
        else:
            gpcls = GlobalPointer
        self.argu_output = gpcls(
            hidden_size=config["hidden_size"],
            heads=config["num_labels"],
            head_size=head_size,
            tril_mask=True,
            RoPE=True,
            if_contrast=False  # Only when self.config["contrastive_method"] == 'role_description', set if_contrast=True
        )
        self.head_output = gpcls(
            hidden_size=config["hidden_size"],
            heads=config["event_type_num"],
            head_size=head_size,
            RoPE=True,
            tril_mask=True,
            if_contrast=False
        )
        self.tail_output = gpcls(
            hidden_size=config["hidden_size"],
            heads=config["event_type_num"],
            head_size=head_size,
            RoPE=True,
            tril_mask=True,
            if_contrast=False
        )

    def forward(
        self,
        input_ids = None,
        attention_mask=None,
        type_inputs_ids = None, 
        type_token_type_ids = None,
        type_attention_mask = None,
        role_index_labels = None,
        labels=None,
        current_epoch_id = None,
        # output_attentions=None,
        # output_hidden_states=None,
    ):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
            # **kwargs
        )

        ''' 23.6.16 type encoder '''
        if self.config["contrastive_method"] == 'role_description':
            type_encoding_output = self.type_encoder(
                input_ids=type_inputs_ids.squeeze(0),
                attention_mask=type_attention_mask.squeeze(0),
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False,
                # **kwargs
            )
            # Taking the [CLS] token of the description, size: role_num*hidden_size
            type_cls_output = type_encoding_output[0][:, 0]


            last_hidden_state = outputs[0]
            argu_output, start_scores, end_scores = self.argu_output(
                last_hidden_state, attention_mask=attention_mask, type_inputs=type_cls_output 
            )
            head_output = self.head_output(
                last_hidden_state, attention_mask=attention_mask
            )
            tail_output = self.tail_output(
                last_hidden_state, attention_mask=attention_mask
            )

            aht_output = (argu_output, head_output, tail_output)
            loss = None
            loss_gp = None
            if labels is not None:
                arg_label = labels[0]
                indice_bs, indice_types, indice_label_num = torch.where(arg_label[:,:,:,0]!=0)
                start_pos = arg_label[indice_bs,indice_types, indice_label_num,torch.zeros_like(indice_bs)]
                start_contrastive_loss = decoder_head_tail_contrastive_loss(scores = start_scores, labels_mat = arg_label, mask = attention_mask,position=start_pos)
                end_pos = arg_label[indice_bs,indice_types, indice_label_num,torch.ones_like(indice_bs)]
                end_contrastive_loss = decoder_head_tail_contrastive_loss(scores = end_scores, labels_mat = arg_label, mask = attention_mask,position=end_pos)
                loss_gp = (
                    sum([globalpointer_loss(o, l) for o, l in zip(aht_output, labels)])
                    / 3 
                )
                loss =(
                    sum([globalpointer_loss(o, l) for o, l in zip(aht_output, labels)])
                    / 3 
                ) + 0.001*start_contrastive_loss + 0.001* end_contrastive_loss
            output = (aht_output,) + outputs[1:]
        
        elif self.config["contrastive_method"] == 'event_description' or self.config["contrastive_method"] == 'No':
            event_description_encoding_output = self.type_encoder(
                input_ids=type_inputs_ids.squeeze(0),
                attention_mask=type_attention_mask.squeeze(0),
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False,
                # **kwargs
            )
            # bert outputs: event_type_num *description_len*hidden_size, label matrix: event_type_num * max_label_num_within_single_event * 2
            last_hidden_state = outputs[0]
            batch_size, seq_length, _ = last_hidden_state.size()
            # bert outputs: event_type_num *description_len*hidden_size, label matrix: event_type_num * max_label_num_within_single_event * 2
            event_description_last_hidden_state = event_description_encoding_output[0]

            indice_event_type, indice_role = torch.where(role_index_labels[:,:,0]!=0)
            label_pos_list = role_index_labels[indice_event_type,indice_role]
            assert len(label_pos_list) == self.config["num_labels"], "Error in the number of labels"
            
            role_description_embedding =[torch.mean(event_description_last_hidden_state[indice_event_type[idx]][each_label[0]:each_label[1]+1],dim=0) for idx,each_label in enumerate(label_pos_list)]
            role_description_embedding = torch.tensor([item.cpu().detach().numpy() for item in role_description_embedding]).to(cur_device)

            role_start_representation = F.normalize(self.dropout(self.role_description_start_linear(role_description_embedding)), dim=-1)
            role_end_representation = F.normalize(self.dropout(self.role_description_end_linear(role_description_embedding)), dim=-1)
            # last_hidden_state_head_representation = F.normalize(self.dropout(self.last_hidden_state_mapping_head_linear(last_hidden_state)), dim=-1)
            # last_hidden_state_tail_representation = F.normalize(self.dropout(self.last_hidden_state_mapping_tail_linear(last_hidden_state)), dim=-1)
            last_hidden_state_head_representation = F.normalize(last_hidden_state, dim=-1)
            last_hidden_state_tail_representation = F.normalize(last_hidden_state, dim=-1)

            
            # batch_size x num_labels x seq_length
            role_start_scores = self.logit_scale.exp() * role_start_representation.unsqueeze(0) @ last_hidden_state_head_representation.transpose(1, 2)
            role_end_scores = self.logit_scale.exp() * role_end_representation.unsqueeze(0) @ last_hidden_state_tail_representation.transpose(1, 2)
            

            
            # all_span_mean_embedding = torch.zeros(batch_size,seq_length,seq_length,self.config["hidden_size"]).to(cur_device)
            # for batch_id, token_list in enumerate(last_hidden_state):
            #     for head_token_id in  range(len(token_list)):
            #         for tail_token_id in range(head_token_id,len(token_list)):
            #             all_span_mean_embedding[batch_id,head_token_id,tail_token_id,:] = torch.mean(last_hidden_state[batch_id][head_token_id:tail_token_id+1],dim=0)
            # setting mask
            # all_span_mask = torch.zeros(batch_size,seq_length,seq_length).to(cur_device)
            # for start_id in range(seq_length):
            #     for end_id in range(start_id,seq_length):
            #         if end_id - start_id < args.max_argument_len:
            #             all_span_mask[:,start_id,end_id] = 1
            # extend_mask = torch.zeros(batch_size,seq_length,seq_length).to(cur_device)
            # for idx in range(batch_size):
            #     extend_mask[idx,:,:] = torch.einsum('i,j->ij',attention_mask[idx],attention_mask[idx]) 
            # all_span_mask = torch.mul(all_span_mask,extend_mask)

            # normal_span_mean_embedding = F.normalize(all_span_mean_embedding.view(batch_size, seq_length * seq_length, -1), dim=-1)

            # normal_role_description_embedding = F.normalize(role_description_embedding ,dim=-1)

            # print(normal_role_description_embedding)
            # print(">"*100)
            # print(normal_span_mean_embedding)
            # exit()
            # span_scores = self.span_logit_scale.exp() * normal_role_description_embedding.unsqueeze(0) @ normal_span_mean_embedding.transpose(1, 2)
            # span_scores = span_scores.view(batch_size, self.config["num_labels"], seq_length, seq_length)
        


            if self.config["contrastive_level"] == 'token_level':
                # Linear layer for mapping event description [CLS] token
                event_description_cls_linear = torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"]).to(cur_device)
                # Linear layer for mapping last_hidded_state
                each_token_hidden_state_linear = torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"]).to(cur_device)
                # intra-event attention
                event_intra_transformer =  get_transformer_encoder(num_layers=1,hidden_size=self.config["hidden_size"]).to(cur_device)
                # intra-event attention
                event_inter_transformer =  get_transformer_encoder(num_layers=1,hidden_size=self.config["hidden_size"]).to(cur_device)
                # Virtual representation of each event type
                event_proxy_fusion_multi_attention = Type_aware_MultiHeadedAttention(h=8,d_model=self.config["hidden_size"],dropout=self.config["dropout_rate"]).to(cur_device)
                # Virtual representation of each event type
                event_proxy_fusion_multi_attention_for_each_grouped_token = Type_aware_MultiHeadedAttention(h=8,d_model=self.config["hidden_size"],dropout=self.config["dropout_rate"]).to(cur_device)
                
                event_proxy_fusion_self_attention_weight_linear_1 = torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"]).to(cur_device)
                event_proxy_fusion_self_attention_weight_linear_2 = torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"]).to(cur_device)


                # event_type_num *hidden_size
                event_description_cls_output = event_description_encoding_output[0][:, 0]
                #event_type_num *hidden_size
                event_description_cls_embedding = F.normalize(self.dropout(event_description_cls_linear(event_description_cls_output)), dim=-1)
                # batch_size * seq_length * hidden_size
                each_token_hidden_state_embedding = F.normalize(self.dropout(each_token_hidden_state_linear(last_hidden_state)), dim=-1)
                # batch_size * event_type_num * seq_length
                # Grouping each token based on the description of each event [CLS] token and the similarity of each token in the last hidden state, where tokens belonging to the same group can have mutual attention
                grouping_mat = event_description_cls_embedding.unsqueeze(0) @ each_token_hidden_state_embedding.transpose(1, 2)
                pre_mask = torch.zeros(batch_size,self.config["event_type_num"],seq_length).to(cur_device)
                pre_mask[torch.where(torch.sigmoid(grouping_mat)>0.5)] = 1.0

                type_aware_attention_mask = torch.einsum('ijk,ijl->ijkl',pre_mask,pre_mask)
                type_aware_attention_mask_ = torch.sum(type_aware_attention_mask,dim=1)
                post_mask = torch.zeros(batch_size,seq_length,seq_length).to(cur_device)
                post_mask[:] = torch.eye(seq_length)

                post_mask[torch.where(type_aware_attention_mask_>=1.0)] = 1.0

                event_intra_attention_mask = torch.mul(post_mask,torch.einsum('ij,ik->ijk',attention_mask,attention_mask))

                # batch_size * event_type_num * seq_length * hidden_size
                grouped_token_embbeding_for_each_event_type = torch.zeros(batch_size,self.config["event_type_num"],seq_length,self.config["hidden_size"]).to(cur_device)
                grouped_token_embbeding_for_each_event_type[torch.where(pre_mask==1.0)] = last_hidden_state[torch.where(pre_mask==1.0)[0],torch.where(pre_mask==1.0)[2]]
                 # batch_size * event_type_num * seq_length
                grouped_token_embbeding_attention_mask = torch.mul(pre_mask,attention_mask.unsqueeze(1))

                # batch_size * event_type_num * 1 * hidden_size
                input_query_for_ench_event_type = self.event_type_embeddings.unsqueeze(1).unsqueeze(0).repeat(batch_size,1,1,1)
                
                # batch_size, event_type_num, 1, hidden_size
                fusion_event_representation_for_each_grouped_token = event_proxy_fusion_multi_attention_for_each_grouped_token(input_query_for_ench_event_type, grouped_token_embbeding_for_each_event_type,grouped_token_embbeding_for_each_event_type, grouped_token_embbeding_attention_mask)

                #batch_size, event_type_num, hidden_size
                fusion_event_representation_as_query = F.normalize(self.dropout(event_proxy_fusion_self_attention_weight_linear_1(fusion_event_representation_for_each_grouped_token.squeeze(2))), dim=-1)

                fusion_event_representation_as_key = F.normalize(self.dropout(event_proxy_fusion_self_attention_weight_linear_2(fusion_event_representation_for_each_grouped_token.squeeze(2))), dim=-1)
                
                #batch_size, event_type_num, event_type_num
                related_event_mat = fusion_event_representation_as_query @ fusion_event_representation_as_key.transpose(1, 2)

                event_relation_pre_mask = torch.zeros(batch_size,self.config["event_type_num"],self.config["event_type_num"],seq_length).to(cur_device)
                event_relation_pre_mask[torch.where(torch.sigmoid(related_event_mat)>0.5)] = pre_mask[torch.where(torch.sigmoid(related_event_mat)>0.5)[0], torch.where(torch.sigmoid(related_event_mat)>0.5)[2]]
                
                #batch_size, event_type_num, seq_length,seq_length
                event_relation_pre_mask_ = torch.einsum('ijk,ijl->ijkl',torch.sum(event_relation_pre_mask,dim=2),torch.sum(event_relation_pre_mask,dim=2))

                event_relation_post_mask = torch.zeros(batch_size,seq_length,seq_length).to(cur_device)
                event_relation_post_mask[:] = torch.eye(seq_length)
                #batch_size,seq_length,seq_length
                event_relation_post_mask[torch.where(torch.sum(event_relation_pre_mask_,dim=1)>=1.0)] = 1.0
                event_inter_attention_mask = torch.mul(event_relation_post_mask,torch.einsum('ij,ik->ijk',attention_mask,attention_mask))

                if current_epoch_id >1:
                    event_intra_hidden_state = event_intra_transformer(last_hidden_state,event_intra_attention_mask)

                    event_inter_hidden_state = event_inter_transformer(event_intra_hidden_state,event_inter_attention_mask)
                else:
                    event_inter_hidden_state = last_hidden_state
            else:
                event_inter_hidden_state = last_hidden_state



            argu_output = self.argu_output(
                event_inter_hidden_state, attention_mask=attention_mask
            )
            head_output = self.head_output(
                event_inter_hidden_state, attention_mask=attention_mask
            )
            tail_output = self.tail_output(
                event_inter_hidden_state, attention_mask=attention_mask
            )

            aht_output = (argu_output, head_output, tail_output)
            # print(argu_output.shape)
            # print(head_output.shape)
            # print(tail_output.shape)
            loss = None
            loss_gp = None
            if labels is not None:
                arg_label = labels[0]
                batch_size_, num_labels_, each_sample_label_num_ = arg_label.size(0), arg_label.size(1),arg_label.size(2)
                indice_bs, indice_types, indice_label_num = torch.where(arg_label[:,:,:,0]!=0)
                
                indice_event_types = self.config["labels_to_event_type"][indice_types][:,0]
                indice_role_types =  self.config["labels_to_event_type"][indice_types][:,1]

                role_start_position = arg_label[indice_bs,indice_types, indice_label_num,torch.zeros_like(indice_bs)]
                role_end_position = arg_label[indice_bs,indice_types, indice_label_num,torch.ones_like(indice_bs)]
                 
                assert len(indice_bs) == len(indice_event_types) and len(indice_bs) == len(role_end_position),"维度出错！！"
                


                if self.config["contrastive_level"] == 'token_level':
                    #batch_size * event_type_num * seq_length * hidden_size
                    argument_span_embbeding_for_each_event_type = torch.zeros(batch_size,self.config["event_type_num"],seq_length,self.config["hidden_size"]).to(cur_device)

                    for each_ids in range(len(indice_bs)):
                        argument_span_embbeding_for_each_event_type[indice_bs[each_ids],indice_event_types[each_ids],role_start_position[each_ids]:role_end_position[each_ids]+1]= last_hidden_state[indice_bs[each_ids],role_start_position[each_ids]:role_end_position[each_ids]+1]

                    #batch_size * event_type_num * seq_length
                    event_proxy_fusion_attention_mask =torch.zeros(batch_size,self.config["event_type_num"],seq_length).to(cur_device)

                    no_meaning_hidden_size = torch.zeros(self.config["hidden_size"]).to(cur_device)
                    event_proxy_fusion_attention_mask[torch.where(torch.all(argument_span_embbeding_for_each_event_type!=no_meaning_hidden_size,dim =-1))] = 1.0
                    

                    # batch_size,event_type_num,1,hidden_size
                    input_query = self.event_type_embeddings.unsqueeze(1).unsqueeze(0).repeat(batch_size,1,1,1)
                    
                    # batch_size, event_type_num, 1, hidden_size
                    fusion_event_proxy =  event_proxy_fusion_multi_attention(input_query,argument_span_embbeding_for_each_event_type,argument_span_embbeding_for_each_event_type,event_proxy_fusion_attention_mask)
                    # event_level huber_loss
                    event_level_huberloss = self.event_level_huberloss(fusion_event_proxy.squeeze(2),event_description_cls_embedding.unsqueeze(0).repeat(batch_size,1,1))

                role_start_contrastive_loss = encoder_head_tail_contrastive_loss(scores = role_start_scores, labels_mat = arg_label, mask = attention_mask,position=role_start_position)
                role_end_contrastive_loss = encoder_head_tail_contrastive_loss(scores = role_end_scores, labels_mat = arg_label, mask = attention_mask,position=role_end_position)
                # start_end_postion  = arg_label[indice_bs,indice_types, indice_label_num]
                # span_level_contrastive_loss = span_contrastive_loss(scores = span_scores, labels_mat = arg_label, mask = all_span_mask,position=start_end_postion)
                loss_gp = (
                    sum([globalpointer_loss(o, l) for o, l in zip(aht_output, labels)])
                    / 3 
                )
                loss =(
                    sum([globalpointer_loss(o, l) for o, l in zip(aht_output, labels)])
                    / 3 
                ) + self.config["description_loss_weight"] *role_start_contrastive_loss +self.config ["description_loss_weight"]*role_end_contrastive_loss #+ 0.1*event_level_huberloss
                # print("gp_loss: ",loss_gp)
                # print("huber_loss: ",event_level_huberloss)
                # print("contrastive_loss: ",role_start_contrastive_loss)
            output = (aht_output,) + outputs[1:]

        
        return ((loss,loss_gp,) + output) if loss is not None else output


    # return AutoModelGPLinker4EE


