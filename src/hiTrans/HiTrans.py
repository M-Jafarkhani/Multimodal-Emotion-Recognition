import torch
import torch.nn as nn
from transformers import BertModel
import math


class HiTrans(nn.Module):
    def __init__(
        self,
        hidden_dim,
        emotion_class_num,
        d_model,
        d_ff,
        heads,
        layers,
        dropout=0,
        input_max_length=512,
    ):
        super(HiTrans, self).__init__()
        self.input_max_length = input_max_length
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.encoder = TransformerEncoder(d_model, d_ff, heads, layers, 0.1)

        self.emo_output_layer = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, emotion_class_num)
        )

        self.spk_output_layer = SpeakerMatchLayer(hidden_dim, dropout)

    def forward(self, dia_input, cls_index, mask):
        bert_outputs = []
        for i in range((dia_input.size(1) - 1) // self.input_max_length + 1):
            cur_input = dia_input[
                :, i * self.input_max_length : (i + 1) * self.input_max_length
            ]
            cur_mask = cur_input.ne(0)
            bert_output, _ = self.bert(cur_input, cur_mask)
            bert_outputs.append(bert_output)
        bert_outputs = torch.cat(bert_outputs, dim=1)

        bert_outputs = bert_outputs[
            torch.arange(bert_outputs.size(0)).unsqueeze(1), cls_index.long()
        ]
        bert_outputs = bert_outputs * mask[:, :, None].float()

        bert_outputs = self.encoder(bert_outputs, mask)
        emo_output = self.emo_output_layer(bert_outputs)
        spk_output = self.spk_output_layer(bert_outputs)

        return emo_output, spk_output


class SpeakerMatchLayer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(SpeakerMatchLayer, self).__init__()
        self.mpl1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.mpl2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.biaffine = Biaffine(hidden_dim // 2, 2)

    def forward(self, x):
        x1 = self.mpl1(x)
        x2 = self.mpl2(x)
        output = self.biaffine(x1, x2)
        return output


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        s = s.permute(0, 2, 3, 1)
        return s


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).contiguous().view(batch_size, -1,
                                                                                   head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.pos_emb(x)
        x = self.dropout(x)
        for i in range(self.layers):
            x = self.transformer_inter[i](i, x, x, mask.eq(0))
        return x

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))