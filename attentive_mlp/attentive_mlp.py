import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skorch import NeuralNet
from skorch.callbacks import EpochTimer, PassthroughScoring, EpochScoring, PrintLog
from skorch.callbacks.lr_scheduler import LRScheduler
from skorch.dataset import ValidSplit

from sklearn.base import BaseEstimator, ClassifierMixin
from skorch.utils import is_dataset, get_dim


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / torch.sqrt(torch.tensor(d_k))
    print(attn_logits.shape)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    print(values.shape)
    return values, attention


class FeatureEmbeddings(nn.Module):

    def __init__(self, features_dim, embedding_dim, dropout=0.0):

        super(FeatureEmbeddings, self).__init__()

        self.norm = nn.BatchNorm1d(features_dim)
        # self.centers = nn.Parameter(torch.zeros((features_dim, 1)))
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Linear(1, embedding_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.norm(x)

        # differences = torch.abs(self.centers - x)
        # x = torch.concat([x, differences], -1)

        x = self.dropout(x)
        x = self.projector(x)

        return x


class DenseLayer(nn.Module):

    def __init__(self, bn_dim, input_dim, output_dim, is_residual=False, activation=nn.GELU):

        super(DenseLayer, self).__init__()

        self.norm = nn.BatchNorm1d(bn_dim)
        self.is_residual = is_residual and input_dim == output_dim

        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            activation()
        )

    def forward(self, x):
        x = self.norm(x)
        out = self.layer(x)

        if self.is_residual:
            return x + out
        return out


class MultiheadSelfAttention(nn.Module):

    def __init__(self, features_dim, embedding_dim, num_heads, add_bias=True):

        super(MultiheadSelfAttention, self).__init__()

        assert embedding_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.norm = nn.BatchNorm1d(features_dim)

        # Соединяем все матрицы весов 1...h вместе для эффективности
        # В оригинальной реализации bias=False, но это опционально
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=add_bias)

        self._reset_parameters()

    def _reset_parameters(self):
        # Оригинальная инициализация из Transformer
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            self.qkv_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):

        x = self.norm(x)

        # batch_size, features_len = x.size()
        batch_size, features_dim, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Отделяем Q, K, V из выхода linear слоя
        qkv = qkv.reshape(batch_size, features_dim, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Применяем scaled dot product
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, features_dim, embed_dim)

        return (values, attention) if return_attention else values


class AttentiveMLP(nn.Module):

    def __init__(self, features_dim, embedding_dim, output_dim, num_heads, hidden_layers, hidden_dim_scale, dropout,
                 add_attention=True, add_softmax=False):
        """
        Inputs:
            input_dim - Количество фичей на входе
            num_heads - Количество голов используемых в attention-блоке
            num_classes - Количество классов на выходе
            encoder_blocks - Количество ResidualLinearLayer перед Self-Attention
            hidden_dim_scale - Преобразование признакового пространства после Attention
            dropout - Процент дропаута
        """
        super(AttentiveMLP, self).__init__()

        self.add_softmax = add_softmax
        self.add_attention = add_attention

        self.embedder = FeatureEmbeddings(features_dim, embedding_dim, dropout)
        # self.input_encoder = DenseLayer(features_dim, embedding_dim, embedding_dim)

        # Attentive Блок
        if self.add_attention:
            self.self_attention = MultiheadSelfAttention(features_dim, embedding_dim, num_heads, add_bias=False)

        self.flatten = nn.Flatten(-2)
        flattened_dim = features_dim * embedding_dim

        self.hidden_layers = nn.Sequential(*[
            DenseLayer(int(flattened_dim * pow(hidden_dim_scale, i)),
                       int(flattened_dim * pow(hidden_dim_scale, i)),
                       int(flattened_dim * pow(hidden_dim_scale, i + 1)))
            for i in range(0, hidden_layers)
        ])

        self.answer_norm = nn.BatchNorm1d(int(flattened_dim * pow(hidden_dim_scale, hidden_layers)))
        self.answer_proj = nn.Linear(int(flattened_dim * pow(hidden_dim_scale, hidden_layers)), output_dim)

        if self.add_softmax:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        x = self.embedder(x)
        # x = self.input_encoder(x)

        if self.add_attention:
            x = self.self_attention(x, mask, return_attention=False)

        x = self.flatten(x)

        x = self.hidden_layers(x)

        x = self.answer_norm(x)
        x = self.answer_proj(x)

        if self.add_softmax:
            x = self.softmax(x)

        return x

    def get_attention_map(self, x, do_norm=True, mask=None):
        if do_norm:
            x = self.embedder(x)
            # x = self.input_encoder(x)
        values, atten_map = self.self_attention(x, mask, return_attention=True)
        return values, atten_map


class AttentiveMLPClassifier(NeuralNet, BaseEstimator, ClassifierMixin):

    def __init__(self,
                 features_dim,
                 embedding_dim,
                 num_classes,
                 num_heads,
                 hidden_layers=0,
                 hidden_dim_scale=0.7,
                 dropout=0.1,
                 device='cpu',
                 max_epochs=30,
                 batch_size=512,
                 add_attention=True,
                 warm_start=False,
                 optimizer=optim.Adam,
                 optimizer__weight_decay=0.001,
                 verbose=1,
                 train_split=ValidSplit(5, stratified=True),
                 lr_scheduler=LRScheduler(max_lr=0.01),
                 iterator_train__shuffle=True,
                 **kwargs):

        assert embedding_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        assert num_classes > 1, "Number of classes must be greater than 1"
        assert hidden_dim_scale > 0.1, "hidden_dim_scale must be greater than 0.1"
        assert num_heads > 0, "Number of heads must be greater than 0"

        self.lr_scheduler = lr_scheduler

        super().__init__(
            module=AttentiveMLP,
            criterion=nn.NLLLoss,
            module__features_dim=features_dim,
            module__embedding_dim=embedding_dim,
            module__output_dim=num_classes,
            module__num_heads=num_heads,
            module__hidden_layers=hidden_layers,
            module__hidden_dim_scale=hidden_dim_scale,
            module__dropout=dropout,
            module__add_attention=add_attention,
            module__add_softmax=True,
            batch_size=batch_size,
            max_epochs=max_epochs,
            optimizer=optimizer,
            optimizer__weight_decay=optimizer__weight_decay,
            verbose=verbose,
            train_split=train_split,
            warm_start=warm_start,
            iterator_train__shuffle=iterator_train__shuffle,
            device=device,
            **kwargs
        )

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
            ('lr_scheduler', self.lr_scheduler)
        ]

    def check_data(self, X, y=None):
        super().check_data(X, y)
        if (not is_dataset(X)) and (get_dim(y) != 1):
            raise ValueError("The target train_data should be 1-dimensional.")

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if isinstance(self.criterion_, torch.nn.NLLLoss):
            eps = torch.finfo(y_pred.dtype).eps
            y_pred = torch.log(y_pred + eps)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    def fit(self, X, y=None, **fit_params):
        return super(AttentiveMLPClassifier, self).fit(X, y, **fit_params)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(-1)

    def get_attention_map(self, example: torch.Tensor, do_norm=True, mask=None):
        if not self.initialized_:
            raise Exception('Module is not initialized')
        example = example.to(torch.device(self.device))
        value, map = self.module_.get_attention_map(example, do_norm, mask)
        return value.squeeze(0).detach(), map.squeeze(0).detach()

    def get_params(self, deep=True, **kwargs):
        params = super().get_params(deep=deep, **kwargs)
        exclude_params = ['optimizer_', 'criterion', 'module', '_kwargs', 'is_binary']
        params = {key: val for key, val in params.items() if key not in exclude_params}
        params = {key: val for key, val in params.items() if not key.startswith('module_')}
        params.update({
            'features_dim': self.module__features_dim,
            'embedding_dim': self.module__embedding_dim,
            'num_classes': self.module__output_dim,
            'num_heads': self.module__num_heads,
            'hidden_layers': self.module__hidden_layers,
            'hidden_dim_scale': self.module__hidden_dim_scale,
            'dropout': self.module__dropout,
            'add_attention': self.module__add_attention
        })
        return params
