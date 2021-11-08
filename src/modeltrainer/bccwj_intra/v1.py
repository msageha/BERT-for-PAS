import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers import BertTokenizer
from transformers import AutoTokenizer, BertModel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 日本語学習済モデルをロードする
        # output_attentions=Trueで順伝播のときにattention weightを受け取れるようにする
        # output_hidden_state=Trueで12層のBertLayerの隠れ層を取得する
        self.bert = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            output_attentions=True,
            output_hidden_states=True,
        )

        # BERTの隠れ層の次元数は768だが、最終4層分のベクトルを結合したものを扱うので、768×4次元としている。
        # 精度が出やすいらしい
        self.linear_ga = nn.Linear(768 * 4, 2)
        self.linear_wo = nn.Linear(768 * 4, 2)
        self.linear_ni = nn.Linear(768 * 4, 2)

        # 重み初期化処理
        nn.init.normal_(self.linear_ga.weight, std=0.02)
        nn.init.normal_(self.linear_ga.bias, 0)
        nn.init.normal_(self.linear_wo.weight, std=0.02)
        nn.init.normal_(self.linear_wo.bias, 0)
        nn.init.normal_(self.linear_ni.weight, std=0.02)
        nn.init.normal_(self.linear_ni.bias, 0)

    # clsトークンのベクトルを取得する用の関数を用意
    def _get_cls_vec(self, vec):
        return vec[:, 0, :].view(-1, 768)

    def forward(self, input_ids, input_target):

        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.bert(input_ids)
        attentions = output["attentions"]
        hidden_states = output["hidden_states"]

        # 最終４層の隠れ層からそれぞれclsトークンのベクトルを取得する
        vec1 = self._get_cls_vec(hidden_states[-1])
        vec2 = self._get_cls_vec(hidden_states[-2])
        vec3 = self._get_cls_vec(hidden_states[-3])
        vec4 = self._get_cls_vec(hidden_states[-4])

        # 4つのclsトークンを結合して１つのベクトルにする。
        vec = torch.cat([vec1, vec2, vec3, vec4], dim=1)

        # 全結合層でクラス分類用に次元を変換
        out = self.linear(vec)

        return F.log_softmax(out, dim=1), attentions


classifier = Model()
