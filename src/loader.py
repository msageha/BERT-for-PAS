from collections import defaultdict
import gensim
import json
import numpy as np
import pandas as pd
import pickle
import re
import torch
from prefetch_generator import BackgroundGenerator


def is_num(text):
    m = re.match("\A[0-9]+\Z", text)
    if m:
        return True
    else:
        return False


def extraction_num(text):
    m = re.search(r"([-0-9]+)", text)
    return int(m.group(1))


class WordVector:
    def __init__(self, emb_type, path=None):
        self.index2word = []
        if emb_type == "Random":
            self.index2word += ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
            with open(path, "r") as f:
                for line in f:
                    word = line.strip()
                    self.index2word.append(word)
            self.vectors = None

        elif (
            emb_type == "Word2Vec"
            or emb_type == "FastText"
            or emb_type == "Word2VecWiki"
        ):
            if emb_type == "Word2Vec" or emb_type == "FastText":
                model = self.load_word_vector(path, is_w2v_format=False)
            else:
                model = self.load_word_vector(path, is_w2v_format=True)

            # PAD
            self.index2word.append("<PAD>")
            pad_vector = np.zeros((1, 200))
            # BOS
            self.index2word.append("<BOS>")
            bos_vector = 2 * np.random.rand(1, 200) - 1
            # EOS
            self.index2word.append("<EOS>")
            eos_vector = 2 * np.random.rand(1, 200) - 1
            # UNK
            self.index2word.append("<UNK>")
            unk_vector = 2 * np.random.rand(1, 200) - 1

            # # for UNK each feature0
            # feature0 = [
            #     '""', '助詞', '未知語', 'URL', '言いよどみ', '連体詞', 'ローマ字文',
            #     'web誤脱', '英単語', '接頭辞', '助動詞', '接尾辞', '記号', '動詞',
            #     '漢文', '副詞', '形容詞', '接続詞', '補助記号', '代名詞', '名詞',
            #     '形状詞', '空白', '感動詞'
            # ]
            # for feature in feature0:
            #     self.index2word.append(f'<UNK:{feature}>')

            self.index2word += model.wv.index2word.copy()
            self.vectors = np.vstack(
                (
                    pad_vector,
                    bos_vector,
                    eos_vector,
                    unk_vector,
                    model.wv.vectors.copy(),
                )
            )

        else:
            print(f"unexpected emb_type: {emb_type}. Please check it.")

        self.word2index = {word: i for i, word in enumerate(self.index2word)}

    def load_word_vector(self, path, is_w2v_format=False):
        print(f"--- start loading Word Vector from {path} ---")
        if is_w2v_format:
            model = gensim.models.KeyedVectors.load_word2vec_format(path)
        else:
            model = gensim.models.KeyedVectors.load(path)
        return model


class FeatureToEmbedID:
    def __init__(self):
        self.feature_size_dict = {
            "feature:0": 24,
            "feature:1": 25,
            "feature:2": 11,
            "feature:3": 5,
            "feature:4": 93,
            "feature:5": 31,
            "feature:6": 30119,
            "feature:7": 35418,
            "feature:8": 1,
            "feature:9": 1,
            "feature:10": 5545,
            "feature:11": 1,
            "feature:12": 7,
            "feature:13": 1,
            "feature:14": 5,
            "feature:15": 1,
            "feature:16": 1,
        }

        self.feature0 = {
            "": 1,
            "助詞": 2,
            "未知語": 3,
            "URL": 4,
            "言いよどみ": 5,
            "連体詞": 6,
            "ローマ字文": 7,
            "web誤脱": 8,
            "英単語": 9,
            "接頭辞": 10,
            "助動詞": 11,
            "接尾辞": 12,
            "記号": 13,
            "動詞": 14,
            "漢文": 15,
            "副詞": 16,
            "形容詞": 17,
            "接続詞": 18,
            "補助記号": 19,
            "代名詞": 20,
            "名詞": 21,
            "形状詞": 22,
            "空白": 23,
            "感動詞": 24,
        }

        self.feature1 = {
            "": 1,
            "ＡＡ": 2,
            "形状詞的": 3,
            "一般": 4,
            "括弧閉": 5,
            "終助詞": 6,
            "フィラー": 7,
            "係助詞": 8,
            "句点": 9,
            "普通名詞": 10,
            "数詞": 11,
            "固有名詞": 12,
            "準体助詞": 13,
            "タリ": 14,
            "括弧開": 15,
            "読点": 16,
            "形容詞的": 17,
            "動詞的": 18,
            "名詞的": 19,
            "格助詞": 20,
            "接続助詞": 21,
            "助動詞語幹": 22,
            "非自立可能": 23,
            "文字": 24,
            "副助詞": 25,
        }

        self.feature2 = {
            "": 1,
            "助数詞可能": 2,
            "一般": 3,
            "副詞可能": 4,
            "人名": 5,
            "サ変形状詞可能": 6,
            "顔文字": 7,
            "助数詞": 8,
            "地名": 9,
            "サ変可能": 10,
            "形状詞可能": 11,
        }

        self.feature3 = {"": 1, "国": 2, "名": 3, "姓": 4, "一般": 5}

        self.feature4 = {
            "": 1,
            "サ行変格": 2,
            "文語助動詞-ヌ": 3,
            "文語下二段-サ行": 4,
            "文語下二段-ラ行": 5,
            "下一段-バ行": 6,
            "下一段-サ行": 7,
            "文語四段-タ行": 8,
            "助動詞-ヌ": 9,
            "文語サ行変格": 10,
            "下一段-ザ行": 11,
            "文語助動詞-タリ-完了": 12,
            "文語助動詞-ゴトシ": 13,
            "下一段-カ行": 14,
            "助動詞-レル": 15,
            "文語助動詞-ナリ-断定": 16,
            "文語ラ行変格": 17,
            "文語四段-ハ行": 18,
            "下一段-ガ行": 19,
            "形容詞": 20,
            "五段-バ行": 21,
            "下一段-ナ行": 22,
            "助動詞-ラシイ": 23,
            "文語助動詞-ズ": 24,
            "助動詞-ナイ": 25,
            "五段-サ行": 26,
            "五段-タ行": 27,
            "文語助動詞-ケリ": 28,
            "助動詞-ダ": 29,
            "文語上一段-ナ行": 30,
            "文語四段-マ行": 31,
            "上一段-マ行": 32,
            "文語下二段-ダ行": 33,
            "文語助動詞-キ": 34,
            "文語上一段-マ行": 35,
            "文語助動詞-ベシ": 36,
            "文語助動詞-ナリ-伝聞": 37,
            "助動詞-ナンダ": 38,
            "上一段-バ行": 39,
            "助動詞-ジャ": 40,
            "文語形容詞-ク": 41,
            "文語上二段-ダ行": 42,
            "文語下二段-タ行": 43,
            "文語助動詞-タリ-断定": 44,
            "文語下二段-ハ行": 45,
            "文語四段-ガ行": 46,
            "文語下二段-マ行": 47,
            "文語助動詞-リ": 48,
            "無変化型": 49,
            "助動詞-ヘン": 50,
            "文語下二段-ナ行": 51,
            "上一段-ア行": 52,
            "上一段-ガ行": 53,
            "助動詞-デス": 54,
            "五段-カ行": 55,
            "助動詞-タ": 56,
            "上一段-ザ行": 57,
            "助動詞-タイ": 58,
            "カ行変格": 59,
            "五段-ガ行": 60,
            "五段-ナ行": 61,
            "文語上二段-バ行": 62,
            "助動詞-ヤス": 63,
            "五段-ワア行": 64,
            "上一段-ラ行": 65,
            "文語助動詞-ム": 66,
            "上一段-ナ行": 67,
            "五段-マ行": 68,
            "文語形容詞-シク": 69,
            "五段-ラ行": 70,
            "文語四段-ラ行": 71,
            "下一段-ラ行": 72,
            "文語四段-サ行": 73,
            "文語四段-カ行": 74,
            "文語助動詞-ラシ": 75,
            "助動詞-ヤ": 76,
            "文語下一段-カ行": 77,
            "助動詞-マイ": 78,
            "文語下二段-ガ行": 79,
            "助動詞-マス": 80,
            "文語助動詞-マジ": 81,
            "文語カ行変格": 82,
            "下一段-タ行": 83,
            "下一段-ダ行": 84,
            "上一段-カ行": 85,
            "文語上二段-ハ行": 86,
            "下一段-ハ行": 87,
            "文語助動詞-ジ": 88,
            "上一段-タ行": 89,
            "下一段-マ行": 90,
            "文語下二段-カ行": 91,
            "文語下二段-ア行": 92,
            "下一段-ア行": 93,
        }

        self.feature5 = {
            "": 1,
            "連用形-イ音便": 2,
            "連体形-撥音便": 3,
            "連用形-一般": 4,
            "語幹-一般": 5,
            "ク語法": 6,
            "終止形-融合": 7,
            "未然形-サ": 8,
            "終止形-一般": 9,
            "語幹-サ": 10,
            "已然形-一般": 11,
            "未然形-撥音便": 12,
            "仮定形-一般": 13,
            "連体形-一般": 14,
            "連体形-省略": 15,
            "未然形-補助": 16,
            "連用形-ニ": 17,
            "仮定形-融合": 18,
            "終止形-促音便": 19,
            "終止形-ウ音便": 20,
            "未然形-一般": 21,
            "連用形-促音便": 22,
            "終止形-撥音便": 23,
            "未然形-セ": 24,
            "意志推量形": 25,
            "命令形": 26,
            "連用形-省略": 27,
            "連用形-撥音便": 28,
            "連用形-ウ音便": 29,
            "連体形-補助": 30,
            "連用形-融合": 31,
        }


class TokenDataFrame:
    def __init__(self):
        self.df = pd.DataFrame(
            columns=[
                "n単語目",
                "単語",
                "形態素0",
                "形態素1",
                "形態素2",
                "形態素3",
                "形態素4",
                "形態素5",
                "id",
                "eq",
                "ga",
                "ga_type",
                "o",
                "o_type",
                "ni",
                "ni_type",
                "verb_type",
                "n文節目",
                "係り先文節",
                "is主辞",
                "is機能語",
                "n文目",
                "is文末",
            ]
        )

    def bos(self):
        df = pd.DataFrame(
            [[-1, "<BOS>", "", "", "", "", "", "", -1, "-1F", 0, 0, -1]],
            columns=[
                "n単語目",
                "単語",
                "形態素0",
                "形態素1",
                "形態素2",
                "形態素3",
                "形態素4",
                "形態素5",
                "n文節目",
                "係り先文節",
                "is主辞",
                "is機能語",
                "n文目",
            ],
        )
        return df

    def eos(self):
        df = pd.DataFrame(
            [[-1, "<EOS>", "", "", "", "", "", "", -1, "-1F", 0, 0, -1]],
            columns=[
                "n単語目",
                "単語",
                "形態素0",
                "形態素1",
                "形態素2",
                "形態素3",
                "形態素4",
                "形態素5",
                "n文節目",
                "係り先文節",
                "is主辞",
                "is機能語",
                "n文目",
            ],
        )
        return df

    def sep(self):
        df = pd.DataFrame(
            [[-1, "<SEP>", "", "", "", "", "", "", -1, "-1F", 0, 0, -1]],
            columns=[
                "n単語目",
                "単語",
                "形態素0",
                "形態素1",
                "形態素2",
                "形態素3",
                "形態素4",
                "形態素5",
                "n文節目",
                "係り先文節",
                "is主辞",
                "is機能語",
                "n文目",
            ],
        )
        return df

    def pad(self):
        df = pd.DataFrame(
            [[-1, "<PAD>", "", "", "", "", "", "", -1, "-1F", 0, 0, -1]],
            columns=[
                "n単語目",
                "単語",
                "形態素0",
                "形態素1",
                "形態素2",
                "形態素3",
                "形態素4",
                "形態素5",
                "n文節目",
                "係り先文節",
                "is主辞",
                "is機能語",
                "n文目",
            ],
        )
        return df


class DatasetLoading:
    def __init__(
        self,
        emb_type,
        emb_path,
        media=["OC", "OY", "OW", "PB", "PM", "PN"],
        pickle_path="../datasets.pickle",
    ):
        self.media = media
        self.pickle_path = pickle_path
        self.emb_type = emb_type
        self.emb_path = emb_path

        datasets = defaultdict(list)
        with open(pickle_path, "rb") as f:
            print(f"--- start loading datasets pickle from {pickle_path} ---")
            _datasets = pickle.load(f)

        for domain in self.media:  # メディアごとに処理を行う
            for file in _datasets:
                if domain in file:
                    _dataset = _datasets[file]
                    datasets[domain].append((_dataset, file))
        self.datasets = datasets
        self.wv = WordVector(emb_type, emb_path)
        self.fe = FeatureToEmbedID()
        self.token_df = TokenDataFrame()

    def making_intra_df(self):
        datasets_intra = defaultdict(list)
        for domain in self.media:
            print(f"--- start loading {domain} ---")
            for df, file in self.datasets[domain]:
                for sentential_df in self._to_sentential_df(df):
                    for x, y in self._df_to_intra_vector(sentential_df):
                        datasets_intra[domain].append((x, y, file))
        self.datasets_intra = datasets_intra

    def split_each_domain(self, type, sizes=[0.7, 0.1, 0.2]):
        if type == "intra":
            dataset = self.datasets_intra
        else:
            raise ValueError()
        trains_dict = {}
        vals_dict = {}
        tests_dict = {}
        for domain in self.media:
            dataset_size = len(dataset[domain])
            end = int(sizes[0] / sum(sizes) * dataset_size)
            trains_dict[domain] = np.array(dataset[domain][:end])
            start = end
            end += int(sizes[1] / sum(sizes) * dataset_size)
            vals_dict[domain] = np.array(dataset[domain][start:end])
            start = end
            tests_dict[domain] = np.array(dataset[domain][start:])
        return trains_dict, vals_dict, tests_dict

    def _to_sentential_df(self, df):
        last_sentence_indices = df["is文末"][df["is文末"] == True].index
        start = 0
        for index in last_sentence_indices:
            end = index
            yield df.loc[start:end]
            start = index + 1

    def _case_id_to_index(self, df, case_id, case_type, is_intra):
        if (
            case_type == "none"
            or case_type == "exoX"
            or case_type == "exo2"
            or case_type == "exo1"
        ):
            return str((df["単語"] == "<EOS>").idxmax())
        elif is_intra and case_type == "inter(zero)":
            return str((df["単語"] == "<EOS>").idxmax())
        else:
            return str((df["id"] == case_id).idxmax())

    def _df_to_intra_vector(self, df):
        df = pd.concat(
            [self.token_df.bos(), df, self.token_df.eos()],
            ignore_index=True,
            sort=False,
        )
        df["単語ID"] = -1
        df["形態素0ID"] = -1
        df["形態素1ID"] = -1
        df["形態素2ID"] = -1
        df["形態素3ID"] = -1
        df["形態素4ID"] = -1
        df["形態素5ID"] = -1
        for index, row in df.iterrows():
            # 各単語についての処理
            if row["単語"] in self.wv.word2index:
                df.loc[index, "単語ID"] = self.wv.word2index[row["単語"]]
            else:
                df.loc[index, "単語ID"] = self.wv.word2index["<UNK>"]

            # 形態素素性
            df.loc[index, "形態素0ID"] = self.fe.feature0[row["形態素0"]]
            df.loc[index, "形態素1ID"] = self.fe.feature1[row["形態素1"]]
            df.loc[index, "形態素2ID"] = self.fe.feature2[row["形態素2"]]
            df.loc[index, "形態素3ID"] = self.fe.feature3[row["形態素3"]]
            df.loc[index, "形態素4ID"] = self.fe.feature4[row["形態素4"]]
            df.loc[index, "形態素5ID"] = self.fe.feature5[row["形態素5"]]

            # その他特徴量
            df.loc[index, "n単語目"] = index
            if row["is主辞"]:
                df.loc[index, "is主辞"] = 1
            else:
                df.loc[index, "is主辞"] = 0
            if row["is機能語"]:
                df.loc[index, "is機能語"] = 1
            else:
                df.loc[index, "is機能語"] = 0
            df.loc[index, "係り先文節"] = extraction_num(row["係り先文節"])
        for index, row in df.iterrows():
            if row["verb_type"] == "noun" or row["verb_type"] == "pred":
                y = row.loc[
                    ["ga", "ga_type", "o", "o_type", "ni", "ni_type", "verb_type"]
                ].copy()
                cases = ["ga", "o", "ni"]
                for case in cases:
                    case_types = y[f"{case}_type"].split(",")
                    case_ids = y[f"{case}"].split(",")
                    case_indices = []
                    for case_type, case_id in zip(case_types, case_ids):
                        case_index = self._case_id_to_index(
                            df, case_id, case_type, True
                        )
                        case_indices.append(case_index)
                    case_indices = ",".join(case_indices)
                    y[case] = case_indices
                x = df.drop(
                    labels=[
                        "形態素0",
                        "形態素1",
                        "形態素2",
                        "形態素3",
                        "形態素4",
                        "形態素5",
                        "id",
                        "ga",
                        "ga_type",
                        "o",
                        "o_type",
                        "ni",
                        "ni_type",
                        "verb_type",
                        "n文目",
                        "is文末",
                    ],
                    axis=1,
                ).copy()
                x["is_target_verb"] = 0
                x.loc[index, "is_target_verb"] = 1
                x["述語からの距離"] = x.index - index
                yield x, y


class DataLoader:
    def __init__(self, data_dict, batch_size, word2index, shuffle=True):
        self.dataset = np.concatenate([data_dict[key] for key in data_dict.keys()])
        self.max_length = 50
        self.batch_size = batch_size
        self.word2index = word2index
        self.index2media = ["OC", "OY", "OW", "PB", "PM", "PN"]
        self.media2index = {key: i for i, key in enumerate(self.index2media)}
        self.shuffle = shuffle
        self.fe = FeatureToEmbedID()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        dataX = self.dataset[item][:, 0]
        dataY = self.dataset[item][:, 1]
        dataZ = self.dataset[item][:, 2]

        X = self.trans_X(dataX)
        Y = self.trans_Y(dataY)
        Z = self.trans_Z(dataZ)

        return X, Y, Z

    def __iter__(self):
        if self.shuffle:
            index = np.random.permutation(len(self))
        else:
            index = np.arange(len(self))

        return BackgroundGenerator(
            (
                self[index[i : i + self.batch_size]]
                for i in range(0, len(self), self.batch_size)
            )
        )

    def sentence2index(self, sentence):
        index_list = []
        for word in sentence:
            if self.word2index.get(word):
                index = self.word2index[word]
            else:
                index = self.word2index["<UNK>"]
            index_list.append(index)
        if len(index_list) > self.max_length:
            index_list = index_list[: self.max_length]
        pad_index = self.word2index["<PAD>"]
        for _ in range(self.max_length - len(index_list)):
            index_list.append(pad_index)
        return index_list

    def feature2index(self, _feature):
        pad_index = 0
        feature = _feature.values.tolist()
        if len(feature) > self.max_length:
            feature = feature[: self.max_length]
        for _ in range(self.max_length - len(feature)):
            feature.append(pad_index)
        return feature

    def trans_X(self, dataX):
        sentences = [data["単語"] for data in dataX]
        X_words = [self.sentence2index(sentence) for sentence in sentences]
        X_features = []
        for key in range(6):
            features = [data[f"形態素{key}ID"] for data in dataX]
            X_feature = [self.feature2index(feature) for feature in features]
            X_features.append(X_feature)
        return torch.tensor(X_words).cuda(), torch.tensor(X_features).cuda()

    def trans_Y(self, dataY):
        y_intradep = np.zeros((len(dataY), self.max_length - 1, 3))
        y_intrazero = np.zeros((len(dataY), self.max_length - 1, 3))
        y_exophora_ga = np.zeros((len(dataY), 4))
        y_exophora_o = np.zeros((len(dataY), 4))
        y_exophora_ni = np.zeros((len(dataY), 4))

        for i, data in enumerate(dataY):
            ga_index = data["ga"]
            if data["ga_type"] == "intra(dep)":
                y_intradep[i][ga_index][0] = 1
            elif data["ga_type"] == "intra(zero)":
                y_intrazero[i][ga_index][0] = 1
            else:
                if data["ga_type"] == "exo1":
                    y_exophora_ga[1] = 1
                elif data["ga_type"] == "exo2":
                    y_exophora_ga[2] = 1
                elif data["ga_type"] == "exoX":
                    y_exophora_ga[3] = 1
                else:
                    y_exophora_ga[0] = 1

            o_index = data["o"]
            if data["o_type"] == "intra(dep)":
                y_intradep[i][o_index][0] = 1
            elif data["o_type"] == "intra(zero)":
                y_intrazero[i][o_index][0] = 1
            else:
                if data["o_type"] == "exo1":
                    y_exophora_o[1] = 1
                elif data["o_type"] == "exo2":
                    y_exophora_o[2] = 1
                elif data["o_type"] == "exoX":
                    y_exophora_o[3] = 1
                else:
                    y_exophora_o[0] = 1

            ni_index = data["ni"]
            if data["ni_type"] == "intra(dep)":
                y_intradep[i][ni_index][0] = 1
            elif data["ni_type"] == "intra(zero)":
                y_intrazero[i][ni_index][0] = 1
            else:
                if data["ni_type"] == "exo1":
                    y_exophora_ni[1] = 1
                elif data["ni_type"] == "exo2":
                    y_exophora_ni[2] = 1
                elif data["ni_type"] == "exoX":
                    y_exophora_ni[3] = 1
                else:
                    y_exophora_ni[0] = 1

        return (
            torch.tensor(y_intradep).cuda(),
            torch.tensor(y_intrazero).cuda(),
            torch.tensor(y_exophora_ga).cuda(),
            torch.tensor(y_exophora_o).cuda(),
            torch.tensor(y_exophora_ni).cuda(),
        )

    def trans_Z(self, dataZ):
        media = [self.get_media(filename) for filename in dataZ]
        Z = np.zeros((len(dataZ), 6))
        for i, key in enumerate(media):
            Z[i][self.media2index[key]] = 1
        return torch.tensor(Z).cuda()

    def get_media(self, filename):
        for key in self.index2media:
            if key in filename:
                return key
        return ""
