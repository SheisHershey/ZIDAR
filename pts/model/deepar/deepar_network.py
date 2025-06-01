from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution

from gluonts.core.component import validated
from gluonts.torch.distributions.distribution_output import DistributionOutput
from pts.model import weighted_average
from pts.modules import MeanScaler, NOPScaler, FeatureEmbedder
from torch.distributions import Categorical,Normal,MixtureSameFamily,TransformedDistribution,AffineTransform #!!
from torch.nn import Transformer,TransformerEncoder,TransformerEncoderLayer,LayerNorm,Linear #!! 增加Transformer网络模块
from .tcn import TemporalConvNet #!! 时序卷积层

def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class DeepARNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_cells: int,
        num_layers2: int,
        num_cells2: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        distr_output2: DistributionOutput, #!! 增加第2个全连接pdf输出层
        distr_output3: DistributionOutput, #!! 增加第3个全连接混合权重输出层
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: List[int],
        lags_seq: List[int],
        scaling: bool = True, # 数据标准化
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.num_layers2 = num_layers2 #!!
        self.num_cells2 = num_cells2 #!!
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling
        self.dtype = dtype

        self.lags_seq = lags_seq

        self.distr_output = distr_output #第1个全连接pdf输出层
        self.distr_output2 = distr_output2 #!! 第2个全连接pdf输出层
        self.distr_output3 = distr_output3  #!! 状态识别模块的多项分布输出层,只用于估计混合概率(状态概率)
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell_type] #RNN网络层
        #!! 第1个非零子分布的RNN
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        #!! 第2个非零子分布的RNN（同一个输入向量，分别传入不同的RNN网络，学习成员分布参数(均值和标准差）
        self.rnn2 = rnn(
            input_size=input_size, #!!
            hidden_size=num_cells,
            num_layers=num_layers, #!!
            dropout=dropout_rate,
            batch_first=True,
        )
        #!! 用于状态识别预测的RNN
        self.rnn3 = rnn(
            input_size=32, #!! TCN层的最后一个channel大小
            hidden_size=num_cells2,# !!
            num_layers=num_layers2, #!!
            dropout=dropout_rate,
            batch_first=True,
        )
        '''
        !! TCN时序卷积层
        '''
        self.TCN = TemporalConvNet( #!! 输入形状是[batchsize, input_size, seq_len]，
            num_inputs=input_size, #!! 输入通道数量=特征数量input_size
            num_channels=[32,32,32] #!! 输出通道数量=卷积核数量(特征图数量)
        )
        '''
        !! 增加第3个Transformer网络结构（学习混合权重（类似于分类概率））
        '''
        #!! 增加第3个Transformer网络结构（状态预测模块，学习混合权重（类似于分类概率））
        # self.transformer = Transformer(
        #     d_model=num_cells * 2,  # 输入大小（C个RNN隐向量的Concat大小)
        #     nhead=8,    # MultiHead的头数 8
        #     num_encoder_layers=6, # Encoder的Block数量 6
        #     num_decoder_layers=6, # Encoder的Block数量 6
        #     dim_feedforward=1024, # FFN层的隐层大小(较大于input_size) 1024
        #     dropout=0.1,
        #     batch_first=True # batch_size第1维，seq_len第2维, feature_dim第3维(input,hidden,output)
        # )
        # !! 增加第3个Transformer网络结构（状态预测模块，学习混合权重（类似于分类概率））
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=num_cells2,
            nhead=8, #8
            dim_feedforward=2048, #512
            dropout=0.1,
            batch_first=True
        )
        self.encoder_norm = LayerNorm(normalized_shape=num_cells2, eps=1e-5)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer,
            num_layers=10, #5
            norm=self.encoder_norm)

        # !! 2个非零子分布输出层的分布形状
        self.target_shape = distr_output.event_shape
        self.target_shape2 = distr_output2.event_shape

        # !! 2个非零子分布输出层的分布形状的分布参数激活函数(rnn_output--->distr_args)
        self.proj_distr_args = distr_output.get_args_proj(num_cells)
        self.proj_distr_args2 = distr_output2.get_args_proj(num_cells)
        # !! 状态识别模块的多项分布输出层(TransformerEncoder输出向量 ---> mix_logits）
        self.proj_distr_args3 = distr_output3.get_args_proj(num_cells2) #!!MixtureDIstribution使用Transformer输出向量作为输入，输出混合权重π
        #self.linear = Linear(num_cells2,  3)

        # 数据特征提取嵌入层embedder和标准化层scaler
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality, embedding_dims=embedding_dimension
        )

        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def unroll_encoder(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
        past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Optional[
            torch.Tensor
        ] = None,  # (batch_size, prediction_length, num_features)
        future_target: Optional[
            torch.Tensor
        ] = None,  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, List], torch.Tensor, Union[torch.Tensor, List], torch.Tensor, Union[torch.Tensor, List], torch.Tensor, torch.Tensor]:

        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat[
                :, self.history_length - self.context_length :, ...
            ]
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:, self.history_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            sequence = torch.cat((past_target, future_target), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )


        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)

        # 数据标准化处理层
        _, scale = self.scaler(
            past_target[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                scale.log() if len(self.target_shape) == 0 else scale.squeeze(1).log(),
            ),
            dim=1,
        )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = lags_scaled.reshape(
            (-1, subsequences_length, len(self.lags_seq) * prod(self.target_shape))
        )

        # (batch_size, sub_seq_len, input_dim)  input_dim并不代表原始输入时间序列的特征数，而是拼接转换得到的特征数
        inputs = torch.cat((input_lags, time_feat, repeated_static_feat), dim=-1) #inputs经过重新拼接而得到
        #print(inputs.shape)


        # unroll encoder
        '''
        !! encoder前向传播(分别用2个RNN结构来提特征)
        '''
        outputs, state = self.rnn(inputs) # 第1个非零子分布RNN前向传播
        outputs2, state2 = self.rnn2(inputs) #!! 第2个非零子分布RNN前向传播
        outputs3, state3 = self.rnn3(inputs) #!! 状态预测模块的RNN前向传播


        # outputs: (batch_size, seq_len, num_cells)
        # state: list of (num_layers, batch_size, num_cells) tensors
        # scale: (batch_size, 1, *target_shape)
        # static_feat: (batch_size, num_features + prod(target_shape))

        return outputs, state, outputs2, state2, outputs3, state3, scale, static_feat #!! 增加第2,3个RNN结构的输出

class DeepARTrainingNetwork(DeepARNetwork):
    def distribution(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> Distribution:
        '''
        (原始)状态预测模块的输入inputs处理模块
        '''
        # 预测阶段
        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat[:, self.history_length - self.context_length:, ...]
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        # 训练阶段
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:, self.history_length - self.context_length:, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            sequence = torch.cat((past_target, future_target), dim=1)
            #print(sequence.shape)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # TimeLag编码
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq, #[1,2,3,4,5,6,7,11,12,13,23,24]
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)

        # 数据标准化处理层
        _, scale = self.scaler(
            past_target[:, -self.context_length:, ...],
            past_observed_values[:, -self.context_length:, ...],
        )

        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                scale.log() if len(self.target_shape) == 0 else scale.squeeze(1).log(),
            ),
            dim=1,
        )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = lags_scaled.reshape(
            (-1, subsequences_length, len(self.lags_seq) * prod(self.target_shape))
        )
        # (batch_size, sub_seq_len, input_dim)  input_dim并不代表原始输入时间序列的特征数，而是拼接转换得到的特征数
        # !! input_lags: target的TimeDelaying编码（t-1,t-2,t-3,t-4,t-5,t-6,t-7,t-11,t-12,t-13,t-23,t-24)
        #     input_lags的第t个特征 对应于 滞后lags_seq[t]个时间步的数据
        # !! time_feat: 时间特征
        # !! repeated_static_feat: 静态特征
        inputs = torch.cat((input_lags, time_feat, repeated_static_feat), dim=-1)  # inputs经过重新拼接而得到
        # print(inputs.shape)

        '''
        !! 状态预测模块
        '''
        TCN_outputs = self.TCN(inputs.transpose(1, 2)) #!! TCN时序卷积层(提取局部特征),输入形状是[bs,input_size,seq_len]，第2,3维与RNN相反; 输出形状是[bs, output_size, seq_len]
        print(TCN_outputs.shape)
        rnn_outputs3, state3 = self.rnn3(TCN_outputs.transpose(1, 2))  # !! 状态预测模块的RNN前向传播
        #transformer_encoder_output = self.transformer_encoder(rnn_outputs3)
        distr_args3 = self.proj_distr_args3(rnn_outputs3)
        mix_logits, _, _ = distr_args3  # 状态概率(混合概率)

        '''
        !! 零值填充模块(训练过程)
        '''
        # detach截断反向传播梯度
        # mix_logits_detach = mix_logits.detach() #!! 零值填充时，需要对mix_logits权重概率进行detach，防止非零预测模块的反向传播影响mix_logits混合权重参数
        # probs_detach = Categorical(logits=mix_logits_detach).probs  ## probs：(batchsize, seq_len, num_components)
                          #!! 零值填充时，需要对probs权重概率进行clone，防止非零预测模块的反向传播影响probs混合权重参数
        mix_logits_clone = mix_logits.clone()
        probs_clone = Categorical(logits=mix_logits_clone).probs
        # 预测阶段
        if future_time_feat is None or future_target is None:
            '''
            对于每个context_length中的时间点，用状态预测模块的混合概率(p1/(p1+p2),p2/(p1+p2)) * batch每个时间点的真实值的非零分位数(0.2, 0.8)，填充原始数据0值
            (或许要确保每个batch时间段一致)
            '''
            lower_loc_quantile = 0.25
            greater_loc_quantile = 0.75
            ## 使用固定分位数
            # nonzero_lower_quantile = 10
            # nonzero_greater_quantile = 100

            #sequence_imput = sequence.detach().clone() #既截断梯度，又进行克隆
            sequence_imput = sequence.clone()
            # 对于每个context_length中的时间点
            context_start_index = self.history_length - self.context_length
            for j in range(context_start_index, sequence.shape[1]):
                # 计算每个时间点j的真实值的非零分位数(0.2, 0.8)
                nonzero_values = torch.Tensor([x for x in sequence[:, j] if x != 0.0])
                # 使用每个时间点j的不同分位数
                nonzero_lower_quantile = torch.quantile(nonzero_values, lower_loc_quantile)
                nonzero_greater_quantile = torch.quantile(nonzero_values, greater_loc_quantile)
                for i in range(sequence.shape[0]):
                    if sequence[i, j] == 0: # 若真实值为0，进行缺失值填充
                        # π1*q1 + π2*q2
                        #sequence_imput[i, j] = probs_clone[i, j - context_start_index, 1] * nonzero_lower_quantile \
                        #                      + probs_clone[i, j - context_start_index, 2] * nonzero_greater_quantile
                        sequence_imput[i, j] ==sequence[i, j]
                        # π1/(π1+π2)*q1 + π2/(π1+π2)*q2
                        # sequence_imput[i, j] = probs[i, j-context_start_index, 1]/(probs[i, j-context_start_index, 1]+probs[i, j-context_start_index, 2]) * nonzero_lower_quantile \
                        #                        + probs[i, j-context_start_index, 2]/(probs[i, j-context_start_index, 1]+probs[i, j-context_start_index, 2]) * nonzero_greater_quantile

            sequence_length = self.history_length
            subsequences_length = self.context_length
        # 训练阶段
        else:
            '''
            对于所有context_length + predict_length中的时间点，
            用状态预测模块的混合概率p1,p2 * batch每个时间点的真实值的非零分位数(0.2, 0.8)，填充原始数据0值
            (或许要确保每个batch时间段一致)
            '''
            ## 使用每个时间点的不同分位数
            lower_loc_quantile = 0.25
            greater_loc_quantile = 0.75
            ## 使用固定分位数
            # nonzero_lower_quantile = 10
            # nonzero_greater_quantile = 100
            #sequence_imput = sequence.detach().clone() #既截断梯度，又进行克隆
            sequence_imput = sequence.clone()
            # 填充所有context_length + predict_length中的时间点(每个时间点t, 真实分位数qt填充xt)
            context_start_index = self.history_length - self.context_length
            for j in range(context_start_index, sequence.shape[1]): #填充所有context_length + predict_length中的时间点(每个时间点t, 真实分位数qt填充xt)
                # 计算每个时间点j的真实值的非零分位数(0.2, 0.8)
                nonzero_values = torch.Tensor([x for x in sequence[:, j] if x != 0.0])
                # 使用每个时间点j的不同分位数
                nonzero_lower_quantile = torch.quantile(nonzero_values, lower_loc_quantile)
                nonzero_greater_quantile = torch.quantile(nonzero_values, greater_loc_quantile)
                for i in range(sequence.shape[0]):
                    if sequence[i, j] == 0: # 若真实值为0，则进行缺失值填充
                        # π1*q1 + π2*q2
                        #sequence_imput[i, j] = probs_clone[i, j - context_start_index, 1] * nonzero_lower_quantile \
                        #                       + probs_clone[i, j - context_start_index, 2] * nonzero_greater_quantile
                        sequence_imput[i, j] == sequence[i, j]
                        # π1/(π1+π2)*q1 + π2/(π1+π2)*q2
                        # sequence_imput[i, j] = (probs[i, j-context_start_index, 1]/(probs[i, j-context_start_index, 1]+probs[i, j-context_start_index, 2])) * nonzero_lower_quantile \
                        #                        + (probs[i, j-context_start_index, 2]/(probs[i, j-context_start_index, 1]+probs[i, j-context_start_index, 2])) * nonzero_greater_quantile


            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length
            # print(sequence2.shape)
            # print(sequence_length)
            # print(subsequences_length)

        lags2 = self.get_lagged_subsequences(
            sequence=sequence_imput,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)

        # 数据标准化处理层(!! 是否应该改为sequence_imput的scale，而不是原始past_target的scale）
        # _, scale = self.scaler(
        #     past_target[:, -self.context_length:, ...],
        #     past_observed_values[:, -self.context_length:, ...],
        # )
        _, scale2 = self.scaler( #只对context范围数据计算scale
            sequence_imput[:, self.history_length - self.context_length : -self.prediction_length, ...],
            past_observed_values[:, -self.context_length:, ...],
        )


        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                scale.log() if len(self.target_shape) == 0 else scale.squeeze(1).log(),
            ),
            dim=1,
        )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled2 = lags2 / scale2.unsqueeze(-1) #!! 是否应该改为sequence_imput的scale，而不是原始past_target的scale）

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags2 = lags_scaled2.reshape(
            (-1, subsequences_length, len(self.lags_seq) * prod(self.target_shape))
        )
        # (batch_size, sub_seq_len, input_dim)  input_dim并不代表原始输入时间序列的特征数，而是拼接转换得到的特征数
        # !! input_lags: 时间点t的target的TimeDelaying编码 [t-1, t-2, t-3, t-4, t-5, t-6, t-7, t-11, t-12, t-13, t-23, t-24]
        # !! time_feat: 时间特征
        # !! repeated_static_feat: 静态特征
        inputs2 = torch.cat((input_lags2, time_feat, repeated_static_feat), dim=-1)  # inputs经过重新拼接而得到


        '''
        非零需求预测模块 (使用状态预测填充0之后的输入)
        '''
        rnn_outputs, state = self.rnn(inputs2) # 第1个非零子分布RNN前向传播
        rnn_outputs2, state2 = self.rnn2(inputs2) #!! 第2个非零子分布RNN前向传播
            # rnn_outputs---Tensor(batchsize, context_l+predict_l, num_cells)
        distr_args = self.proj_distr_args(rnn_outputs) #第1个全连接pdf层前向传播 输出子分布参数
        distr_args2 = self.proj_distr_args2(rnn_outputs2) #!! 第2个全连接pdf层前向传播，输出子分布参数


        '''
        构建混合分布----状态预测模块的mix_logits + 非零需求预测模块的loc,scale
        '''
        loc1, distr_scale1 = distr_args # 较小非零分布
        loc2, distr_scale2 = distr_args2# 较大非零分布
        mix_logits, _, _ = distr_args3 #状态概率(混合概率)

        loc0 = torch.zeros_like(loc1) # 零分布均值固定为0
        distr_scale0 = torch.zeros_like(distr_scale1) + 0.01 # 零分布标准差固定为0.01

        loc = torch.stack((loc0, loc1, loc2), dim=-1)
        distr_scale = torch.stack((distr_scale0, distr_scale1, distr_scale2), dim=-1)
        distr_args_mix = mix_logits, loc, distr_scale

        '''
        !! 根据distr_args_mix，得到混合分布distr（避免使用混合分布全连接层distr_output3的接口)
        '''
        distr = MixtureSameFamily( Categorical(logits=mix_logits), Normal(loc, distr_scale))
        distr = TransformedDistribution(distr, [AffineTransform(loc=0, scale=scale)])

        #!! 根据第3个混合分布全连接层的distr_args3，返回最终的混合分布distr
        ## scale是一个bool参数，表示是否进行数据标准化，需要根据scale来将数据分布还原回标准化之前的分布
        return distr,\
               distr_args_mix #!! 返回值加上全连接层输出的分布参数，将均值加入到loss函数中，使得分布均值更大



    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        '''
        !! 返回混合分布全连接层得到的 混合分布distr
        !! 同时返回混合分布的分布参数distr_args
        '''
        distr, distr_args = self.distribution(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
        )


        # put together target sequence
        # target_shape: (batch_size, seq_len)  , seq_len = context_length + prediction_length
        # (batch_size, seq_len, *target_shape)
        # 输入的时间序列窗口数据target(batchsize,  context_length+prediction_length)
        target = torch.cat(
            (
                past_target[:, self.history_length - self.context_length :, ...],
                future_target,
            ),
            dim=1,
        )
        mix_logits, loc, dist_scale = distr_args # 2or3维元组tuple（mix_logits, loc, dist_scale)
                                                # mix_logits, loc, dist_scale的形状都是：(batchsize, seq_len, num_components)
        probs = Categorical(logits=mix_logits).probs  ## probs：(batchsize, seq_len, num_components)
        #print(loc)
        #print(target[:, -1])

        '''
        !! 损失函数loss中，加入分布均值loss项
        '''
        greater_loc_weight = 0.1  # !! loss函数中较大非零分布均值项的权重
        lower_loc_weight = 0.1 # !! loss函数中较小非零分布均值项的权重
        mix_prob_weight = 0.5  # !! loss函数中混合权重交叉熵的权重

        greater_loc_ratio = 1  # !! 最后一个预测时间点的target较大非零分布均值项的倍数
        lower_loc_ratio = 0.5  # !! 最后一个预测时间点的target较小非零分布均值项的倍数

        lower_loc_quantile = 0.25  # !! 最后一个预测时间点的target较小零分布均值项的非0值分位数
        greater_loc_quantile = 0.75  # !! 最后一个预测时间点的target较大非零分布均值项的非0值分位数

        '''
        原始的负对数似然loss-----单个batch所有样本i在所有时间点t上的负对数似然的向量
        loss：(batch_size, seq_len)  
        '''
        #loss = -distr.log_prob(target)  # 原始los
        loss = -distr.indicator_log_prob_2(target) # 负对数似然(真实值为0, 对数似然改为log(π1×1))

        # (batch_size, seq_len, *target_shape)
        observed_values = torch.cat(
            (
                past_observed_values[
                :, self.history_length - self.context_length:, ...
                ],
                future_observed_values,
            ),
            dim=1,
        )

        # mask the loss at one time step iff one or more observations is missing in the target dimensions
        # loss_weights: (batch_size, seq_len)
        loss_weights = (
            observed_values
            if (len(self.target_shape) == 0)
            else observed_values.min(dim=-1, keepdim=False)
        )

        # batch所有样本i在所有时间点t上的负对数似然的加权和，作为最终的负对数似然loss
        weighted_loss = weighted_average(loss, weights=loss_weights)
        neg_log_likelihood = weighted_loss
        #print('负对数似然loss:', weighted_loss)

        '''
        所有时间点上的较大非零分布和较小非零分布均值loss项
        '''
        greatermean_loss_sum = 0.0 #较大非零分布均值求和项
        lower_loss_sum = 0.0 #较小非零分布均值求和项
        mixprob_loss_sum = 0.0
        seq_len = target.shape[1]
        for t in range(target.shape[1]):
            target_y_nonzero = torch.Tensor([x for x in target[:, t] if x != 0.0])  # 第t个时间点的target的非0值的均值
            if len(target_y_nonzero) == 0:
                target_y_nonzero_mean = 0.0
            else:
                target_y_nonzero_mean = torch.mean(target_y_nonzero)

            ##!! 时间点t的target非0值的分位数
            target_y_q25 = torch.quantile(target_y_nonzero, lower_loc_quantile)
            target_y_q75 = torch.quantile(target_y_nonzero, greater_loc_quantile)

            '''
            均值参数的L2正则化项
            '''
            greatermean_loss = greater_loc_weight * torch.mean(
                 torch.square(loc[:, t, 2].squeeze() - target_y_q75))
            lowermean_loss = lower_loc_weight * torch.mean(
                 torch.square(loc[:, t, 1].squeeze() - target_y_q25))

            # greatermean_loss和lowermean_loss，都加在weighted_loss上
            #weighted_loss = weighted_loss + greatermean_loss / seq_len + lowermean_loss / seq_len   # 对所有时间t取平均(sum / T)

            '''
            混合权重交叉熵惩罚项（2分类）
            !! 第t个时间点
            '''
            #根据最后一个预测时间点的target，定义0-1序列
            target_01 = torch.zeros((target.shape[0],)) # target_01: (batchsize, )
            for i in range(target.shape[0]):
                if target[i][t] == 0.0:
                    target_01[i] = 0
                elif target[i][t] > 0.0:
                    target_01[i] = 1
            # print(target[:, -1])
            # print(target_01)
            # print(probs[:, -1, 1])

            criterion = nn.BCELoss() #2分类交叉熵
            #criterion = nn.CrossEntropyLoss() #多分类损失

            # print(probs[:, -1, 1].shape)
            # print(target_01.shape)

            #mixprob_loss = mix_prob_weight * criterion(probs[:, t, 1], target_01) #混合权重的2分类损失惩罚项
            mixprob_loss = mix_prob_weight * criterion(probs[:, t, 1]+probs[:, t, 2], target_01) #混合权重的多分类损失惩罚项(真实标签改为Long类型)
            #print(weighted_loss, greatermean_loss, mixprob_loss)

            greatermean_loss_sum += greatermean_loss
            lower_loss_sum += lowermean_loss
            mixprob_loss_sum += mixprob_loss

        print(weighted_loss, greatermean_loss_sum / seq_len, lower_loss_sum / seq_len,  mixprob_loss_sum / seq_len)
        weighted_loss = weighted_loss + greatermean_loss_sum / seq_len + lower_loss_sum / seq_len + mixprob_loss_sum / seq_len  # 对所有时间t取平均(sum / T)
        return weighted_loss, loss, neg_log_likelihood


class DeepARPredictionNetwork(DeepARNetwork):
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None: #设定sample 100次
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one, at the first time-step
        # of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder( #利用decoder网络，基于全连接层得到的概率分布，抽样生成预测样本samples
        self,
        static_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_target_imput: torch.Tensor, #!! 对于context_length数据的零填充数据
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        scale2: torch.Tensor, #!! 零值填充后的scale
        begin_states: Union[torch.Tensor, List[torch.Tensor]],
        begin_states2: Union[torch.Tensor, List[torch.Tensor]],
        begin_states3: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length, num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch. Shape: (batch_size, 1, 1).
        begin_states : List or Tensor
            list of initial states for the LSTM layers or tensor for GRU.
            the shape of each tensor of the list should be (num_layers, batch_size, num_cells)
        Returns
        --------
        Tensor
            A tensor containing sampled paths.
            Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size * self.num_parallel_samples for increasing parallelism
        ## (batch_size * self.num_parallel_samples,  seq_len,  target_shape)
        repeated_past_target = past_target.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_past_target_imput = past_target_imput.repeat_interleave(  #!! 对于context_length数据的零填充数据
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_time_feat = time_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        ).unsqueeze(1)
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
        repeated_scale2 = scale2.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        if self.cell_type == "LSTM":
            repeated_states = [
                s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
                for s in begin_states
            ]
        else:
            repeated_states = begin_states.repeat_interleave(
                repeats=self.num_parallel_samples, dim=1
            )
        #!! 增加第2个LSTM的repeated_states，repeated_states表示decoder的输入初始状态
        if self.cell_type == "LSTM":
            repeated_states2 = [
                s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
                for s in begin_states2
            ]
        else:
            repeated_states2 = begin_states2.repeat_interleave(
                repeats=self.num_parallel_samples, dim=1
            )
        #!! 增加第3个LSTM的repeated_states，repeated_states表示decoder的输入初始状态
        if self.cell_type == "LSTM":
            repeated_states3 = [
                s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
                for s in begin_states3
            ]
        else:
            repeated_states3 = begin_states3.repeat_interleave(
                repeats=self.num_parallel_samples, dim=1
            )

        future_samples = [] # 保存所有predict_length上的预测samples结果
        future_distr_args_mix = [] # 保存所有predict_length上的分布参数结果

        # for each future time-units we draw new samples for this time-unit and update the state
        ## 多步滚动预测
        for k in range(self.prediction_length):

            '''
            !! (原始)状态预测模块的输入inputs处理模块
            '''
            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = lags / repeated_scale.unsqueeze(-1)

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = lags_scaled.reshape(
                (-1, 1, prod(self.target_shape) * len(self.lags_seq))
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            decoder_input = torch.cat(
                (input_lags, repeated_time_feat[:, k : k + 1, :], repeated_static_feat),
                dim=-1,
            )
            # print(decoder_input.shape)

            '''
            !! 测试过程的非零需求预测模块零填充
               context(encoder)时间点---用每个context自身时间点的真实分位数qt来填充xt
               predict(decoder)时间点---用k阶自回归填充（context时间点用自身时间点的真实分位数qt，predict时间点用预测值分位数）
            '''
            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags2 = self.get_lagged_subsequences(
                sequence=repeated_past_target_imput,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled2 = lags2 / repeated_scale2.unsqueeze(-1) #!! 零值填充的标准化用scale2

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags2 = lags_scaled2.reshape(
                (-1, 1, prod(self.target_shape) * len(self.lags_seq))
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags + num_time_features + num_static_features)
            decoder_input2 = torch.cat(
                (input_lags2, repeated_time_feat[:, k : k + 1, :], repeated_static_feat),
                dim=-1,
            )

            '''
              状态预测模块
              ## decoder_input表示encoder输出的编码向量，作为decoder网络的输入向量；而rnn隐向量是encoder和decoder共享的同一个rnn结构
            '''
            TCN_outputs = self.TCN(decoder_input.transpose(1, 2))  # !! TCN时序卷积层(提取局部特征)
            rnn_outputs3, repeated_states3 = self.rnn3(TCN_outputs.transpose(1, 2), repeated_states3)
            #transformer_encoder_output = self.transformer_encoder(rnn_outputs3)  # TransformerEncoder的输入与输出维度相同，因为基于自注意力机制
            distr_args3 = self.proj_distr_args3(rnn_outputs3)
            mix_logits, _, _ = distr_args3  # 状态概率(混合概率)
            # rnn_outputs3, repeated_states3 = self.rnn3(decoder_input, repeated_states3)

            '''
            !! 非零需求预测模块
                (rnn_output --> 成员分布参数loc和distr_scale)
            '''
            # rnn_outputs---Tensor(batchsize, context_l+predict_l, num_cells)
            rnn_outputs, repeated_states = self.rnn(decoder_input2, repeated_states)  # repeated_states表示decoder的输入初始状态
            rnn_outputs2, repeated_states2 = self.rnn2(decoder_input2, repeated_states2)
            distr_args = self.proj_distr_args(rnn_outputs)  # 第1个全连接pdf层前向传播 输出子分布参数
            distr_args2 = self.proj_distr_args2(rnn_outputs2)  # !! 第2个全连接pdf层前向传播，输出子分布参数

            '''
            !! 1个全连接层，学习混合权重π（类似于分类概率）
              (transformer_output --> 混合权重π)
            '''
            loc1, distr_scale1 = distr_args  # 较小非零分布
            loc2, distr_scale2 = distr_args2  # 较大非零分布
            mix_logits, _, _ = distr_args3  # 状态概率(混合概率)

            '''
            !! 混合分布参数distr_args_mix = 子分布的分布参数loc1, loc2,distr_scale1, distr_scale2  + 混合分布权重mix_logits
            '''
            loc0 = torch.zeros_like(loc1)  # 零分布均值固定为0
            distr_scale0 = torch.zeros_like(distr_scale1) + 0.01  # 零分布标准差固定为0.01

            loc = torch.stack((loc0, loc1, loc2), dim=-1)
            distr_scale = torch.stack((distr_scale0, distr_scale1, distr_scale2), dim=-1)
            distr_args_mix = mix_logits, loc, distr_scale

            '''
            !! 根据distr_args_mix，得到混合分布distr（避免使用混合分布全连接层distr_output3的接口)
            '''
            distr = MixtureSameFamily(Categorical(logits=mix_logits), Normal(loc, distr_scale))
            distr = TransformedDistribution(distr, [AffineTransform(loc=0, scale=repeated_scale)]) #!! 改成repeated_scale


            # (batch_size * num_samples, 1, *target_shape)
            # 根据概率分布进行抽样new_samples
            new_samples = distr.sample() #(bs * 100,  1) 滚动预测，每次只预测1个时间点


            '''
            !! 测试过程的非零需求预测模块零填充
               predict(decoder)时间点---用k阶平均非零分位数填充（前k个时间点的平均非零分位数，context时间点用真实值分位数，predict时间点用预测值分位数）
                   (或许要确保每个batch时间段一致)
            '''
            mix_logits_clone = mix_logits.clone()  # !! 零值填充时，需要对mix_logits权重概率进行clone，防止非零预测模块的反向传播影响mix_logits混合权重参数
            probs_clone = Categorical(logits=mix_logits_clone).probs  ## probs：(batchsize, seq_len, num_components)

            # 使用不同分位数
            lower_loc_quantile = 0.25
            greater_loc_quantile = 0.75
            past_k = 6
            ## 使用固定分位数
            # nonzero_lower_quantile = 10
            # nonzero_greater_quantile = 100

            new_samples_imput = new_samples.clone()
            # 对于当前滚动预测的时间点t(decoder部分)， 用k阶平均非零分位数来填充(k=6)
            nonzero_lower_quantile = 0.0
            nonzero_greater_quantile = 0.0
            repeated_past_target_lastk = repeated_past_target[:, -past_k:]
            for t in range(repeated_past_target_lastk.shape[1]):
                nonzero_values = torch.Tensor([x for x in repeated_past_target_lastk[:, t].reshape(-1) if x != 0.0])
                nonzero_lower_quantile += torch.quantile(nonzero_values, lower_loc_quantile)
                nonzero_greater_quantile += torch.quantile(nonzero_values, greater_loc_quantile)
            nonzero_lower_quantile = nonzero_lower_quantile / past_k
            nonzero_greater_quantile = nonzero_greater_quantile / past_k
            print('较低分位数%f:'%(nonzero_lower_quantile))
            print('较高分位数%f:' % (nonzero_greater_quantile))

            for i in range(new_samples.shape[0]):
                if new_samples[i, 0] <= 0:  # 若预测值<=0，进行缺失值填充
                        # π1*q1 + π2*q2
                        #new_samples_imput[i, 0] = probs_clone[i, 0, 1] * nonzero_lower_quantile \
                        #                       + probs_clone[i, 0, 2] * nonzero_greater_quantile
                        new_samples_imput[i, 0] == new_samples[i, 0]
                        # π1/(π1+π2)*q1 + π2/(π1+π2)*q2
                        # sequence_imput[i, j] = (probs[i, j-context_start_index, 1] / (probs[i, j-context_start_index, 1] + probs[i, j-context_start_index, 2])) * nonzero_lower_quantile \
                        #                        + (probs[i, j-context_start_index, 2] / (probs[i, j-context_start_index, 1] + probs[i, j-context_start_index, 2])) * nonzero_greater_quantile

            # (batch_size * num_samples, seq_len, *target_shape)
            # 滚动预测，将预测时间点t的预测值new_samples和预测填充值new_samples_imput，作为下一个预测时间点t+1的输入
            #         加入repeated_past_target和repeated_past_target_imput
            repeated_past_target = torch.cat((repeated_past_target, new_samples), dim=1)
            repeated_past_target_imput = torch.cat((repeated_past_target_imput, new_samples_imput), dim=1)

            future_samples.append(new_samples)
            future_distr_args_mix.append(distr_args_mix)
        # (batch_size * num_samples, prediction_length, *target_shape)

        samples = torch.cat(future_samples, dim=1) #(batchsize*100, 1)
        # print('samples:',samples.shape)

        # (batch_size, num_samples, prediction_length, *target_shape)
        return samples.reshape( #所有predict_length时间上的预测samples（batch_size, 抽样数num_samples, 预测时间长度prediction_length)
            (
                (-1, self.num_parallel_samples)
                + (self.prediction_length,)
                + self.target_shape
            )
        ),distr_args_mix  # !! 返回预测样本samples和混合分布参数distr_args3

    # noinspection PyMethodOverriding,PyPep8Naming
    def forward(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
        past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: torch.Tensor,  # (batch_size, prediction_length, num_features)
    ) -> torch.Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns
        -------
        Tensor
            Predicted samples
        """

        # unroll the decoder in "prediction mode", i.e. with past data only
        '''
        !! encoder 得到3个RNN层的隐向量
        '''
        # _, state, _2, state2, _3, state3, scale, static_feat = self.unroll_encoder( #RNN前向传播 输出隐层向量h
        #     feat_static_cat=feat_static_cat,
        #     feat_static_real=feat_static_real,
        #     past_time_feat=past_time_feat,
        #     past_target=past_target,
        #     past_observed_values=past_observed_values,
        #     future_time_feat=None,
        #     future_target=None,
        # )

        '''
        (原始)状态预测模块的输入inputs处理模块
        '''
        # 预测阶段
        time_feat = past_time_feat[:, self.history_length - self.context_length:, ...]
        sequence = past_target
        sequence_length = self.history_length
        subsequences_length = self.context_length

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)

        # 数据标准化处理层
        _, scale = self.scaler(
            past_target[:, -self.context_length:, ...],
            past_observed_values[:, -self.context_length:, ...],
        )

        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                scale.log() if len(self.target_shape) == 0 else scale.squeeze(1).log(),
            ),
            dim=1,
        )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = lags_scaled.reshape(
            (-1, subsequences_length, len(self.lags_seq) * prod(self.target_shape))
        )
        # (batch_size, sub_seq_len, input_dim)  input_dim并不代表原始输入时间序列的特征数，而是拼接转换得到的特征数
        # !! input_lags: target的TimeDelaying编码
        # !! time_feat: 时间特征
        # !! repeated_static_feat: 静态特征
        inputs = torch.cat((input_lags, time_feat, repeated_static_feat), dim=-1)  # inputs经过重新拼接而得到
        # print(inputs.shape)

        '''
        !! 状态预测模块
        '''
        TCN_outputs = self.TCN(inputs.transpose(1, 2)) #!! TCN时序卷积层(提取局部特征),输入形状是[bs,input_size,seq_len]，第2,3维与RNN相反
        rnn_outputs3, state3 = self.rnn3(TCN_outputs.transpose(1, 2))  # !! 状态预测模块的RNN前向传播
        #transformer_encoder_output = self.transformer_encoder(rnn_outputs3)
        distr_args3 = self.proj_distr_args3(rnn_outputs3)
        mix_logits, _, _ = distr_args3  # 状态概率(混合概率)
        probs = Categorical(logits=mix_logits).probs  ## probs：(batchsize, seq_len, num_components)

        '''
        !! 状态预测模块对context_length数据进行填充0，作为非零需求预测模块的输入inputs
        '''
        # 预测阶段
        time_feat = past_time_feat[:, self.history_length - self.context_length:, ...]

        '''
        !! 零值填充模块(预测过程)
           context(encoder)时间点---用每个context自身时间点的真实分位数qt来填充xt
           predict(decoder)时间点---用k阶自回归填充（context时间点用自身时间点的真实分位数qt，predict时间点用预测值分位数）
        '''
        # 使用每个时间点的不同分位数
        lower_loc_quantile = 0.25
        greater_loc_quantile = 0.75
        ## 使用固定分位数
        # nonzero_lower_quantile = 10
        # nonzero_greater_quantile = 100

        sequence_imput = sequence.clone()
        # 对于每个context_length中的时间点(encoder部分)
        context_start_index = self.history_length - self.context_length
        for j in range(context_start_index, sequence.shape[1]):
            # 每个时间点j的真实值的非零分位数(0.2, 0.8)
            nonzero_values = torch.Tensor([x for x in sequence[:, j] if x != 0.0])
            # 使用每个时间点j的不同分位数
            nonzero_lower_quantile = torch.quantile(nonzero_values, lower_loc_quantile)
            nonzero_greater_quantile = torch.quantile(nonzero_values, greater_loc_quantile)
            for i in range(sequence.shape[0]):
                if sequence[i, j] == 0:  # 若context_length中的真实值为0，进行缺失值填充
                    # π1*q1 + π2*q2
                    #sequence_imput[i, j] = probs[i, j-context_start_index, 1] * nonzero_lower_quantile \
                    #                       + probs[i, j-context_start_index, 2] * nonzero_greater_quantile
                    sequence_imput[i, j] == sequence[i, j]
                    # π1/(π1+π2)*q1 + π2/(π1+π2)*q2
                    # sequence_imput[i, j] = (probs[i, j-context_start_index, 1] / (probs[i, j-context_start_index, 1] + probs[i, j-context_start_index, 2])) * nonzero_lower_quantile \
                    #                        + (probs[i, j-context_start_index, 2] / (probs[i, j-context_start_index, 1] + probs[i, j-context_start_index, 2])) * nonzero_greater_quantile

        sequence_length = self.history_length
        subsequences_length = self.context_length

        lags2 = self.get_lagged_subsequences(
            sequence=sequence_imput,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)

        # 数据标准化处理层(!! 是否应该改为sequence_imput的scale，而不是原始past_target的scale）
        # _, scale = self.scaler(
        #     past_target[:, -self.context_length:, ...],
        #     past_observed_values[:, -self.context_length:, ...],
        # )
        _, scale2 = self.scaler( #只对context范围数据计算scale
            sequence_imput[:, -self.context_length:, ...],
            past_observed_values[:, -self.context_length:, ...],
        )


        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                scale.log() if len(self.target_shape) == 0 else scale.squeeze(1).log(),
            ),
            dim=1,
        )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled2 = lags2 / scale2.unsqueeze(-1)  # !! 是否应该改为sequence_imput的scale，而不是原始past_target的scale）

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags2 = lags_scaled2.reshape(
            (-1, subsequences_length, len(self.lags_seq) * prod(self.target_shape))
        )
        # (batch_size, sub_seq_len, input_dim)  input_dim并不代表原始输入时间序列的特征数，而是拼接转换得到的特征数
        # !! input_lags: 时间点t的target的TimeDelaying编码 [t-1, t-2, t-3, t-4, t-5, t-6, t-7, t-11, t-12, t-13, t-23, t-24]
        # !! time_feat: 时间特征
        # !! repeated_static_feat: 静态特征
        inputs2 = torch.cat((input_lags2, time_feat, repeated_static_feat), dim=-1)  # inputs经过重新拼接而得到

        '''
        非零需求预测模块 (使用状态预测填充0之后的输入)
        '''
        rnn_outputs, state = self.rnn(inputs2)  # 第1个非零子分布RNN前向传播
        rnn_outputs2, state2 = self.rnn2(inputs2)  # !! 第2个非零子分布RNN前向传播

        '''
        !! decoder 2个RNN结构各自的前向传播，再通过全连接层得到混合分布，抽样生成预测样本
           ** 根据encoder的rnn隐状态向量state，编码得到decoder_input，作为decoder网络的输入向量
        '''
        return self.sampling_decoder( #decoder网络前向传播，基于全连接层得到的概率分布，抽样生成预测样本samples
            past_target=past_target,
            past_target_imput=sequence_imput,#!! 对于context_length数据的零填充数据
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            scale2=scale2,
            begin_states= state,
            begin_states2 = state2,  #!! 增加第2个RNN结构的隐状态向量
            begin_states3 = state3  #!! 状态预测模块的RNN结果的隐状态向量
        )
