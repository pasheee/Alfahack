import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.residual = (in_features == out_features)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        if self.residual:
            out = out + x
        return out

class TabNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, n_steps=3):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(n_steps)])
        self.attentions = nn.ModuleList([nn.MultiheadAttention(output_dim, 8) for _ in range(n_steps)])
    
    def forward(self, x):
        outputs = []
        for fc, attention in zip(self.fc, self.attentions):
            out = fc(x)
            out, _ = attention(out.unsqueeze(0), out.unsqueeze(0), out.unsqueeze(0))
            outputs.append(out.squeeze(0))
        return torch.cat(outputs, dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout(src2)
        return src


class CreditScoringModel(nn.Module):
    def __init__(self, input_dim, num_classes=1, droprob = 0.3):
        super().__init__()
        
        self.dense_branch = nn.Sequential(
            DenseBlock(input_dim, 1024, dropout = droprob),
            DenseBlock(1024, 512, dropout = droprob),
            DenseBlock(512, 256, dropout = droprob)
        )
        
        self.tabnet = TabNetBlock(input_dim, 64)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True, 
            dropout = droprob
        )
        
        self.transformer = TransformerBlock(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout = droprob
        )
        
        self.attention = nn.MultiheadAttention(256, 8)
        
        self.projection = nn.Linear(960, 256)

        self.attention_projection = nn.Linear(256, 256)
        self.final_dense = nn.Sequential(
            nn.Linear(960, 512),  
            nn.ReLU(),
            nn.Dropout(droprob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(droprob),
            nn.Linear(256, num_classes)
        )
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(192)  
        self.bn3 = nn.BatchNorm1d(512) 
        
    def forward(self, x):
        dense_out = self.dense_branch(x)
        dense_out = self.bn1(dense_out)
        
        tab_out = self.tabnet(x)
        tab_out = self.bn2(tab_out)
        
        lstm_out, (hidden, cell) = self.lstm(x.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)
        lstm_out = self.bn3(lstm_out)
        
        trans_out = self.transformer(dense_out.unsqueeze(0))
        trans_out = trans_out.squeeze(0)

        concat_features = torch.cat([tab_out, lstm_out, trans_out], dim=-1)
        concat_features = self.projection(concat_features)
        
        attn_out, _ = self.attention(
            dense_out.unsqueeze(0),
            concat_features.unsqueeze(0),
            concat_features.unsqueeze(0)
        )
        attn_out = attn_out.squeeze(0)
        attn_out = self.attention_projection(attn_out)
        
        combined = torch.cat([
            dense_out,
            tab_out[:, :256],
            lstm_out[:, :256],
            attn_out
        ], dim=1)
        
        output = self.final_dense(combined)
        return output


    
class SimplerCreditScoringModel_v1(nn.Module):
    def __init__(self, input_dim, num_classes=1, droprob = 0.3):
        super().__init__()
        
        self.dense_branch = nn.Sequential(
            DenseBlock(input_dim, 512, dropout = droprob),
            DenseBlock(512, 256, dropout = droprob)
        )
        
        self.tabnet = TabNetBlock(input_dim, 32, n_steps=2)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        self.final_dense = nn.Sequential(
            nn.Linear(416, 256),
            nn.ReLU(),
            nn.Dropout(droprob),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        dense_out = self.dense_branch(x)
        tab_out = self.tabnet(x)
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)
        
        combined = torch.cat([
            dense_out,
            tab_out[:, :32],
            lstm_out[:, :128]
        ], dim=1)

        return self.final_dense(combined)





class AdvancedCreditScoringModel_v1(nn.Module):
    def __init__(self, input_dim, num_classes=1, droprob = 0.3):
        super().__init__()
        
        # Transformer-based feature extraction
        self.transformer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=2,
            dim_feedforward=256,
            dropout=droprob
        )
        
        # Sequence modeling branch
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=droprob
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=2,
            dropout=droprob
        )
        
        # Feature extraction branch
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(droprob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Residual connection
        self.residual = nn.Linear(input_dim, 128)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 128(feature) + 256(transformer) + 128(lstm)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(droprob),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Transformer branch
        trans_out = self.transformer(x.unsqueeze(1)).squeeze(1)
        
        # LSTM branch
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]  # take last output
        
        # Attention branch
        att_out, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        att_out = att_out.squeeze(0)
        
        # Feature branch with residual
        feat_out = self.feature_net(x) + self.residual(x)
        
        # Combine all features
        combined = torch.cat([
            feat_out,
            lstm_out,
            att_out[:, :128]  # take part of attention output
        ], dim=1)
        
        return self.classifier(combined)




class AdvancedCreditScoringModel_v2(nn.Module):
    def __init__(self, input_dim, num_classes=1, droprob=0.3):
        super().__init__()
        
        # Transformer-based feature extraction
        self.transformer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=256,
            dropout=droprob
        )
        
        # Sequence modeling branch
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=droprob
        )
        
        # Feature extraction branch
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(droprob)
        )
        
        # Residual connection
        self.residual = nn.Linear(input_dim, 128)
        
        # Final classifier - adjusting input size to match concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 128(feature) + 256(lstm) + 128(transformer)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(droprob),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Transformer branch
        trans_out = self.transformer(x.unsqueeze(1)).squeeze(1)
        trans_out = trans_out[:, :128]  # берем первые 128 признаков
        
        # LSTM branch
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]  # take last output
        
        # Feature branch with residual
        feat_out = self.feature_net(x) + self.residual(x)
        
        # Combine all features
        combined = torch.cat([
            feat_out,      # 128
            lstm_out,      # 256 (bidirectional: 128 * 2)
            trans_out      # 128
        ], dim=1)         # Total: 512
        
        return self.classifier(combined)


class AdvancedCreditScoringModel_cat(nn.Module):
    def __init__(self, input_dim, cat_dims=None, num_classes=1, droprob=0.3, embedding_dim=16):
        super().__init__()
        
        self.cat_dims = cat_dims or {}  
        self.embedding_layers = nn.ModuleDict()
        
        self.total_embedding_dim = 0
        for feat_idx, num_categories in self.cat_dims.items():
            self.embedding_layers[str(feat_idx)] = nn.Embedding(
                num_embeddings=num_categories,
                embedding_dim=embedding_dim
            )
            self.total_embedding_dim += embedding_dim
            
        self.adjusted_input_dim = input_dim - len(self.cat_dims) + self.total_embedding_dim
        
        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.adjusted_input_dim,
            nhead=1,
            dim_feedforward=256,
            dropout=droprob
        )
        
        self.lstm = nn.LSTM(
            input_size=self.adjusted_input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=droprob
        )

        self.feature_net = nn.Sequential(
            nn.Linear(self.adjusted_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(droprob)
        )
        
        self.residual = nn.Linear(self.adjusted_input_dim, 128)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 128(feature) + 256(lstm) + 128(transformer)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(droprob),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def _process_categorical_features(self, x):
        continuous_features = x.clone()
        embedding_outputs = []
        
        for feat_idx in sorted(self.cat_dims.keys(), reverse=True):
            cat_feature = x[:, feat_idx].long()
            print(cat_feature[16])
            max_cat = self.cat_dims[feat_idx]
            print(max_cat)
            if torch.max(cat_feature) >= max_cat:
                raise ValueError(f"Categorical feature at index {feat_idx} contains values >= {max_cat}")
            embedded = self.embedding_layers[str(feat_idx)](cat_feature)
            embedding_outputs.append(embedded)
            continuous_features = torch.cat(
                [continuous_features[:, :feat_idx], 
                 continuous_features[:, feat_idx+1:]], 
                dim=1
            )
    
        if embedding_outputs:
            return torch.cat([continuous_features] + embedding_outputs, dim=1)
        return continuous_features

    def forward(self, x):
        x = self._process_categorical_features(x)
        
        trans_out = self.transformer(x.unsqueeze(1)).squeeze(1)
        
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]  

        combined = torch.cat([trans_out, lstm_out], dim=1)
        
        combined += self.residual(x)
        
        output = self.classifier(combined)
        return output