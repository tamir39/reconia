import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align


class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_text_len: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.conv_reduce = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv_roi = nn.Conv2d(512, d_model, kernel_size=(7, 7))
        self.bn_roi = nn.BatchNorm2d(d_model)

        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, max_text_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pe_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_boxes, text_len = input_ids.shape

        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.conv_reduce(x)

        boxes_list = [boxes[batch_idx] for batch_idx in range(batch_size)]
        rois = roi_align(feat, boxes_list, output_size=(7, 7), spatial_scale=1.0 / 16.0)
        image_emb = F.relu(self.bn_roi(self.conv_roi(rois)))
        image_emb = image_emb.squeeze(-1).squeeze(-1).view(batch_size, num_boxes, 1, self.d_model)
        image_emb = image_emb.expand(-1, -1, text_len, -1)

        char_emb = self.char_embed(input_ids)
        pos_emb = self.pos_embed[:, :, :text_len, :]
        text_emb = self.pe_dropout(char_emb + pos_emb)

        text_flat = text_emb.view(batch_size * num_boxes, text_len, self.d_model)
        mask_flat = attention_mask.view(batch_size * num_boxes, text_len)
        text_out = self.transformer(text_flat, src_key_padding_mask=mask_flat)
        text_out = text_out.view(batch_size, num_boxes, text_len, self.d_model)

        fusion = self.layer_norm(text_out + image_emb)
        char_features = self.output_dropout(fusion)

        valid_chars = (~attention_mask).unsqueeze(-1).float()
        node_features = (char_features * valid_chars).sum(dim=2) / valid_chars.sum(dim=2).clamp(min=1.0)

        return char_features, node_features


class GLCN(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        eta: float = 1.0,
        gamma: float = 0.4,
    ) -> None:
        super().__init__()
        self.eta = eta
        self.gamma = gamma

        self.weight_gl = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.weight_gl, std=0.02)

        self.linear_alpha = nn.Linear(6, d_model)
        self.linear_vih = nn.Linear(d_model, d_model)
        self.linear_vjh = nn.Linear(d_model, d_model)
        self.linear_update = nn.Linear(d_model, d_model)

    def forward(
        self,
        node_features: torch.Tensor,
        relations: torch.Tensor,
        box_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_boxes = node_features.shape[1]

        diff = torch.abs(node_features.unsqueeze(2) - node_features.unsqueeze(1))
        edge_logits = F.leaky_relu(torch.matmul(diff, self.weight_gl))

        inv_mask = ~box_mask.unsqueeze(1).expand(-1, num_boxes, -1)
        edge_logits = edge_logits.masked_fill(inv_mask, -1e4)
        adj_matrix = F.softmax(edge_logits, dim=-1)

        dist = torch.norm(node_features.unsqueeze(2) - node_features.unsqueeze(1), dim=-1)
        dist_loss = torch.clamp(adj_matrix + self.eta * dist, max=10.0)
        valid_pairs = (box_mask.unsqueeze(2) & box_mask.unsqueeze(1)).float()
        valid_counts = box_mask.float().sum(dim=-1).clamp(min=1.0)
        loss_gl1 = (torch.exp(dist_loss) * valid_pairs).sum(dim=(-2, -1)) / (valid_counts ** 2)
        loss_gl2 = self.gamma * torch.sum((adj_matrix * valid_pairs) ** 2, dim=(-2, -1))
        loss_gl = torch.mean(loss_gl1 + loss_gl2)

        alpha0 = self.linear_alpha(relations)
        vih = self.linear_vih(node_features).unsqueeze(2)
        vjh = self.linear_vjh(node_features).unsqueeze(1)
        transformed_features = F.relu(vih + vjh + alpha0)

        features_agg = torch.sum(adj_matrix.unsqueeze(-1) * transformed_features, dim=2)
        graph_features = F.relu(self.linear_update(features_agg))

        return graph_features, loss_gl


class BiLSTMDecoder(nn.Module):
    def __init__(
        self,
        num_tags: int,
        d_model: int = 512,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear_union = nn.Linear(2 * d_model, d_model)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_emissions = nn.Linear(2 * hidden_size, num_tags)

    def forward(
        self,
        char_features: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_boxes, text_len, d_model = char_features.shape
        graph_expanded = graph_features.unsqueeze(2).expand(-1, -1, text_len, -1)
        fused = self.linear_union(torch.cat([char_features, graph_expanded], dim=-1))

        packed_features = fused.view(batch_size * num_boxes, text_len, d_model)
        lstm_out, _ = self.bilstm(packed_features)
        emissions = self.fc_emissions(lstm_out)
        return emissions.view(batch_size, num_boxes, text_len, -1)


class CRF(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = emissions.shape
        batch_idx = torch.arange(batch_size, device=emissions.device)
        score = self.start_transitions[tags[0]] + emissions[0, batch_idx, tags[0]]
        for i in range(1, seq_len):
            trans_score = self.transitions[tags[i - 1], tags[i]]
            emit_score = emissions[i, batch_idx, tags[i]]
            score = score + (trans_score + emit_score) * mask[i]
        last_indices = torch.clamp(mask.long().sum(dim=0) - 1, min=0)
        last_tags = tags.gather(0, last_indices.unsqueeze(0)).squeeze(0)
        score = score + self.end_transitions[last_tags]
        return score

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = emissions.shape[0]
        score = self.start_transitions + emissions[0]
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        seq_len, batch_size, _ = emissions.shape
        score = self.start_transitions + emissions[0]
        history = []
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
        score = score + self.end_transitions
        best_tags_list = []
        for batch_idx in range(batch_size):
            seq_end = int(mask[:, batch_idx].long().sum().item()) - 1
            if seq_end < 0:
                best_tags_list.append([])
                continue
            best_last_tag = score[batch_idx].argmax().item()
            best_tags = [best_last_tag]
            for step_history in reversed(history[:seq_end]):
                best_last_tag = step_history[batch_idx, best_tags[-1]].item()
                best_tags.append(best_last_tag)
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gold = self._gold_score(emissions, tags, mask)
        forward_score = self._forward_alg(emissions, mask)
        return torch.mean(forward_score - gold)


class PICKModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_text_len: int = 64,
        lambda_gl: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_gl = lambda_gl

        self.encoder = MultimodalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_text_len=max_text_len,
        )
        self.glcn = GLCN(d_model=d_model)
        self.decoder = BiLSTMDecoder(
            num_tags=num_tags,
            d_model=d_model,
            hidden_size=d_model,
            dropout=dropout,
        )
        self.crf = CRF(num_tags=num_tags)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        box_mask: torch.Tensor,
        relations: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_preds: bool = True,
    ) -> tuple[torch.Tensor, list[list[list[int]]] | None]:
        batch_size, num_boxes, text_len = input_ids.shape
        char_features, node_features = self.encoder(images, boxes, input_ids, attention_mask)
        graph_features, loss_gl = self.glcn(node_features, relations, box_mask)
        emissions = self.decoder(char_features, graph_features)

        valid_char_mask = (~attention_mask) & box_mask.unsqueeze(-1)
        mask_flat = valid_char_mask.view(batch_size * num_boxes, text_len).transpose(0, 1)
        emissions_flat = emissions.view(batch_size * num_boxes, text_len, -1).transpose(0, 1)

        total_loss = torch.tensor(0.0, device=images.device)
        if labels is not None:
            labels_flat = labels.view(batch_size * num_boxes, text_len).transpose(0, 1)
            active_box_indices = mask_flat[0].nonzero(as_tuple=True)[0]
            if active_box_indices.numel() > 0:
                active_emissions = emissions_flat[:, active_box_indices]
                active_labels = labels_flat[:, active_box_indices]
                active_mask = mask_flat[:, active_box_indices]
                crf_loss = self.crf(active_emissions, active_labels, active_mask)
                total_loss = crf_loss + self.lambda_gl * loss_gl
            else:
                total_loss = self.lambda_gl * loss_gl

        preds = None
        if return_preds:
            decoded = self.crf.decode(emissions_flat, mask_flat)
            preds = [decoded[batch_idx * num_boxes : (batch_idx + 1) * num_boxes] for batch_idx in range(batch_size)]

        return total_loss, preds