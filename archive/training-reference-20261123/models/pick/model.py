import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align


class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_text_len: int = 64,
        d_model: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.char_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, max_text_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.conv1x1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv_roi = nn.Conv2d(512, d_model, kernel_size=(7, 7))
        self.bn_roi = nn.BatchNorm2d(d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_boxes, text_len = input_ids.shape

        char_embeddings = self.char_embedding(input_ids)
        pos_embeddings = self.pos_embedding[:, :, :text_len, :]
        text_embeddings = self.dropout(char_embeddings + pos_embeddings)

        text_flat = text_embeddings.view(batch_size * num_boxes, text_len, self.d_model)
        mask_flat = attention_mask.view(batch_size * num_boxes, text_len)
        empty_boxes = mask_flat.all(dim=1)
        mask_flat = mask_flat.clone()
        mask_flat[empty_boxes, 0] = False
        text_features = self.transformer_encoder(text_flat, src_key_padding_mask=mask_flat)
        text_features = text_features.masked_fill(empty_boxes[:, None, None], 0.0)
        text_features = text_features.view(batch_size, num_boxes, text_len, self.d_model)

        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.conv1x1(x)

        roi_features = roi_align(feature_map, list(boxes), output_size=(7, 7), spatial_scale=1.0 / 16.0)
        image_features = F.relu(self.bn_roi(self.conv_roi(roi_features)))
        image_features = image_features.squeeze(-1).squeeze(-1).view(batch_size, num_boxes, 1, self.d_model)
        image_features = image_features.expand(-1, -1, text_len, -1)

        fused_features = self.layer_norm(text_features + image_features)
        fused_features = self.dropout(fused_features)

        char_mask = (~attention_mask).unsqueeze(-1).float()
        node_features = (fused_features * char_mask).sum(dim=2) / char_mask.sum(dim=2).clamp(min=1.0)

        return fused_features, node_features


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

        self.learn_w = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.learn_w, std=0.02)

        self.linear_alpha = nn.Linear(6, d_model)
        self.linear_vih = nn.Linear(d_model, d_model)
        self.linear_vjh = nn.Linear(d_model, d_model)
        self.linear_update = nn.Linear(d_model, d_model)

    def forward(
        self,
        node_features: torch.Tensor,
        box_mask: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_boxes = node_features.shape[1]

        node_diff = node_features.unsqueeze(2) - node_features.unsqueeze(1)
        abs_diff = torch.abs(node_diff)
        edge_logits = F.leaky_relu(torch.matmul(abs_diff, self.learn_w))

        inv_mask = ~box_mask.unsqueeze(1).expand(-1, num_boxes, -1)
        edge_logits = edge_logits.masked_fill(inv_mask, -1e4)
        soft_adj = F.softmax(edge_logits, dim=-1)

        relation_features = self.linear_alpha(relations)
        node_i = self.linear_vih(node_features).unsqueeze(2)
        node_j = self.linear_vjh(node_features).unsqueeze(1)
        transformed_features = F.relu(node_i + node_j + relation_features)

        aggregated_features = torch.sum(soft_adj.unsqueeze(-1) * transformed_features, dim=2)
        graph_features = F.relu(self.linear_update(aggregated_features))

        distance = torch.norm(node_diff.float(), dim=-1)
        loss_exponent = torch.clamp(soft_adj.float() + self.eta * distance, max=10.0)
        valid_pairs = (box_mask.unsqueeze(2) & box_mask.unsqueeze(1)).float()
        valid_counts = box_mask.float().sum(dim=-1).clamp(min=1.0)
        gl_loss1 = (torch.exp(loss_exponent) * valid_pairs).sum(dim=(-2, -1)) / (valid_counts ** 2)
        gl_loss2 = self.gamma * torch.sum((soft_adj * valid_pairs) ** 2, dim=(-2, -1))
        gl_loss = torch.mean(gl_loss1 + gl_loss2)

        return graph_features, gl_loss


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
        self.linear_union = nn.Linear(2 * d_model, d_model)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear_emissions = nn.Linear(2 * hidden_size, num_tags)

    def forward(
        self,
        fused_features: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_boxes, text_len, d_model = fused_features.shape

        graph_expanded = graph_features.unsqueeze(2).expand(-1, -1, text_len, -1)
        union_features = torch.cat([fused_features, graph_expanded], dim=-1)
        projected_features = self.linear_union(union_features)

        sequence_features = projected_features.view(batch_size * num_boxes, text_len, d_model)
        lstm_features, _ = self.bilstm(sequence_features)
        emissions = self.linear_emissions(lstm_features)

        return emissions.view(batch_size, num_boxes, text_len, -1)


class CRF(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        text_len, batch_size, _ = emissions.shape
        batch_idx = torch.arange(batch_size, device=emissions.device)
        score = self.start_transitions[tags[0]] + emissions[0, batch_idx, tags[0]]

        for i in range(1, text_len):
            transition_score = self.transitions[tags[i - 1], tags[i]]
            emission_score = emissions[i, batch_idx, tags[i]]
            score = score + (transition_score + emission_score) * mask[i]

        last_indices = torch.clamp(mask.long().sum(dim=0) - 1, min=0)
        last_tags = tags.gather(0, last_indices.unsqueeze(0)).squeeze(0)
        score = score + self.end_transitions[last_tags]
        return score

    def _forward_score(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        text_len = emissions.shape[0]
        score = self.start_transitions + emissions[0]

        for i in range(1, text_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        text_len, batch_size, _ = emissions.shape
        score = self.start_transitions + emissions[0]
        history: list[torch.Tensor] = []

        for i in range(1, text_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score = score + self.end_transitions
        best_tags_list: list[list[int]] = []

        for batch_idx in range(batch_size):
            text_end = int(mask[:, batch_idx].long().sum().item()) - 1
            if text_end < 0:
                best_tags_list.append([])
                continue

            best_tag = score[batch_idx].argmax().item()
            best_tags = [best_tag]
            for step_history in reversed(history[:text_end]):
                best_tag = step_history[batch_idx, best_tags[-1]].item()
                best_tags.append(best_tag)

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gold_score = self._gold_score(emissions, tags, mask)
        forward_score = self._forward_score(emissions, mask)
        return torch.mean(forward_score - gold_score)


class PICKModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        lambda_gl: float = 0.01,
    ) -> None:
        super().__init__()
        self.encoder = MultimodalEncoder(vocab_size=vocab_size)
        self.glcn = GLCN()
        self.decoder = BiLSTMDecoder(num_tags=num_tags)
        self.crf = CRF(num_tags=num_tags)
        self.lambda_gl = lambda_gl

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
        fused_features, node_features = self.encoder(images, boxes, input_ids, attention_mask)
        graph_features, gl_loss = self.glcn(node_features, box_mask, relations)
        emissions = self.decoder(fused_features, graph_features)

        char_mask = (~attention_mask) & box_mask.unsqueeze(-1)
        mask_flat = char_mask.view(batch_size * num_boxes, text_len).transpose(0, 1)
        emissions_flat = emissions.view(batch_size * num_boxes, text_len, -1).transpose(0, 1)

        total_loss = torch.tensor(0.0, device=images.device)
        if labels is not None:
            tags_flat = labels.view(batch_size * num_boxes, text_len).transpose(0, 1)
            box_indices = mask_flat[0].nonzero(as_tuple=True)[0]
            if box_indices.numel() > 0:
                crf_loss = self.crf(
                    emissions_flat[:, box_indices].float(),
                    tags_flat[:, box_indices],
                    mask_flat[:, box_indices],
                )
                total_loss = crf_loss + self.lambda_gl * gl_loss
            else:
                total_loss = self.lambda_gl * gl_loss

        preds = None
        if return_preds:
            decoded = self.crf.decode(emissions_flat, mask_flat)
            preds = [decoded[batch_idx * num_boxes : (batch_idx + 1) * num_boxes] for batch_idx in range(batch_size)]

        return total_loss, preds
