import torch
from torch import nn
from transformers import AutoTokenizer, BartModel
import numpy as np

class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=2000, model_name="facebook/bart-large", dropout=0.1):
        super(BartWithClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.bart = BartModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

class ImprovedHierarchicalBartClassifier(nn.Module):
    """
    State-of-the-art hierarchical BART classifier with proper hierarchical modeling
    """
    def __init__(self, num_parent_labels, num_child_labels, model_name="facebook/bart-large", 
                 dropout=0.1, parent_weight=0.3, fusion_dim=512, fusion_type="gated",
                 use_hierarchy_mask=True, hierarchy_penalty_weight=0.1, 
                 scheduled_sampling=False, noise_robustness=True):
        super().__init__()
        
        self.num_parent_labels = num_parent_labels
        self.num_child_labels = num_child_labels
        self.parent_weight = parent_weight
        self.fusion_type = fusion_type  # "gated", "attention", or "simple"
        self.use_hierarchy_mask = use_hierarchy_mask
        self.hierarchy_penalty_weight = hierarchy_penalty_weight
        self.scheduled_sampling = scheduled_sampling
        self.noise_robustness = noise_robustness
        
        # Shared BART backbone
        self.bart = BartModel.from_pretrained(model_name)
        self.hidden_size = self.bart.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Improved parent classifier with deeper architecture
        self.parent_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_parent_labels)
        )
        
        # Fusion mechanism based on type
        if fusion_type == "gated":
            # Gated fusion - learns how to combine features
            self.parent_gate = nn.Linear(num_parent_labels, self.hidden_size)
            self.feature_gate = nn.Linear(self.hidden_size, self.hidden_size)
            self.fusion_gate = nn.Sigmoid()
            child_input_dim = self.hidden_size
        elif fusion_type == "attention":
            # Attention-based fusion
            self.parent_projection = nn.Linear(num_parent_labels, self.hidden_size)
            self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=8, dropout=dropout, batch_first=True)
            child_input_dim = self.hidden_size
        else:  # "simple" - improved simple concatenation
            self.parent_projection = nn.Linear(num_parent_labels, fusion_dim // 4)
            child_input_dim = self.hidden_size + fusion_dim // 4
        
        # Improved child classifier with hierarchical awareness
        self.child_classifier = nn.Sequential(
            nn.Linear(child_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_child_labels)
        )
        
        # Hierarchical constraint matrix (will be populated based on actual hierarchy)
        if use_hierarchy_mask:
            self.register_buffer('hierarchy_mask', torch.ones(num_child_labels, num_parent_labels))
        
        # Scheduled sampling probability (starts at 1.0, decreases during training)
        self.register_buffer('sampling_prob', torch.tensor(1.0))
        
    def set_hierarchy_mask(self, parent_to_child_map):
        """
        Set the hierarchy mask based on parent-child relationships
        Args:
            parent_to_child_map: Dict mapping parent_idx -> list of child_idx
        """
        if not self.use_hierarchy_mask:
            return
            
        # Initialize mask to zeros (no connections)
        mask = torch.zeros(self.num_child_labels, self.num_parent_labels)
        
        # Set connections based on hierarchy
        for parent_idx, child_indices in parent_to_child_map.items():
            for child_idx in child_indices:
                if child_idx < self.num_child_labels and parent_idx < self.num_parent_labels:
                    mask[child_idx, parent_idx] = 1.0
        
        self.hierarchy_mask.data = mask
        
    def update_scheduled_sampling(self, epoch, total_epochs):
        """Update scheduled sampling probability"""
        if self.scheduled_sampling:
            # Linear decay from 1.0 to 0.0
            self.sampling_prob.data = torch.tensor(max(0.0, 1.0 - epoch / total_epochs))

    def forward(self, input_ids, attention_mask=None, mode="joint", parent_targets=None):
        """
        Improved forward pass with consistent training/inference and better fusion
        """
        # Get BART representations
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)
        
        result = {}
        
        # Parent predictions
        parent_logits = self.parent_classifier(cls_output)
        result['parent_logits'] = parent_logits
        
        if mode == "parent_only":
            return result
        
        # For child predictions - CONSISTENT between training and inference
        parent_probs = torch.sigmoid(parent_logits)
        
        # Optional: Scheduled sampling during training
        if self.training and self.scheduled_sampling and parent_targets is not None:
            # Gradually transition from ground truth to predictions
            use_ground_truth = torch.rand(1).item() < self.sampling_prob.item()
            if use_ground_truth:
                parent_input = parent_targets.float()
            else:
                parent_input = parent_probs
        else:
            parent_input = parent_probs
        
        # Add noise for robustness during training
        if self.training and self.noise_robustness:
            noise = torch.randn_like(parent_input) * 0.01
            parent_input = torch.clamp(parent_input + noise, 0.0, 1.0)
        
        # Fusion mechanism
        if self.fusion_type == "gated":
            # Gated fusion
            parent_gate_out = self.parent_gate(parent_input)
            feature_gate_out = self.feature_gate(cls_output)
            gate = self.fusion_gate(parent_gate_out + feature_gate_out)
            fused_features = gate * cls_output + (1 - gate) * parent_gate_out
        elif self.fusion_type == "attention":
            # Attention-based fusion
            parent_proj = self.parent_projection(parent_input).unsqueeze(1)  # [batch, 1, hidden]
            cls_expanded = cls_output.unsqueeze(1)  # [batch, 1, hidden]
            attended, _ = self.attention(parent_proj, cls_expanded, cls_expanded)
            fused_features = attended.squeeze(1)  # [batch, hidden]
        else:  # "simple"
            # Improved simple concatenation with projection
            parent_proj = self.parent_projection(parent_input)
            fused_features = torch.cat([cls_output, parent_proj], dim=-1)
        
        # Child predictions
        child_logits = self.child_classifier(fused_features)
        
        # Apply hierarchical constraints during inference
        if False:  # TEMPORARILY DISABLED - was: not self.training and self.use_hierarchy_mask
            child_logits = self._apply_hierarchical_constraints(child_logits, parent_probs)
        
        result['child_logits'] = child_logits
        return result
    
    def _apply_hierarchical_constraints(self, child_logits, parent_probs):
        """Apply hierarchical constraints safely"""
        if not hasattr(self, 'hierarchy_mask') or self.hierarchy_mask is None:
            return child_logits
            
        # Create constraint mask based on parent probabilities
        parent_mask = (parent_probs > 0.5).float()  # [batch, num_parents]
        constraint_mask = torch.matmul(parent_mask, self.hierarchy_mask.T)  # [batch, num_children]
        
        # Safer constraint application
        # Instead of extreme negative values, use a reasonable threshold
        min_logit = -10.0  # This gives probability ≈ 0.000045, which is fine
        constrained_logits = torch.where(
            constraint_mask > 0,
            child_logits,
            torch.clamp(child_logits, max=min_logit)  # Cap at reasonable negative value
        )
        return constrained_logits

    def _compute_hierarchy_penalty(self, parent_probs, child_probs, parent_targets, child_targets):
        """Compute penalty for violating hierarchical constraints - safer version"""
        if not hasattr(self, 'hierarchy_mask') or self.hierarchy_mask is None:
            return torch.tensor(0.0, device=parent_probs.device)
        
        try:
            # Penalty when child is predicted but parent is not
            parent_binary = (parent_probs > 0.5).float()
            child_binary = (child_probs > 0.5).float()
            
            # Expected parent support based on child predictions
            expected_parent = torch.matmul(child_binary, self.hierarchy_mask)  # [batch, num_parents]
            expected_parent = (expected_parent > 0).float()  # Binarize
            
            # Penalty for missing parent support - use MSE instead of relu for stability
            penalty = torch.mean((expected_parent - parent_binary) ** 2)
            return penalty
        except Exception as e:
            # If anything goes wrong, return zero penalty
            return torch.tensor(0.0, device=parent_probs.device)
    
    def compute_hierarchical_loss(self, parent_logits, child_logits, parent_targets, child_targets, criterion):
        """Improved hierarchical loss with consistency penalty"""
        # Compute losses - these will have shapes [batch, num_labels]
        parent_loss_tensor = criterion(parent_logits, parent_targets.float())  # [batch, 48]
        child_loss_tensor = criterion(child_logits, child_targets.float())     # [batch, 1884]
        
        # Reduce to scalars by taking mean
        parent_loss = parent_loss_tensor.mean()  # scalar
        child_loss = child_loss_tensor.mean()    # scalar
        
        # Hierarchical consistency loss
        hierarchy_penalty = torch.tensor(0.0, device=parent_logits.device)
        if self.use_hierarchy_mask and self.hierarchy_penalty_weight > 0:
            hierarchy_penalty = self._compute_hierarchy_penalty(
                torch.sigmoid(parent_logits), torch.sigmoid(child_logits), 
                parent_targets, child_targets
            )
        
        # Weighted combination - now all are scalars
        total_loss = (self.parent_weight * parent_loss + 
                     (1 - self.parent_weight) * child_loss + 
                     self.hierarchy_penalty_weight * hierarchy_penalty)
        
        return {
            'total_loss': total_loss,
            'parent_loss': parent_loss,
            'child_loss': child_loss,
            'hierarchy_penalty': hierarchy_penalty
        }
    
    def _compute_hierarchy_penalty(self, parent_probs, child_probs, parent_targets, child_targets):
        """Compute penalty for violating hierarchical constraints - safer version"""
        if not hasattr(self, 'hierarchy_mask') or self.hierarchy_mask is None:
            return torch.tensor(0.0, device=parent_probs.device)
        
        try:
            # Penalty when child is predicted but parent is not
            parent_binary = (parent_probs > 0.5).float()
            child_binary = (child_probs > 0.5).float()
            
            # Expected parent support based on child predictions
            expected_parent = torch.matmul(child_binary, self.hierarchy_mask)  # [batch, num_parents]
            expected_parent = (expected_parent > 0).float()  # Binarize
            
            # Penalty for missing parent support - use MSE instead of relu for stability
            penalty = torch.mean((expected_parent - parent_binary) ** 2)
            return penalty
        except Exception as e:
            # If anything goes wrong, return zero penalty
            return torch.tensor(0.0, device=parent_probs.device)

# Keep backward compatibility
HierarchicalBartClassifier = ImprovedHierarchicalBartClassifier

def get_model(model_type="bart_classifier", num_labels=2000, **kwargs):
    """Factory function to get different model types"""
    if model_type == "bart_classifier":
        return BartWithClassifier(num_labels=num_labels, **kwargs)
    elif model_type == "hierarchical_bart":
        # Extract hierarchical-specific parameters
        num_parent_labels = kwargs.pop('num_parent_labels', 100)
        num_child_labels = kwargs.pop('num_child_labels', num_labels)
        return ImprovedHierarchicalBartClassifier(
            num_parent_labels=num_parent_labels,
            num_child_labels=num_child_labels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def compute_hierarchical_loss_adaptive(self, parent_logits, child_logits, parent_targets, child_targets, criterion):
    """
    Adaptive hierarchical loss that focuses on MCC improvement
    """
    # Sample-wise losses for better control
    if hasattr(criterion, 'reduction') and criterion.reduction != 'none':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    parent_losses = criterion(parent_logits, parent_targets.float())  # [batch, num_parents]
    child_losses = criterion(child_logits, child_targets.float())    # [batch, num_children]
    
    # Average across labels for each sample
    parent_loss_per_sample = parent_losses.mean(dim=1)  # [batch]
    child_loss_per_sample = child_losses.mean(dim=1)    # [batch]
    
    # Hierarchical consistency penalty
    hierarchy_penalty = torch.tensor(0.0, device=parent_logits.device)
    if self.use_hierarchy_mask and self.hierarchy_penalty_weight > 0:
        hierarchy_penalty = self._compute_hierarchy_penalty_improved(
            torch.sigmoid(parent_logits), torch.sigmoid(child_logits), 
            parent_targets, child_targets
        )
    
    # Focus loss: Give more weight to samples with hierarchical violations
    parent_probs = torch.sigmoid(parent_logits)
    child_probs = torch.sigmoid(child_logits)
    
    # Detect hierarchical violations (child predicted but no parent)
    if hasattr(self, 'hierarchy_mask') and self.hierarchy_mask is not None:
        parent_support = torch.matmul(child_probs, self.hierarchy_mask)  # Expected parent support
        actual_parent = parent_probs.sum(dim=1, keepdim=True)
        violation_weight = torch.relu(parent_support.sum(dim=1) - actual_parent.squeeze()) + 1.0
    else:
        violation_weight = torch.ones_like(parent_loss_per_sample)
    
    # Weighted combination with violation focus - reduce to scalars
    weighted_parent_loss = (parent_loss_per_sample * violation_weight).mean()
    weighted_child_loss = (child_loss_per_sample * violation_weight).mean()
    
    total_loss = (self.parent_weight * weighted_parent_loss + 
                 (1 - self.parent_weight) * weighted_child_loss + 
                 self.hierarchy_penalty_weight * hierarchy_penalty)
    
    return {
        'total_loss': total_loss,
        'parent_loss': weighted_parent_loss,
        'child_loss': weighted_child_loss,
        'hierarchy_penalty': hierarchy_penalty
    }

def _compute_hierarchy_penalty_improved(self, parent_probs, child_probs, parent_targets, child_targets):
    """
    Improved hierarchy penalty focusing on MCC improvement
    """
    if not hasattr(self, 'hierarchy_mask') or self.hierarchy_mask is None:
        return torch.tensor(0.0, device=parent_probs.device)
    
    # Focus on samples where child is predicted
    child_active = (child_probs > 0.5).float()
    parent_active = (parent_probs > 0.5).float()
    
    # Expected parent support
    expected_parent = torch.matmul(child_active, self.hierarchy_mask)  # [batch, num_parents]
    expected_parent = (expected_parent > 0).float()
    
    # Soft penalty using Focal Loss style
    penalty = torch.pow(expected_parent - parent_active, 2) * expected_parent
    return penalty.mean()