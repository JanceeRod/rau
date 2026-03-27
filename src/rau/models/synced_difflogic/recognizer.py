"""
rau-compatible wrapper around SyncedRecurrentDiffLogicModel for formal
language recognition.

The inner DiffLogic model uses GroupSum as its final layer (preserved
intact). A set of linear task heads sits on top of the GroupSum output:

    recognition_head  : Linear(hidden_units, 1)      – always present
    lm_head           : Linear(hidden_units, vocab)   – optional
    next_symbols_head : Linear(hidden_units, vocab)   – optional

To match the rau include_first=True convention used by RNN/LSTM, the
forward pass prepends a position-0 output computed from the initial hidden
state (no token input), so the returned tensor has shape
[batch, seq_len+1, hidden_units].  This means last_index (raw sequence
length N) correctly selects the hidden state after the last real token, and
language-modeling targets (full_tensor, shape [batch, N+1]) align directly.

Constraint: embedding_dim <= 2 * hidden_units
  (required by the LogicLayer 2*out_dim >= in_dim rule for N layers)

Import note: SyncedRecurrentDiffLogicModel lives in recurrent_difflogic/src/
which must be on sys.path.  This file adds it automatically via a relative
path calculation based on the known project layout:
  <project_root>/rau/src/rau/models/synced_difflogic/recognizer.py
  <project_root>/recurrent_difflogic/src/
"""

import os
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure recurrent_difflogic/src/ is importable.
# This file is 5 levels deep inside the project root, so going up 5 dirs
# leads to <project_root>, and recurrent_difflogic/src/ is a sibling of rau/.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RD_SRC = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', '..', '..', '..', 'recurrent_difflogic', 'src'))
if _RD_SRC not in sys.path:
    sys.path.insert(0, _RD_SRC)

from models.synced_recurrent_difflogic import SyncedRecurrentDiffLogicModel  # noqa: E402


class SyncedDiffLogicRecognizer(nn.Module):
    """rau-compatible recognizer built on SyncedRecurrentDiffLogicModel.

    Layer sizing (all uniform at hidden_units):
        N layers: [hidden_units] * num_layers    (process input at each step)
        K layers: [hidden_units] * num_layers    (recurrent hidden state update)
        M layers: [hidden_units] * num_layers    passed in; constructor appends
                  one final layer of size hidden_units*group_factor, so the last
                  M layer has out_dim=hidden_units*group_factor.
        GroupSum:  k=hidden_units, group_factor=group_factor
                   With group_factor=2 (default) each output feature is the sum
                   of 2 neurons, giving GroupSum a meaningful aggregation role.
                   group_factor=1 makes GroupSum an identity (no aggregation).

    The per-timestep GroupSum output (shape [batch, seq_len, hidden_units])
    serves as the feature representation fed to the task heads.
    """

    def __init__(
        self,
        num_input_tokens: int,
        embedding_dim: int,
        hidden_units: int,
        num_layers: int,
        dropout: float,
        use_language_modeling_head: bool = False,
        use_next_symbols_head: bool = False,
        output_vocabulary_size: int = None,
        device: str = None,   # None → auto-detect CUDA; else 'cuda' or 'cpu'
        difflogic_init_type: str = 'noisy_residual',
        hidden_state_init_type: str = 'zero',
        connections: str = 'random',
        group_factor: int = 2,
    ):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if embedding_dim > 2 * hidden_units:
            raise ValueError(
                f'embedding_dim ({embedding_dim}) must be <= 2 * hidden_units '
                f'({2 * hidden_units}) to satisfy the LogicLayer dimension '
                f'constraint (2 * out_dim >= in_dim) for N layers.'
            )

        # Uniform layer sizes for all layer groups.
        # The SyncedRecurrentDiffLogicModel constructor asserts len(m_layers_sizes) > 0
        # before appending the final output layer (num_classes * group_factor), so
        # m_layers_sizes must have at least one entry.  Using num_layers entries for
        # each group keeps all groups symmetric.
        n_layers_sizes = [hidden_units] * num_layers
        k_layers_sizes = [hidden_units] * num_layers
        m_layers_sizes = [hidden_units] * num_layers

        self.inner = SyncedRecurrentDiffLogicModel(
            num_input_tokens=num_input_tokens,
            embedding_dim=embedding_dim,
            seq_length=0,          # stored but unused in forward()
            n_layers_sizes=n_layers_sizes,
            k_layers_sizes=k_layers_sizes,
            m_layers_sizes=m_layers_sizes,
            num_classes=hidden_units,
            group_factor=group_factor,
            device=device,
            dropout_prob=dropout,
            difflogic_init_type=difflogic_init_type,
            hidden_state_init_type=hidden_state_init_type,
            connections=connections,
            # The rau pipeline has no dedicated PAD token: index 0 is a real
            # content token and padding fill values are replaced with 0 before
            # reaching the model (see model_interface.prepare_batch).  Setting
            # padding_idx=None prevents nn.Embedding from zeroing and freezing
            # the embedding for index 0, matching the RNN/LSTM behaviour
            # (use_padding=False in EmbeddingUnidirectional).
            padding_idx=None,
        )

        self.hidden_units = hidden_units

        # Task-specific linear heads on top of GroupSum features.
        self.recognition_head = nn.Linear(hidden_units, 1)
        if use_language_modeling_head:
            if output_vocabulary_size is None:
                raise ValueError(
                    'output_vocabulary_size must be provided when '
                    'use_language_modeling_head=True'
                )
            self.lm_head = nn.Linear(hidden_units, output_vocabulary_size)
        else:
            self.lm_head = None
        if use_next_symbols_head:
            if output_vocabulary_size is None:
                raise ValueError(
                    'output_vocabulary_size must be provided when '
                    'use_next_symbols_head=True'
                )
            self.next_symbols_head = nn.Linear(hidden_units, output_vocabulary_size)
        else:
            self.next_symbols_head = None

    def train(self, mode: bool = True):
        """Keep the inner DiffLogic model's logic layers soft during both
        training and rau validation.

        The rau pipeline calls model.train() / model.eval() to toggle dropout.
        Without this override, model.eval() would propagate into self.inner and
        put its LogicLayer instances into discrete (argmax) mode, causing a
        train/eval mismatch while the embedding remains a soft sigmoid.

        Instead we always call self.inner.set_mode('train') or
        self.inner.set_mode('eval'), which keeps logic layers soft in both
        cases (matching the original author's validation behaviour) while still
        toggling dropout correctly on all other modules (task heads, etc.)
        via super().train(mode).
        """
        super().train(mode)
        self.inner.set_mode('train' if mode else 'eval')
        return self

    def _compute_initial_output(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute the M-layer + GroupSum output for the initial state.

        Uses zeros for the N-layer contribution (no token input yet) and
        the model's initial hidden state for the K-layer contribution.
        This produces a [batch, hidden_units] tensor analogous to the
        initial hidden state output in rau's include_first=True convention.
        """
        initial_n_out = torch.zeros(
            batch_size, self.inner.n_out,
            device=device,
            dtype=torch.float
        )
        initial_hidden = self.inner._init_hidden(batch_size).to(
            device=device, dtype=torch.float
        )
        combined = torch.cat([initial_n_out, initial_hidden], dim=1)

        m_out = combined
        for i, layer in enumerate(self.inner.m_layers):
            m_out = layer(m_out)
            m_out = self.inner.m_dropouts[i](m_out)

        return self.inner.final_sum(m_out)  # [batch, hidden_units]

    def forward(
        self,
        x: torch.Tensor,
        last_index: torch.Tensor = None,
        positive_mask: torch.Tensor = None,
    ):
        """Run the model and return task logits.

        Args:
            x: Integer token ids, shape [batch, seq_len].
            last_index: 0-D index of the last non-padding token for each
                example, shape [batch].  Equal to the raw sequence length N
                (as returned by pad_sequences with return_lengths=True).
            positive_mask: Boolean mask of positive examples, shape [batch].

        Returns:
            Tuple (recognition_logit, lm_logits, next_symbols_logits).
                recognition_logit : [batch]
                lm_logits          : [pos_batch, seq_len+1, vocab_size] or None
                next_symbols_logits: [pos_batch, seq_len+1, vocab_size] or None
        """
        batch_size = x.size(0)

        # Inner model: [batch, seq_len, hidden_units] (post-GroupSum features)
        # binary_reg_loss = mean(sigmoid(emb) * (1 - sigmoid(emb))) encourages
        # binary activations; cache it so the training loop can include it.
        outputs, binary_reg_loss = self.inner(x)
        self._last_binary_reg_loss = binary_reg_loss

        # Prepend position-0 output (initial state, no token processed yet).
        # This gives seq_len+1 positions, matching the include_first=True
        # convention: position j holds the representation AFTER processing
        # the j-th token (with position 0 = before any token = initial state).
        # Consequently last_index=N correctly selects the output after tN.
        initial_out = self._compute_initial_output(batch_size, x.device)
        outputs = torch.cat([initial_out.unsqueeze(1), outputs], dim=1)
        # outputs: [batch, seq_len+1, hidden_units]

        # --- Recognition head ---
        # Gather the representation at last_index for each example.
        last_hidden = torch.gather(
            outputs,
            1,
            last_index[:, None, None].expand(-1, -1, outputs.size(2))
        ).squeeze(1)  # [batch, hidden_units]
        recognition_logit = self.recognition_head(last_hidden).squeeze(1)  # [batch]

        # --- Optional LM and next-symbols heads ---
        lm_logits = None
        next_symbols_logits = None
        if self.lm_head is not None or self.next_symbols_head is not None:
            positive_outputs = outputs[positive_mask]  # [pos_batch, seq_len+1, hidden_units]
            if self.lm_head is not None:
                lm_logits = self.lm_head(positive_outputs)  # [pos_batch, seq_len+1, vocab_size]
            if self.next_symbols_head is not None:
                next_symbols_logits = self.next_symbols_head(positive_outputs)

        return recognition_logit, lm_logits, next_symbols_logits
